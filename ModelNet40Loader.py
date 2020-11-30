import torch
import torch.utils.data as data
import numpy as np
import random
import time
import os, sys, h5py, subprocess, shlex
import ctypes 
from ctypes import *

BASE_DIR = "./datasets/"

class idx(Structure):
    _fields_ = [("idx1", c_int), ("idx2", c_int)] 

def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]


def _load_data_file(name):
    f = h5py.File(name)
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    return data, normal, label

def rotate_point_cloud(points, normals):
    rotation_angle = np.random.uniform(0, 1) * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    
    rotated_data = np.dot(points.reshape((-1, 3)), rotation_matrix)
    rotated_normal = np.dot(normals.reshape((-1, 3)), rotation_matrix)
    return rotated_data, rotated_normal

def rotate_point_cloud_so3(points, normals):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotation_angle_A = np.random.uniform() * 2 * np.pi
    rotation_angle_B = np.random.uniform() * 2 * np.pi
    rotation_angle_C = np.random.uniform() * 2 * np.pi

    cosval_A = np.cos(rotation_angle_A)
    sinval_A = np.sin(rotation_angle_A)
    cosval_B = np.cos(rotation_angle_B)
    sinval_B = np.sin(rotation_angle_B)
    cosval_C = np.cos(rotation_angle_C)
    sinval_C = np.sin(rotation_angle_C)
    rotation_matrix = np.array([[cosval_B*cosval_C, -cosval_B*sinval_C, sinval_B],
                                [sinval_A*sinval_B*cosval_C+cosval_A*sinval_C, -sinval_A*sinval_B*sinval_C+cosval_A*cosval_C, -sinval_A*cosval_B],
                                [-cosval_A*sinval_B*cosval_C+sinval_A*sinval_C, cosval_A*sinval_B*sinval_C+sinval_A*cosval_C, cosval_A*cosval_B]])
    rotated_data = np.dot(points, rotation_matrix)
    rotated_normal = np.dot(normals, rotation_matrix)
    return rotated_data, rotated_normal

def jitter_point_cloud(data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    N, C = data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += data
    return jittered_data

def scale_point_cloud(points, normals, low=0.8, high=1.2):
    scale = np.random.uniform(low=0.8, high=1.2)
    points = points * scale
    normals = normals * scale
    return points, normals

def translate_pointcloud(pointcloud, normal):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    translated_normal = np.multiply(normal, xyz1).astype('float32')

    return translated_pointcloud, translated_normal

def batch_knn_search(x):
    inner = 2 * np.matmul(x, x.transpose(1, 0))
    xx = np.sum(x**2, axis=-1, keepdims=True)
    pairwise_distance = xx - inner + xx.transpose(1, 0)

    return pairwise_distance

class ModelNet40Cls(data.Dataset):

    def __init__(self, partition='train', d_a=True, SO3=False):
        super().__init__()
        
        if not os.path.exists(BASE_DIR):
            os.makedirs(BASE_DIR)
        self.folder = "modelnet40_ply_hdf5_2048"
        self.data_dir = os.path.join(BASE_DIR, self.folder)
        self.url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
        
        if not os.path.exists(self.data_dir):
            zipfile = os.path.basename(self.url)
            os.system('wget --no-check-certificate %s; unzip %s' % (self.url, zipfile))
   
        self.partition = partition
        self.d_a = d_a
        self.SO3 = SO3
        
        if self.partition == 'train':
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'train_files.txt'))
        else:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'test_files.txt'))

        point_list, label_list, normal_list = [], [], []
        for f in self.files:
            points, normals, labels = _load_data_file(os.path.join(BASE_DIR, f))
            point_list.append(points)
            normal_list.append(normals)
            label_list.append(labels)

        if self.partition == 'train':
            shuffle_list = list(zip(point_list, normal_list, label_list)) 
            random.shuffle(shuffle_list)
            point_list, normal_list, label_list = zip(*shuffle_list)

        self.data = np.concatenate(point_list, 0)
        self.normal = np.concatenate(normal_list, 0)
        self.label = np.concatenate(label_list, 0)
    
    def __getitem__(self, item):
        
        if self.partition == 'train':
            pt_idxs = np.random.choice(2048, 1024, replace=False)
            current_points = self.data[item, pt_idxs, :].copy()
            current_normals = self.normal[item, pt_idxs, :].copy()
        else:
            current_points = self.data[item, :1024, :].copy()
            current_normals = self.normal[item, :1024, :].copy()
        label = self.label[item].copy()
        
        if self.d_a:
            current_points, current_normals = rotate_point_cloud(current_points, current_normals)
            current_points = jitter_point_cloud(current_points)
            current_normals = jitter_point_cloud(current_normals)
            current_points, current_normals = scale_point_cloud(current_points, current_normals, low=0.8, high=1.2)
        
        if self.SO3:
            current_points, current_normals = rotate_point_cloud_so3(current_points, current_normals)
        
        current_points = current_points - np.mean(current_points, axis=0, keepdims=True)
        current_normals = self.denoise(current_normals, current_points)

        current_points = torch.from_numpy(current_points).type(torch.FloatTensor).permute(1, 0)
        current_normals = torch.from_numpy(current_normals).type(torch.FloatTensor).permute(1, 0)
        label = torch.from_numpy(np.array(label)).type(torch.LongTensor)
        return current_points, current_normals, label
 
    def __len__(self):
        return self.data.shape[0]

    def knn_search(self, d_matrix, points, normals, knn):
        knn_idx = np.argsort(d_matrix, axis=-1)[:, 1:knn + 1]
        points_knn = points[knn_idx]
        normals_knn = normals[knn_idx]
        return points_knn, normals_knn, knn_idx
        
    def denoise(self, normals, mid):
        angle = np.sum(normals * mid, axis=-1) / (np.linalg.norm(normals, ord=2, axis=-1) * np.linalg.norm(mid, ord=2, axis=-1) + 1e-6)
        for i in range(normals.shape[0]):
            if angle[i] <=0:
                normals[i] = -normals[i]
        return normals
    
    
if __name__ == "__main__":
    from torchvision import transforms
    import data_utils as d_utils

    transforms = transforms.Compose([
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRotate(axis=np.array([1, 0, 0])),
        d_utils.PointcloudScale(),
        d_utils.PointcloudTranslate(),
        d_utils.PointcloudJitter()
    ])
    dset = ModelNet40Cls(16, "./", train=True, transforms=transforms)
    print(dset[0][0])
    print(dset[0][1])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
