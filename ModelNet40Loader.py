import torch
import torch.utils.data as data
import numpy as np
import random
import time
import os, sys, h5py, subprocess, shlex
import ctypes 
from ctypes import *

BASE_DIR = "/home/chen/code/dataset/"

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

        self.folder = "modelnet40_ply_hdf5_2048"
        self.data_dir = os.path.join(BASE_DIR, self.folder)
        self.url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"


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
        '''
        pt_idxs = np.random.choice(2048, 1024, replace=False)
        current_points = self.data[item, pt_idxs, :].copy()
        current_normals = self.normal[item, pt_idxs, :].copy()
        '''
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
    '''    
    def __getitem__(self, idx):
        pt_idxs = np.random.choice(2048, 1024, replace=False)
        current_points = self.points[idx, pt_idxs, :].copy()
        current_normals = self.normals[idx, pt_idxs, :].copy()
    #    current_points = self.points[idx, :1024, :].copy()
    #    current_normals = self.normals[idx, :1024, :].copy()

        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)        
                
        if self.d_a:
            current_points = jitter_point_cloud(current_points)
            current_normals = jitter_point_cloud(current_normals)
            current_points, current_normals = scale_point_cloud(current_points, current_normals, low=0.8, high=1.2)
        #    current_points, current_normals = translate_pointcloud(current_points, current_normals)
        
        if self.rotation:
            current_points, current_normals = rotate_point_cloud(current_points, current_normals)

        mean = np.mean(current_points, axis=0, keepdims=True)
        relative_points = current_points - mean   
        
        current_normals = self.denoise(current_normals, relative_points)

    #    d_matrix = np.array([np.expand_dims(current_points[i, :], axis=0) - current_points for i in range(current_points.shape[0])])
    #    d_matrix = np.linalg.norm(d_matrix, ord=2, axis=-1)
        d_matrix = batch_knn_search(current_points)

    #    points = self.spatial_f_c(current_points)
        points = self.spatial_f(current_points)

        points_knn, normals_knn, knn_idx = self.knn_search(d_matrix, current_points, current_normals, 32)

        features = self.darboux(current_points, current_normals, points_knn, normals_knn)

        points = torch.from_numpy(points).type(torch.FloatTensor)  
        features = torch.from_numpy(features).type(torch.FloatTensor)
        knn_idx = torch.from_numpy(knn_idx).type(torch.LongTensor)   

        return points, features, knn_idx, label
    '''   
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
    
    def spatial_f(self, points):
        mean = np.mean(points, axis=0, keepdims=True)
        p0 = points - mean
        d0 = np.linalg.norm(p0, ord=2, axis=-1, keepdims=True)
        
        idx1 = np.argsort(-d0, axis=0)[0]
        p1 = points - points[idx1]
        d1 = np.linalg.norm(p1, ord=2, axis=-1, keepdims=True)

        d = d0 + d1
        i = 0
        
        while 1:
            idx2 = np.argsort(-d, axis=0)[0]
            vec1 = points[idx1] - mean
            vec2 = points[idx2] - mean
            
            angle = np.sum(vec1 * vec2, axis=-1) / (np.linalg.norm(vec1, ord=2, axis=-1) * np.linalg.norm(vec2, ord=2, axis=-1))
            p2 = points - points[idx2]        
            d2 = np.linalg.norm(p2, ord=2, axis=-1, keepdims=True)
            d = d + d2

            d[idx1] = 0
            d[idx2] = 0

            if np.abs(angle) > 0.999:
                if i>=100:
                    print('Its a line!!')
                    break
                else:
                    i = i + 1   
                    continue
            else:
                break
        
        vec3 = np.cross(vec1, vec2)
        vec4 = np.cross(vec2, vec3)

        x = np.sum(p0 * vec2, axis=-1) / (np.linalg.norm(vec2, ord=2, axis=-1))
        y = np.sum(p0 * vec3, axis=-1) / (np.linalg.norm(vec3, ord=2, axis=-1))
        z = np.sum(p0 * vec4, axis=-1) / (np.linalg.norm(vec4, ord=2, axis=-1))
        
        x = np.expand_dims(x, axis=-1)    
        y = np.expand_dims(y, axis=-1)
        z = np.expand_dims(z, axis=-1)

    #    P = np.concatenate([x, y, z], axis=-1)
   
        return np.concatenate([x, y, z], axis=-1).transpose(1, 0)

    def spatial_f_c(self, points):
        points = np.asarray(points, dtype = np.float32)
        points_c = points.ctypes.data_as(ctypes.c_char_p) 

        so = ctypes.cdll.LoadLibrary   
    
        lib = so("./utils/spatial_f.so") 
        idx_c = idx()

        lib.spa_f(points_c, ctypes.byref(idx_c))

        idx1 = np.array(idx_c.idx1)
        idx2 = np.array(idx_c.idx2)

        mean = np.mean(points, axis=0, keepdims=True)
        p0 = points - mean

        vec1 = points[int(idx1)] - mean
        vec2 = points[int(idx2)] - mean

        vec3 = np.cross(vec1, vec2)
        vec4 = np.cross(vec2, vec3)

        x = np.sum(p0 * vec2, axis=-1) / (np.linalg.norm(vec2, ord=2, axis=-1))
        y = np.sum(p0 * vec3, axis=-1) / (np.linalg.norm(vec3, ord=2, axis=-1))
        z = np.sum(p0 * vec4, axis=-1) / (np.linalg.norm(vec4, ord=2, axis=-1))
        
        x = np.expand_dims(x, axis=-1)    
        y = np.expand_dims(y, axis=-1)
        z = np.expand_dims(z, axis=-1)

        return np.concatenate([x, y, z], axis=-1).transpose(1, 0)
    '''
    def spatial_f(self, points, normals, mean):   #8:6d; 7:9d; 9:7d
        mid = points - mean
        d0 = np.linalg.norm(mid, ord=2, axis=-1, keepdims=True)

        idx1 = np.argsort(-d0, axis=0)[0]
        p1 = points - points[idx1] 
        d1 = np.linalg.norm(p1, ord=2, axis=-1, keepdims=True)

        d = -d0-d1
        i = 0
        
        while 1:
            idx2 = np.argsort(d, axis=0)[0]
            vec1 = points[idx1] - mean
            vec2 = points[idx2] - mean
            
            angle = np.sum(vec1 * vec2, axis=-1) / (np.linalg.norm(vec1, ord=2, axis=-1) * np.linalg.norm(vec2, ord=2, axis=-1))
            p2 = points - points[idx2]        
            d2 = np.linalg.norm(p2, ord=2, axis=-1, keepdims=True)
        
            d = d - d2
            if angle > 0.9999:
                if i>=10:
                    print('Its a line!!')
                    break
                else:
                    i = i + 1   
                    continue
            else:
                break

        i = 0
        d = -d0-d1-d2

        while 1:
            idx3 = np.argsort(d, axis=0)[0]
            p3 = points - points[idx3]        
            d3 = np.linalg.norm(p3, ord=2, axis=-1, keepdims=True)

            vec1 = points[idx1] - mean
            vec2 = points[idx2] - mean
            vec3 = points[idx3] - mean
            vec = np.concatenate([vec1, vec2, vec3], axis=0)
            
            det = np.linalg.det(vec)

            angle1 = np.sum(vec1 * vec3, axis=-1) / (np.linalg.norm(vec1, ord=2, axis=-1) * np.linalg.norm(vec3, ord=2, axis=-1))
            angle2 = np.sum(vec2 * vec3, axis=-1) / (np.linalg.norm(vec2, ord=2, axis=-1) * np.linalg.norm(vec3, ord=2, axis=-1))
            
            d = d - d3
            if det == 0 and i < 10:
                i = i + 1
                continue
            else:
                if angle1 > 0.9999 or angle2 > 0.9999:
                    continue
                else:
                    if i >= 10:
                        print('Its a plane!!')
                    break
        
        
        vec1 = points[idx1] - mean
        vec2 = points[idx2] - mean
        vec3 = points[idx3] - mean
        vec4 = points[idx1] - points[idx3]
        vec5 = points[idx2] - points[idx3]

        v1 = np.cross(vec1, vec2)
        v2 = np.cross(vec1, vec3)
        v3 = np.cross(vec2, vec3)
        v4 = np.cross(vec4, vec5)
        
        d_n = np.linalg.norm(normals, ord=2, axis=-1)

        a1 = np.sum(v1 * normals, axis=-1) / (np.linalg.norm(v1, ord=2, axis=-1) * d_n + 1e-6)
        a2 = np.sum(v2 * normals, axis=-1) / (np.linalg.norm(v2, ord=2, axis=-1) * d_n + 1e-6)
        a3 = np.sum(v3 * normals, axis=-1) / (np.linalg.norm(v3, ord=2, axis=-1) * d_n + 1e-6)
        a4 = np.sum(v4 * normals, axis=-1) / (np.linalg.norm(v4, ord=2, axis=-1) * d_n + 1e-6)
        
        a1 = np.expand_dims(a1, axis=-1)    
        a2 = np.expand_dims(a2, axis=-1)
        a3 = np.expand_dims(a3, axis=-1)
        a4 = np.expand_dims(a4, axis=-1)

        d0 = np.exp(-d0 / np.mean(d0, axis=0, keepdims=True))
        d1 = np.exp(-d1 / np.mean(d1, axis=0, keepdims=True))
        d2 = np.exp(-d2 / np.mean(d2, axis=0, keepdims=True))
        d3 = np.exp(-d3 / np.mean(d3, axis=0, keepdims=True))

        return np.expand_dims(np.concatenate([d0, d1, a1, d2, a2, d3, a3, a4], axis=-1).transpose(1, 0), axis=-1) 
    '''
    def fds_f(self, mid, points, normals):
        d1 = np.linalg.norm(mid, ord=2, axis=-1, keepdims=True)
        
        sum = -d1  
        idx2 = np.argsort(sum, axis=0)[0]
        p2 = points - points[idx2]       
        d2 = np.linalg.norm(p2, ord=2, axis=-1, keepdims=True)
        
        sum = sum - d2
        idx3 = np.argsort(sum, axis=0)[0]
        p3 = points - points[idx3]
        d3 = np.linalg.norm(p3, ord=2, axis=-1, keepdims=True)

        sum = sum - d3
        idx4 = np.argsort(sum, axis=0)[0]
        p4 = points - points[idx4]
        d4 = np.linalg.norm(p4, ord=2, axis=-1, keepdims=True)
        
        sum = sum - d4
        idx5 = np.argsort(sum, axis=0)[0]
        p5 = points - points[idx5]
        d5 = np.linalg.norm(p5, ord=2, axis=-1, keepdims=True)
        
        sum = sum - d5
        idx6 = np.argsort(sum, axis=0)[0]
        p6 = points - points[idx6]
        d6 = np.linalg.norm(p6, ord=2, axis=-1, keepdims=True)
        
        sum = sum - d6
        idx7 = np.argsort(sum, axis=0)[0]
        p7 = points - points[idx7]
        d7 = np.linalg.norm(p7, ord=2, axis=-1, keepdims=True)
        
        sum = sum - d7
        idx8 = np.argsort(sum, axis=0)[0]
        p8 = points - points[idx8]
        d8 = np.linalg.norm(p8, ord=2, axis=-1, keepdims=True)
        
        d1 = np.exp(-d1 / np.mean(d1, axis=0, keepdims=True))
        d2 = np.exp(-d2 / np.mean(d2, axis=0, keepdims=True))
        d3 = np.exp(-d3 / np.mean(d3, axis=0, keepdims=True))
        d4 = np.exp(-d4 / np.mean(d4, axis=0, keepdims=True))
        d5 = np.exp(-d5 / np.mean(d5, axis=0, keepdims=True))
        d6 = np.exp(-d6 / np.mean(d6, axis=0, keepdims=True))
        d7 = np.exp(-d7 / np.mean(d7, axis=0, keepdims=True))
        d8 = np.exp(-d8 / np.mean(d8, axis=0, keepdims=True))
        
        return np.expand_dims(np.concatenate([d1, d2, d3, d4, d5, d6, d7, d8], axis=-1).transpose(1, 0), axis=-1)
    
    def normal_estimation(self, d_matrix, points, knn):
        idx = np.argsort(d_matrix, axis=-1)[:, :knn]
        points_knn = points[idx]
        cov = np.array([np.cov(points_knn[i], rowvar=False) for i in range(points_knn.shape[0])])
        eigvals, eigs = np.linalg.eigh(cov)
        e_min = eigs[:, :, 0]
        return e_min
    
    def covariance(self, points_knn, normals_knn, points, normals):
        mid = points_knn - np.expand_dims(points, axis=1) #1024x16x3

        d = np.linalg.norm(mid, 2, axis=-1, keepdims=True)
        
        l1 = np.linalg.norm(normals, 2, axis=-1, keepdims=True)
        l1 = np.expand_dims(l1, axis=1)
        l2 = np.linalg.norm(normals_knn, 2, axis=-1, keepdims=True)
        
        a1 = np.sum(mid * np.expand_dims(normals, axis=1), axis=-1, keepdims=True) / (l1 * d) #1024x16x1
        a2 = np.sum(mid * normals_knn, axis=-1, keepdims=True) / (l2 * d) #1024x16x1
        a3 = np.sum(normals_knn * np.expand_dims(normals, axis=1), axis=-1, keepdims=True) / (l1 * l2) #1024x16x1
        
        cov = np.array([np.cov(points_knn[i], rowvar=False) for i in range(points_knn.shape[0])])
        eigvals, eigs = np.linalg.eigh(cov)
        e_max = np.expand_dims(eigs[:, :, 2], axis=1)
        
        a4 = np.sum(mid * e_max, axis=-1, keepdims=True) / d
        
        d = np.exp(-d / np.mean(d, axis=0, keepdims=True))
        
        return np.concatenate([d, a1, a2, a3, a4], axis=-1).transpose(2, 0, 1)

    def darboux(self, points, normals, points_knn, normals_knn):   
        mid = points_knn - np.expand_dims(points, axis=1)
       
        d = np.linalg.norm(mid, 2, axis=-1)#2048x16
        l1 = np.linalg.norm(normals, 2, axis=-1, keepdims=True)
        l2 = np.linalg.norm(mid, 2, axis=-1)
        l3 = np.linalg.norm(normals_knn, 2, axis=-1)
        
        a1 = np.matmul(mid, np.expand_dims(normals, axis=1).transpose(0, 2, 1)).squeeze(-1) / (l1 * l2) #2048x16
        a2 = np.sum(mid * normals_knn, axis=-1) / (l2 * l3) #2048x16
        a3 = np.sum(normals_knn * np.expand_dims(normals, axis=1), axis=-1) / (l1 * l3)
        
        v1 = np.cross(mid, np.expand_dims(normals, axis=1)) #2048x16x3
        v2 = np.cross(v1, np.expand_dims(normals, axis=1))
        v3 = np.cross(mid, normals_knn)
        v4 = np.cross(v3, normals_knn)

        a4 = np.sum(v1 * v3, axis=-1) / (np.linalg.norm(v1, 2, axis=-1) * np.linalg.norm(v3, 2, axis=-1) + 1e-6)#2048x16
        a5 = np.sum(v2 * v4, axis=-1) / (np.linalg.norm(v2, 2, axis=-1) * np.linalg.norm(v4, 2, axis=-1) + 1e-6)
            
    #    a6 = np.sum(v1 * v4, axis=-1) / (np.linalg.norm(v1, 2, axis=-1) * np.linalg.norm(v4, 2, axis=-1) + 1e-6)#2048x16
    #    a7 = np.sum(v2 * v3, axis=-1) / (np.linalg.norm(v2, 2, axis=-1) * np.linalg.norm(v3, 2, axis=-1) + 1e-6)
        
    #    d = np.exp(-d / np.mean(d, axis=0, keepdims=True))

        a1 = np.expand_dims(a1, axis=0)    
        a2 = np.expand_dims(a2, axis=0)
        a3 = np.expand_dims(a3, axis=0)
        a4 = np.expand_dims(a4, axis=0)
        a5 = np.expand_dims(a5, axis=0)
    #    a6 = np.expand_dims(a6, axis=0)
     #   a7 = np.expand_dims(a7, axis=0)
        d = np.expand_dims(d, axis=0)
        
        return np.concatenate([d, a1, a2, a3, a4, a5], axis=0) 
    
    def analysis(self, points, normals, points_knn, normals_knn):   
        mid = points_knn - np.expand_dims(points, axis=1)
        
        d = np.linalg.norm(mid, 2, axis=-1)#2048x16
        
        d = np.exp(-d / np.mean(d, axis=0, keepdims=True))
        
        d = np.expand_dims(d, axis=0)
        
        return d
    
    def ppf(self, points, normals, points_knn, normals_knn):
        mid = points_knn - np.expand_dims(points, axis=1)
        d = np.linalg.norm(mid, 2, axis=-1)#2048x16
        l1 = np.linalg.norm(normals, 2, axis=-1, keepdims=True)
        l2 = np.linalg.norm(mid, 2, axis=-1)
        l3 = np.linalg.norm(normals_knn, 2, axis=-1)
        
        a1 = np.matmul(mid, np.expand_dims(normals, axis=1).transpose(0, 2, 1)).squeeze(-1) / (l1 * l2) #2048x16
        a2 = np.sum(mid * normals_knn, axis=-1) / (l2 * l3) #2048x16
        a3 = np.sum(normals_knn * np.expand_dims(normals, axis=1), axis=-1) / (l1 * l3)
        
        d = np.exp(-d / np.mean(d, axis=0, keepdims=True))
        
        a1 = np.expand_dims(a1, axis=0)    
        a2 = np.expand_dims(a2, axis=0)
        a3 = np.expand_dims(a3, axis=0)
        d = np.expand_dims(d, axis=0)
        
        return np.concatenate([d, a1, a2, a3], axis=0) 
    
    def pca_correlation(self, points, pca_eig):
            pca_matrix = np.matmul(np.linalg.inv(pca_eig), points.transpose(1, 0))
            pca_matrix = np.expand_dims(pca_matrix, axis=-1)
            return pca_matrix

    def pca(self):
        pca_f = []
        
        for i in range(self.points.shape[0]):
            mid = np.mean(self.points[i], axis=0, keepdims=True)
            d0 = np.linalg.norm(self.points[i] - mid, 2, axis=-1, keepdims=True)
            key_p = []
            for j in range(128):
                idx = np.argsort(-d0, axis=0)[0]
                p = self.points[i][idx]
                key_p += [p]
                d = np.linalg.norm(self.points[i] - p, 2, axis=-1, keepdims=True)
                d0 = d0 + d
            key_p = np.squeeze(np.array(key_p), axis=1)

            meanVals = np.mean(key_p, axis=0, keepdims=True)
            meanRemoved = key_p - meanVals

            covMat = np.cov(meanRemoved, rowvar=0)
            print('Precessing:', i)
            eigVals, eigVects = np.linalg.eig(covMat)
            pca_f += [eigVects]
            
        
        pca_f = np.array(pca_f)

        if self.train:
            pca_file = os.path.join(self.data_dir, 'train_pca_f.h5')
        else:
            pca_file = os.path.join(self.data_dir, 'test_pca_f.h5')

        f = h5py.File(pca_file, 'w')
        f['pca'] = pca_f
        f.close()
        return pca_f 

class ModelNet40Cls_KC(data.Dataset):

    def __init__(self, num_points, train, download=True):
        super().__init__()
        
        self.folder = "modelnet40_ply_hdf5_2048"
        self.data_dir = os.path.join(BASE_DIR, self.folder)
        self.url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"

        if download and not os.path.exists(self.data_dir):
            zipfile = os.path.join(BASE_DIR, os.path.basename(self.url))
            subprocess.check_call(
                shlex.split("curl {} -o {}".format(self.url, zipfile))
            )

            subprocess.check_call(
                shlex.split("unzip {} -d {}".format(zipfile, BASE_DIR))
            )

            subprocess.check_call(shlex.split("rm {}".format(zipfile)))

        self.train, self.num_points = train, num_points
        
        if self.train:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'train_files.txt'))
            self.pca_f = os.path.join(self.data_dir, 'train_pca_f.h5')
        
        else:
            self.files =  _get_data_files( \
                os.path.join(self.data_dir, 'test_files.txt'))
            self.pca_f = os.path.join(self.data_dir, 'test_pca_f.h5')

        point_list, label_list, normal_list = [], [], []
        for f in self.files:
            points, normals, labels = _load_data_file(os.path.join(BASE_DIR, f))
            point_list.append(points)
            normal_list.append(normals)
            label_list.append(labels)
        
        self.points = np.concatenate(point_list, 0)
        self.normals = np.concatenate(normal_list, 0)
        self.labels = np.concatenate(label_list, 0)
        
    def __getitem__(self, idx):

        pt_idxs = np.random.choice(2048, 1024, replace=False)
        current_points = self.points[idx, pt_idxs, :].copy()
        current_points = torch.from_numpy(current_points).type(torch.FloatTensor)
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)
        return current_points, label
    
    def __len__(self):
        return self.points.shape[0]




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
