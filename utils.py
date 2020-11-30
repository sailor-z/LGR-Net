from __future__ import print_function
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
#from modules import nn as syncbn
import numpy as np
import torch.nn.functional as F
from pt_utils.svd.batch_svd import batch_svd

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k+1, dim=-1)[1][:, :, 1:]   # (batch_size, num_points, k)
   # idx = pairwise_distance.topk(k=k, dim=-1)[1][:, :, :]   # (batch_size, num_points, k)
    return idx

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = torch.zeros(batch_data.size()).cuda()
    rotation_matrix = []
    for k in range(batch_data.size(0)):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix += [np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])]
    rotation_matrix = torch.from_numpy(np.asarray(rotation_matrix)).type(torch.FloatTensor).cuda()
    rotated_data = torch.matmul(batch_data, rotation_matrix)
    return rotated_data, rotation_matrix

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.size()
    _, M, _ = dst.size()
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def spatial_f(points, centroids):
    mean = torch.mean(points, dim=1, keepdim=True) #[B, 1, C]
    vec1 = centroids[:, 0].unsqueeze_(1)   #[B, 1, C]
    vec2 = centroids[:, 1].unsqueeze_(1)   #[B, 1, C]

    vec1 -= mean  #[B, 1, C]
    vec2 -= mean

    vec3 = torch.cross(vec1, vec2)  #[B, 1, C]
    vec4 = torch.cross(vec2, vec3)

    d2 = torch.norm(vec2, p=2, dim=-1)
    d3 = torch.norm(vec3, p=2, dim=-1)
    d4 = torch.norm(vec4, p=2, dim=-1)
    
    x = torch.sum(points * vec2, dim=-1) / (d2 + 1e-10)
    y = torch.sum(points * vec3, dim=-1) / (d3 + 1e-10)
    z = torch.sum(points * vec4, dim=-1) / (d4 + 1e-10)

    x.unsqueeze_(dim=-1)  
    y.unsqueeze_(dim=-1)  
    z.unsqueeze_(dim=-1)  

    return torch.cat([x, y, z], dim=-1) #[B, N, C]

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((x, feature - x), dim=3).permute(0, 3, 1, 2)
    return feature
  #  return feature.permute(0, 3, 1, 2)

def grouping(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 

    return feature.permute(0, 3, 1, 2)

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.size()
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    
    centroid = torch.mean(xyz, dim=1, keepdim=True) #[B, 1, C]
    dist = torch.sum((xyz - centroid) ** 2, -1)
    farthest = torch.max(dist, -1)[1]

    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def darboux(points, normals, k):   
    B, C, N = points.size()
    idx = knn(points, k)

    points_knn = grouping(points, k, idx)
    normals_knn = grouping(normals, k, idx)

    mid = points_knn - points.unsqueeze(-1) #[B, C, N, K]

    d = torch.norm(mid, p=2, dim=1)    #[B, N, K]   
    l1 = torch.norm(normals, p=2, dim=1).unsqueeze(-1) #[B, N, 1]
    l2 = torch.norm(normals_knn, p=2, dim=1) #[B, N, K]
    
    a1 = torch.sum(mid * normals.unsqueeze(-1), dim=1) / (d * l1 + 1e-10) #[B, N, K]
    a2 = torch.sum(mid * normals_knn, dim=1) / (d * l2 + 1e-10) #[B, N, K]
    a3 = torch.sum(normals_knn * normals.unsqueeze(-1), dim=1) / (l2 * l1 + 1e-10) #[B, N, K]

    mid = mid.permute(0, 2, 3, 1).contiguous().view(-1, k, C) #[BN, K, C]
    normals_knn = normals_knn.permute(0, 2, 3, 1).contiguous().view(-1, k, C)
    normals = normals.permute(0, 2, 1).contiguous().view(-1, 1, C)

    v1 = torch.cross(mid, normals.repeat(1, k, 1))#[BN, K, C]
    v2 = torch.cross(v1, normals.repeat(1, k, 1))
    v3 = torch.cross(mid, normals_knn)
    v4 = torch.cross(v3, normals_knn)

    d1 = torch.norm(v1, p=2, dim=-1)
    d2 = torch.norm(v2, p=2, dim=-1)
    d3 = torch.norm(v3, p=2, dim=-1)
    d4 = torch.norm(v4, p=2, dim=-1)

    a4 = torch.sum(v1 * v3, dim=-1) / (d1 * d3 + 1e-10) #[BN, K]
    a4 = a4.view(B, N, k)
    a5 = torch.sum(v2 * v4, dim=-1) / (d2 * d4  + 1e-10) #[BN, K]
    a5 = a5.view(B, N, k)
    
    a6 = torch.sum(v1 * v4, dim=-1) / (d1 * d4 + 1e-10) #[BN, K]
    a6 = a6.view(B, N, k)
    a7 = torch.sum(v2 * v3, dim=-1) / (d2 * d3  + 1e-10) #[BN, K]
    a7 = a7.view(B, N, k)

    a1.unsqueeze_(1)
    a2.unsqueeze_(1)
    a3.unsqueeze_(1)
    a4.unsqueeze_(1)
    a5.unsqueeze_(1)
    a6.unsqueeze_(1)
    a7.unsqueeze_(1)
    d.unsqueeze_(1)
 #   feature1 = torch.cat([d, a1, a2, a3, a4, a5], dim=1)
 #   feature2 = get_graph_feature(xyz, k=32)
    return torch.cat([d, a1, a2, a3, a4, a5, a6, a7], dim=1)
#    return torch.cat([feature2, feature1], dim=1)  #[B, C, N, K]

def global_transform(points, npoints, train, knn):
    points = points.permute(0, 2, 1)
    idx = farthest_point_sample(points, npoints)
    centroids = index_points(points, idx)   #[B, S, C] 
    U, S, V = batch_svd(centroids)

    if train == True:
        index = torch.randint(2, (points.size(0), 1, 3)).type(torch.FloatTensor).cuda()
        V_ = V * index
        V -= 2 * V_
    else:
        key_p = centroids[:, 0, :].unsqueeze(1)
        angle = torch.matmul(key_p, V)
        index = torch.le(angle, 0).type(torch.FloatTensor).cuda()      
        V_ = V * index
        V -= 2 * V_

    xyz = torch.matmul(points, V).permute(0, 2, 1)

    feature = get_graph_feature(xyz, k=knn) #[B, C, S, K] 
    return feature
  #  xyz1 = spatial_f(points, centroids[:, :2]).permute(0, 2, 1)  #[B, C, N]
  #  xyz2 = spatial_f(points, centroids[:, 2:4]).permute(0, 2, 1)  #[B, C, N]
  #  xyz3 = spatial_f(points, centroids[:, 4:]).permute(0, 2, 1)  #[B, C, N]
 #  feature = get_graph_feature(xyz, k=32) #[B, C, S, K] 
 #   
   # xyz2 = spatial_f(points, centroids[:, 2:4])
   # xyz3 = spatial_f(points, centroids[:, 4:])
   # return xyz1, xyz2, xyz3



