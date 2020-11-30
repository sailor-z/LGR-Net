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
from utils import global_transform, darboux, get_graph_feature

class dar_feat_att(nn.Module):
    def __init__(self, global_feat=True, batchsize=32):
        super(dar_feat_att, self).__init__()
        self.batchsize = batchsize

        self.conv1 = ResNet_Block(5, 64, True)
        self.conv2 = ResNet_Block(64, 128, True)
        self.conv3 = ResNet_Block(128, 128, False)
        self.conv4 = ResNet_Block(128, 128, False)
        self.conv5 = ResNet_Block(128, 1024, True)

        self.spa_conv = nn.Sequential(
          nn.Conv2d(2, 1, (1, 1)),
          nn.BatchNorm2d(1),
          nn.ReLU(True)
        )

        self.chl_conv = nn.Sequential(
          nn.Linear(256, 128),
          nn.BatchNorm1d(128),
          nn.ReLU(True),
          nn.Linear(128, 128),
          nn.BatchNorm1d(128),
          nn.ReLU(True),
          nn.Linear(128, 256),
          nn.BatchNorm1d(256),
          nn.ReLU(True),
        )

        self.global_feat = global_feat

    def spatial_attention(self, x):
        ave_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        out = torch.cat([ave_out, max_out], dim=1)
        return out

    def channel_attention(self, x):
        ave_out = F.avg_pool2d(x, (x.size(2), x.size(-1)))
        max_out = F.max_pool2d(x, (x.size(2), x.size(-1)))
        out = torch.cat([ave_out, max_out], dim=1)
        return out.view(x.size(0), -1)

    def forward(self, x):
        n_pts = x.size(2)
        knn = x.size(-1)

        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        sp_att = self.spatial_attention(out3)
        sp_att = self.spa_conv(sp_att)
        sp_att = F.softmax(sp_att, dim=-1)
        out3 = out3 * sp_att

        ch_att = self.channel_attention(out3)
        ch_att = torch.sum(self.chl_conv(ch_att).view(self.batchsize, 128, 2, 1), dim=2, keepdim=True)
        ch_att = F.softmax(ch_att, dim=1)

        out3 = out3 * ch_att
        out4 = out2 + out3

        out4 = self.conv4(out4)

        out5 = F.max_pool2d(out4, (1, knn))

        out6 = self.conv5(out5)

        out = F.max_pool2d(out6, (n_pts, 1))
        out = out.view(-1, 1024)  #2048
        if self.global_feat:
            return out
        else:
            out = out.view(-1, 1024, 1, 1).repeat(1, 1, n_pts, 1)  #2048
            return torch.cat([out, out1, out2, out3, out4, out5, out6], 1)

class feature_fusion(nn.Module):
    def __init__(self):
        super(feature_fusion, self).__init__()

    #    self.conv1 = ResNet_Block(128, 128, False)
    #    self.conv2 = ResNet_Block(128, 128, False)

        self.conv = nn.Sequential(
          nn.Conv2d(1024, 1024, (1, 1), bias=False),
          nn.BatchNorm2d(1024),
          nn.LeakyReLU(negative_slope=0.2),
        )
    def forward(self, x):
        out = self.conv(x)
        att = F.softmax(out, dim=-1)
        out = x * att
        out = torch.sum(out, dim=-1, keepdim=True)
        return out

class spatial_att(nn.Module):
    def __init__(self):
        super(spatial_att, self).__init__()
        self.conv = nn.Sequential(
          nn.Conv2d(2048, 512, (1, 1), bias=False),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(negative_slope=0.2),
          nn.Conv2d(512, 128, (1, 1), bias=False),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(negative_slope=0.2),
        )
        self.spa_conv = nn.Sequential(
          nn.Conv2d(2, 1, (1, 1), bias=False),
          nn.BatchNorm2d(1),
          nn.LeakyReLU(negative_slope=0.2),
        )
    def spatial_attention(self, x):
        ave_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        out = torch.cat([ave_out, max_out], dim=1)
        out = self.spa_conv(out)
        return out

    def forward(self, x):
        out = self.conv(x)
        att = self.spatial_attention(out)
        att = F.softmax(att, dim=2)
        out = x * att
        idx = torch.max(out, dim=2, keepdim=True)[1]
        out = torch.gather(x, dim=2, index=idx)
        return out

class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 3), stride=(1, stride), padding=(0, 1),  bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 3), stride=(1, stride), padding=(0, 1),  bias=False),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return self.lrelu(out)

class Graph_Kernel(nn.Module):
    def __init__(self, inchannel, outchannel, pre):
        super(Graph_Kernel, self).__init__()
        self.pre = pre
        self.kernel = torch.nn.Parameter(torch.zeros([inchannel, outchannel]))
        torch.nn.init.xavier_normal_(self.kernel.data)

        self.bn = nn.BatchNorm2d(outchannel)

    def forward(self, x, g):
        if self.pre == True:
            out = torch.bmm(g.squeeze(-1), self.kernel.unsqueeze(0).repeat(g.size(0), 1, 1))
        else:
            out = torch.bmm(g.squeeze(-1), x.squeeze(-1).permute(0, 2, 1))
            out = torch.bmm(out, self.kernel.unsqueeze(0).repeat(g.size(0), 1, 1))

        out = out.permute(0, 2, 1).unsqueeze(-1)
        return F.relu(self.bn(out))

class dar_feat(nn.Module):
    def __init__(self, global_feat=True, knn=16, train_idx=True, cv_bias=False):
        super(dar_feat, self).__init__()
        self.knn = knn
        self.train_idx = train_idx
        self.cv_bias = cv_bias

        self.gb_gconv_1 = nn.Sequential(
          nn.Conv2d(6, 64, (1, 1), bias=self.cv_bias),
          nn.BatchNorm2d(64),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.gb_gconv_2 = nn.Sequential(
          nn.Conv2d(64, 128, (1, 1), bias=self.cv_bias),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.gb_gconv_3 = nn.Sequential(
          nn.Conv2d(128, 512, (1, 1), bias=self.cv_bias),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.gb_gconv_4 = nn.Sequential(
          nn.Conv2d(512, 1024, (1, 1), bias=self.cv_bias),
          nn.BatchNorm2d(1024),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.lc_gconv_1 = nn.Sequential(
          nn.Conv2d(8, 64, (1, 1), bias=self.cv_bias),
          nn.BatchNorm2d(64),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.lc_gconv_2 = nn.Sequential(
          nn.Conv2d(64, 128, (1, 1), bias=self.cv_bias),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.lc_gconv_3 = nn.Sequential(
          nn.Conv2d(128, 512, (1, 1), bias=self.cv_bias),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.lc_gconv_4 = nn.Sequential(
          nn.Conv2d(512, 1024, (1, 1), bias=self.cv_bias),
          nn.BatchNorm2d(1024),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv = nn.Sequential(
          nn.Conv2d(1024, 2048, (1, 1), bias=self.cv_bias),
          nn.BatchNorm2d(2048),
          nn.LeakyReLU(negative_slope=0.2),
        )
        self.feature_fusion = feature_fusion()

        self.global_feat = global_feat

    def region_pooling(self, num, x):
        _, _, _, k_num = x.size()
        group = torch.chunk(x, num, dim=-1)
        feature = []
        for i in range(num):
            feature += [torch.max(group[i], dim=-1, keepdim=False)[0]]
        feature = torch.stack(feature).permute(1, 2, 3, 0)
        return feature

    def forward(self, points, normals):
        n_pts = points.size(2)
      #  global_f = get_graph_feature(points, k=32)
        global_f = global_transform(points, 32, self.train_idx, self.knn) #[B, 3C, N]
        local_f = darboux(points, normals, self.knn)  #[B, C, N, K]

        l_out = self.lc_gconv_1(local_f)
        l_out = self.lc_gconv_2(l_out)
        l_out = F.max_pool2d(l_out, (1, self.knn))
        l_out = self.lc_gconv_3(l_out)
        l_out = self.lc_gconv_4(l_out)

        g_out = self.gb_gconv_1(global_f)
        g_out = self.gb_gconv_2(g_out)
        g_out = F.max_pool2d(g_out, (1, self.knn))
        g_out = self.gb_gconv_3(g_out)
        g_out = self.gb_gconv_4(g_out)

      #  out = (l_out + g_out) / 2
        out = torch.cat([g_out, l_out], dim=-1)
        out = self.feature_fusion(out)

        out = self.conv(out)
        out = F.max_pool2d(out, (n_pts, 1))
        out = out.view(-1, 2048)

        if self.global_feat:
            out = out.view(-1, 2048)
            return out
        else:
            out = out.view(-1, 2048, 1, 1).repeat(1, 1, n_pts, 1)
        #    return torch.cat([out, out1, out2], 1)
            return torch.cat([out, g_out, l_out], 1)

class Dar_Cls(nn.Module):
    def __init__(self, k=40, knn=16, train_idx=True, cv_bias=False):
        super(Dar_Cls, self).__init__()

        self.class_nums = k
        self.knn = knn
        self.cv_bias = cv_bias
        self.train_idx = train_idx

        self.feat = dar_feat(global_feat=True, knn=self.knn, train_idx=self.train_idx, cv_bias=self.cv_bias)

        self.classify = nn.Sequential(
          nn.Linear(2048, 512, bias=self.cv_bias),
          nn.BatchNorm1d(512),
          nn.Dropout(0.5),
          nn.LeakyReLU(negative_slope=0.2),
          nn.Linear(512, 256),
          nn.BatchNorm1d(256),
          nn.Dropout(0.5),
          nn.LeakyReLU(negative_slope=0.2),
          nn.Linear(256, self.class_nums)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, points, normals):
        x = self.feat(points, normals)
        x = self.classify(x)
        return x
        #return F.log_softmax(x, dim=-1).view(x.size(0), -1)

class Dar_Seg(nn.Module):
    def __init__(self, k=50, knn=16, train_idx=True):
        super(Dar_Seg, self).__init__()

        self.k = k
        self.train_idx = train_idx
        self.knn = knn

        self.feat = dar_feat(global_feat=False, knn=self.knn, train_idx=self.train_idx, cv_bias=False)
        self.conv1 = nn.Sequential(
          nn.Conv2d(4096+16, 512, (1, 1), bias=False),   #5120 + 16
          nn.BatchNorm2d(512),
          nn.Dropout(0.5),
          nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
          nn.Conv2d(512, 256, (1, 1), bias=False),
          nn.BatchNorm2d(256),
          nn.Dropout(0.5),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv3 = torch.nn.Conv2d(256, self.k, (1, 1), bias=False)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, points, normals, cls_label):
        x = self.feat(points, normals)
        cls_label = cls_label.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, points.size(2), 1)
        x = self.conv1(torch.cat([x, cls_label], dim=1))
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        return x

class KnnBatchNorm(nn.Module):
  def __init__(self):
    super(KnnBatchNorm, self).__init__()
  def forward(self, x):
    x_mean = torch.mean(x, dim=-1, keepdim=True)
    x_std = torch.std(x, dim=-1, keepdim=True)
    return (x - x_mean) / (x_std + 1e-10)

class ScaneObjectNN(nn.Module):
    def __init__(self, k=15, knn=16, train_idx=True):
        super(ScaneObjectNN, self).__init__()
        self.knn = knn
        self.class_nums = k
        self.train_idx = train_idx

        self.gb_gconv_1 = nn.Sequential(
          nn.Conv2d(6, 64, (1, 4), stride=(1, 4), bias=False),
          nn.BatchNorm2d(64),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.gb_gconv_2 = nn.Sequential(
          nn.Conv2d(64, 128, (1, 2), stride=(1, 2), bias=False),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.gb_gconv_3 = nn.Sequential(
          nn.Conv2d(128, 512, (1, 2), stride=(1, 2), bias=False),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.gb_gconv_4 = nn.Sequential(
          nn.Conv2d(512, 1024, (1, 2), bias=False),
          nn.BatchNorm2d(1024),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.lc_gconv_1 = nn.Sequential(
          nn.Conv2d(8, 64, (1, 4), stride=(1, 4), bias=False),
          nn.BatchNorm2d(64),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.lc_gconv_2 = nn.Sequential(
          nn.Conv2d(64, 128, (1, 2), stride=(1, 2), bias=False),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.lc_gconv_3 = nn.Sequential(
          nn.Conv2d(128, 512, (1, 2), stride=(1, 2), bias=False),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.lc_gconv_4 = nn.Sequential(
          nn.Conv2d(512, 1024, (1, 2), bias=False),
          nn.BatchNorm2d(1024),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv = nn.Sequential(
          nn.Conv2d(2048, 2048, (1, 1), bias=False),
          nn.BatchNorm2d(2048),
          nn.LeakyReLU(negative_slope=0.2),
        )

        self.classify = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, self.class_nums)
            )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, points, normals):
        n_pts = points.size(2)

        global_f = global_transform(points, 32, self.train_idx) #[B, 3C, N]
        local_f = darboux(points, normals, self.knn)  #[B, C, N, K]

        l_out = self.lc_gconv_1(local_f)
        l_out = self.lc_gconv_2(l_out)
        l_out = self.lc_gconv_3(l_out)
        l_out = self.lc_gconv_4(l_out)

        g_out = self.gb_gconv_1(global_f)
        g_out = self.gb_gconv_2(g_out)
        g_out = self.gb_gconv_3(g_out)
        g_out = self.gb_gconv_4(g_out)

        out = torch.cat([g_out, l_out], dim=1)

        out = self.conv(out)

        out = F.max_pool2d(out, (n_pts, 1))

        out = out.view(-1, 2048)
        out = self.classify(out)
      #  return F.log_softmax(out, dim=-1).view(out.size(0), -1)
        return out

class loss_f(nn.Module):
    def __init__(self):
        super(loss_f, self).__init__()

    def forward(self, pred, label):
        l_cls = F.nll_loss(pred, label)
        return l_cls

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss
