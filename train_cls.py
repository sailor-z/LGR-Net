from __future__ import print_function
import os
import sys
import random
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import sklearn.metrics as metrics
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from ModelNet40Loader import ModelNet40Cls
from Network import Dar_Cls, loss_f, cal_loss
import torch.nn.functional as F

def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 500 epochs"""
    lr = max((opt.lr * (opt.decay_rate ** (epoch  // opt.decay_step))), 1e-5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def str2bool(v):
    return v.lower() in ("true", "1")

def _init_():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--reg_weight', type=float, default=0.0001, help='reg weight')
    parser.add_argument('--num_points', type=int, default=1024, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=16)
    parser.add_argument('--nepoch', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--use_sgd', type=bool, default=True, help='Use SGD')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--outf', type=str, default='./logs/',  help='output folder')
    parser.add_argument('--model', type=str, default = './logs/',  help='model path')
    parser.add_argument('--load', type=str2bool, default = False,  help='load model')
    parser.add_argument('--lr', type=float, default = 0.001,  help='learning rate')
    parser.add_argument('--decay_rate', type=float, default = 0.7,  help='decay rate')
    parser.add_argument('--decay_step', type=int, default = 100,  help='decay step')
    parser.add_argument('--gpu_num', type=str, default = "1",  help='gpu_num')
    parser.add_argument('--test_interval', type=int, default = 1,  help='test interval')
    parser.add_argument('--knn', type=int, default = 32,  help='knn')
    parser.add_argument('--eval', type=bool,  default= False, help='evaluate the model')
    parser.add_argument('--bias', type=bool,  default=False, help='bias of convolution')
    parser.add_argument('--SO3', type=bool,  default=True, help='SO3 rotation')
    
    opt = parser.parse_args()
    print (opt)
    
    blue = lambda x:'\033[94m' + x + '\033[0m'
    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    
    return opt

def train(opt):
    dataset = ModelNet40Cls(partition='train', d_a=True, SO3=opt.SO3)
    testdataset = ModelNet40Cls(partition='test', d_a=False, SO3=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers), drop_last=True)                                       
    '''  #ShapeNet
    dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, npoints = opt.num_points)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))

    test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, train = False, npoints = opt.num_points)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))
    '''
    print('dataset_len:', len(dataset))

    if not os.path.exists(opt.outf):
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

    if not os.path.exists(opt.model):
        try:
            os.makedirs(opt.model)
        except OSError:
            pass

    classifier = Dar_Cls(k=40, knn=opt.knn, train_idx=True, cv_bias=opt.bias)

    #device_ids = [0, 1]
    #classifier = nn.DataParallel(classifier, device_ids=device_ids)

    classifier.cuda()

    loss = cal_loss
    #loss = loss_f().cuda()

    if opt.use_sgd:
        print("Use SGD")
        optimizer = optim.SGD(classifier.parameters(), lr=opt.lr*100, momentum=opt.momentum, weight_decay=1e-4)#, weight_decay=5e-4   should be added later
    else:
        print("Use Adam")
        optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, weight_decay=1e-4)#, weight_decay=5e-4

    if opt.use_sgd:     
        scheduler = CosineAnnealingLR(optimizer, opt.nepoch, eta_min=opt.lr)
    else:
        scheduler = CosineAnnealingLR(optimizer, opt.nepoch, eta_min=opt.lr*0.01)

    if opt.load == True:
        classifier.load_state_dict(torch.load(opt.model + 'cls_best.pth'))
        if os.path.exists(opt.outf + 'hyperparameters.txt'):
            hyperparameters = np.loadtxt(opt.outf + 'hyperparameters.txt')
            ending_epoch = hyperparameters
            acc_old = np.loadtxt(opt.outf + 'best_acc.txt')
            if opt.use_sgd:  
                for i in range(int(ending_epoch)):
                    scheduler.step()
        else:
            acc_old = 0
            ending_epoch = 0
    else:
        acc_old = 0
        ending_epoch = 0

    classifier = classifier.train()

    for epoch in range(int(ending_epoch), opt.nepoch):
        #adjust_learning_rate(optimizer, epoch, opt)
        scheduler.step()
        train_loss = 0.0
        count = 0.0
        train_pred = []
        train_true = []
        for i, data in enumerate(dataloader, 0):
            points, normals, target = data
            points = points.cuda()
            normals = normals.cuda()
            target = target.cuda().squeeze(-1)

            pred = classifier(points, normals)
            
            optimizer.zero_grad()

            l = loss(pred, target)
            
            l.backward()
            optimizer.step()
            
            preds = pred.max(1)[1]
            count += opt.batchSize
            train_loss += l.item() * opt.batchSize

            train_true.append(target.data.cpu().numpy())
            train_pred.append(preds.data.cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f' % (epoch, train_loss*1.0/count, metrics.accuracy_score(train_true, train_pred))
        print(outstr)
        torch.save(classifier.state_dict(), opt.model + 'cls_best.pth')

        acc = test(opt, testdataset, 'cls')
        
        if acc > acc_old:
            np.savetxt(opt.model + 'best_acc.txt', np.expand_dims(np.array(acc), axis =-1))
            np.savetxt(opt.model + 'hyperparameters.txt', np.expand_dims(np.array(epoch), axis =-1))
            torch.save(classifier.state_dict(), opt.model + 'cls_best.pth')
            acc_old = acc

    acc = test(opt, testdataset, 'cls_best')

def test(opt, testdataset, model_name):
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=16,
                                          shuffle=False, num_workers=int(opt.workers), drop_last=False)   
                                          
    classifier = Dar_Cls(k=40, knn=opt.knn, train_idx=False, cv_bias=opt.bias).cuda()
    classifier.load_state_dict(torch.load(opt.model + model_name + '.pth'))
#    classifier = torch.load(opt.model + model_name + '.pkl').cuda()
    classifier.eval()
    
    test_acc = 0.0
    test_true = []
    test_pred = []

    for i, data in enumerate(testdataloader, 0):
        points, normals, target = data

        points, target = points.cuda(), target.cuda().squeeze(-1)
        normals = normals.cuda()

        pred = classifier(points, normals)
       
        preds = pred.max(1)[1]
        test_true.append(target.data.cpu().numpy())
        test_pred.append(preds.data.cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    test_acc = metrics.accuracy_score(test_true, test_pred)

    outstr = 'Test :: test acc: %.6f'%(test_acc)

    print(outstr)

    return test_acc
        
if __name__ == "__main__":

    args = _init_()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    print('Using GPU : ' + str(args.gpu_num))

    if not args.eval:
        train(args)
    else:
        testdataset = ModelNet40Cls(partition='test', d_a=False, SO3=args.SO3)
        test_acc = test(args, testdataset, 'cls_best')
        np.savetxt(args.outf + 'test_acc.txt', np.expand_dims(test_acc, axis=0))
  
        
