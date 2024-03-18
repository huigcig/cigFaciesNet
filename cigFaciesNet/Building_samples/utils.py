import os
import copy
import torch
import random
import torchvision
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from io import BytesIO
import scipy.misc
#import tensorflow as tf
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from PIL import Image


def dataupsample_2d(x,axis,num_up,method = 'nearest'):
  #x = np.transpose(x)
    if axis == 0:
        x1 = np.linspace(0,x.shape[0]-1,x.shape[0])
        x_new = np.linspace(0,x.shape[0]-1,num_up)
        gxu = np.zeros((num_up,x.shape[1]),dtype=np.single)
        for i in range(x.shape[1]):
            f = interpolate.interp1d(x1,x[:,i],kind = method)
            gxu[:,i] = f(x_new)
    elif axis == 1:
        x1 = np.linspace(0,x.shape[1]-1,x.shape[1])
        x_new = np.linspace(0,x.shape[1]-1,num_up)
        gxu = np.zeros((x.shape[0],num_up),dtype=np.single)
        for i in range(x.shape[0]):
            f = interpolate.interp1d(x1,x[i,:],kind = method)
            gxu[i,:] = f(x_new)
    return gxu

# 标准化
def mea_std_norm(x):
    if torch.is_tensor(x) and torch.std(x) != 0:
            x = (x - torch.mean(x)) / torch.std(x)
    elif np.std(x) != 0:
            x = (x - np.mean(x)) / np.std(x)
    return x

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1: 
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
# 曲线光滑函数
def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed   

# 定义数据集    
class build_dataset(Dataset):
    def __init__(self, samples_list, dataset_path):
        self.samples_list = samples_list
        self.dataset_path = dataset_path
        
    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_file_name = self.samples_list[idx]
        sample_file_path = os.path.join(self.dataset_path, sample_file_name + ".dat")
        sample = np.fromfile(sample_file_path, dtype=np.int32).reshape(128,128).T
        sample = sample[np.newaxis,:,:]
        return  sample

class build_dataset_classification(Dataset):
    def __init__(self, samples_list, smaples_file,class_name):
        self.samples_list = samples_list
        self.dataset_path = os.path.join(smaples_file, class_name + "_3")
        self.class_name = class_name
        
    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        classes = ["parallel","clinoform","fill","hummocky","chaotic"]
        sample_file_name = self.samples_list[idx]
        sample_file_path = os.path.join(self.dataset_path, sample_file_name + ".dat")
        sample = np.fromfile(sample_file_path, dtype=np.int32).reshape(128,128).T
        sample = sample[np.newaxis,:,:]
        label = np.eye(5)[classes.index(self.class_name)]
        return  [sample,label]
    
def plot_loss(path):
    log = np.load(path,allow_pickle=True).item()

    epoch_loss_g,epoch_loss_d = log["epoch_loss_g"],log["epoch_loss_d"]
    epoch_loss_real,epoch_loss_fake = log["epoch_loss_real"],log["epoch_loss_fake"]
    epoch_lr_g,epoch_lr_d = log["epoch_lr_g"],log["epoch_lr_d"]

    # 训练loss曲线
    x = [i for i in range(len(epoch_loss_g))]
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(x, smooth(epoch_loss_g, 0.6), label='Generator loss')
    ax.plot(x, smooth(epoch_loss_d, 0.6), label='Discriminator loss')
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Loss', fontsize=15)
    ax.set_title(f'Training curve', fontsize=15)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=10)

    ax = fig.add_subplot(1, 3, 2)
    ax.plot(x, smooth(epoch_loss_real, 0.6), label='Real loss')
    ax.plot(x, smooth(epoch_loss_fake, 0.6), label='Fake loss')
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Loss', fontsize=15)
    ax.set_title(f'Training curve', fontsize=15)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=10)

    ax = fig.add_subplot(1, 3, 3)
    ax.plot(x, epoch_lr_g,  label='Learning Rate - Generator')
    ax.plot(x, epoch_lr_d,  label='Learning Rate - Discriminator')
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Learning Rate', fontsize=15)
    ax.set_title(f'Learning rate curve', fontsize=15)
    ax.grid(True)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()

    
    