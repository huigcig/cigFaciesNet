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

class build_dataset_classification(Dataset):
    def __init__(self, samples_list, dataset_path, class_name):
        self.samples_list = samples_list
        self.dataset_path = dataset_path
        self.class_name = class_name
        
    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_file_name = self.samples_list[idx]
        sample_file_path = os.path.join(self.dataset_path, sample_file_name + ".dat")
        sample = np.fromfile(sample_file_path, dtype=np.int32).reshape(128,128).T
        sample = np.where(sample==0,-1,sample)
        sample = sample[np.newaxis,:,:]
        
        return  sample,class_name


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
        sample = np.where(sample==0,-1,sample)
        sample = sample[np.newaxis,:,:]
        
        return  sample
