import torch
import torch.nn as nn  
import torch.nn.functional as F

class ClusterNet(nn.Module):
    def __init__(self, opt):
        self.opt = opt
        if opt.dataset == 'MNIST':
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 10, 5)
            self.fc1 = nn.Linear(160, 120)
            
            self.fc2 = nn.Linear(120, 30)
            self.fc3 = nn.Linear(30, 10)
            
            if opt.batch_norm:
                self.bn = nn.BatchNorm1d(30)
            
        elif opt.dataset == 'CIFAR10':
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16*5*5, 120)
            
            self.fc2 = nn.Linear(120, 30)
            self.fc3 = nn.Linear(30, 10)
            
            if opt.batch_norm:
                self.bn = nn.BatchNorm1d(30)
                
    def forward(self, x):
        return x