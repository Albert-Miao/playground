import math
from sklearn.cluster import MiniBatchKMeans
import numpy as np

import torch
import torch.nn as nn  
import torch.nn.functional as F

import torch.optim as optim

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class NaiveNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        self.stats = torch.zeros([10000, opt.hidden_rep_dim + 1])
        self.stat_index = 0
        
        self.hidden_rep_dim = opt.hidden_rep_dim
        self.batch_size = opt.batch_size
        self.super_batch_size = opt.super_batch_size
        
        self.criterion = nn.CrossEntropyLoss()
        
        if opt.dataset == 'MNIST':
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 10, 5)
            self.fc1 = nn.Linear(160, 120)
            
            self.fc2 = nn.Linear(120, self.hidden_rep_dim)
            self.fc3 = nn.Linear(self.hidden_rep_dim, 10)
            
            if opt.batch_norm:
                self.bn = nn.BatchNorm1d(self.hidden_rep_dim)
            
        elif opt.dataset == 'CIFAR10':
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16*5*5, 120)
            
            self.fc2 = nn.Linear(120, self.hidden_rep_dim)
            self.fc3 = nn.Linear(self.hidden_rep_dim, 10)
            
            if opt.batch_norm:
                self.bn = nn.BatchNorm1d(self.hidden_rep_dim)
                
        self.optimizer = optim.SGD(self.parameters(), lr=opt.lr, momentum=opt.momentum)
        
    def forward(self, x):
        hidden_reps = 0
        cluster_loss = 0
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        if self.opt.batch_norm:
            x = self.bn(x)
        
        self.stats[self.stat_index:self.stat_index+x.size()[0], :self.hidden_rep_dim] = x.detach().clone().cpu()

        x = F.relu(x)
        x = self.fc3(x)
    
        return x, cluster_loss, hidden_reps
    

class ClusterNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        self.stats = torch.zeros([10000, opt.hidden_rep_dim + 1])
        self.stat_index = 0
        
        self.hidden_rep_dim = opt.hidden_rep_dim
        self.batch_size = opt.batch_size
        self.super_batch_size = opt.super_batch_size
        self.num_clusters = opt.num_clusters
        
        self.kmeans = None
        self.centers = None
        
        self.assignment = None
        
        self.id_mat = None
        if self.opt.model_type == 'shiftingCluster':
            self.id_mat = torch.eye(30).cuda().to(int)
            test = torch.tensor([((1 - (0.8) ** 2) / self.hidden_rep_dim) ** (1/2), 0.8]).cuda()
            self.id_mat = test[self.id_mat]
        
        self.cl_alpha = opt.cl_alpha
        self.cl_beta = opt.cl_beta
        self.cl_rate = opt.initial_cl_rate
        self.cl_rate_speed = opt.cl_rate_speed
        
        self.criterion = nn.CrossEntropyLoss()
        
        if opt.dataset == 'MNIST':
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 10, 5)
            self.fc1 = nn.Linear(160, 120)
            
            self.fc2 = nn.Linear(120, self.hidden_rep_dim)
            self.fc3 = nn.Linear(self.hidden_rep_dim, 10)
            
            if opt.batch_norm:
                self.bn = nn.BatchNorm1d(self.hidden_rep_dim)
            
        elif opt.dataset == 'CIFAR10':
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16*5*5, 120)
            
            self.fc2 = nn.Linear(120, self.hidden_rep_dim)
            self.fc3 = nn.Linear(self.hidden_rep_dim, 10)
            
            if opt.batch_norm:
                self.bn = nn.BatchNorm1d(self.hidden_rep_dim)
                
        self.optimizer = optim.SGD(self.parameters(), lr=opt.lr, momentum=opt.momentum)

                
    def forward(self, x):
        hidden_reps = 0
        cluster_loss = 0
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        if self.opt.batch_norm:
            x = self.bn(x)
            
        self.stats[self.stat_index:self.stat_index+x.size()[0], :self.hidden_rep_dim] = x.detach().clone().cpu()
        hidden_reps = x
        
        if not self.kmeans is None:
            if self.opt.model_type == 'simpleCluster':
                x_clusters = self.kmeans.cluster_centers_[self.kmeans.predict(x.detach().cpu().numpy())]
                dists = torch.sum((x - torch.from_numpy(x_clusters).cuda()) ** 2 / x.size()[1], dim=1)
                cluster_loss = torch.mean(dists) * self.cl_alpha * self.cl_rate
                
            if self.opt.model_type == 'explodingCluster':
                # Exploding clusters - encourages points to move away from all clusters except their own, but also stick to own cluster
                center_inds = self.kmeans.predict(x.detach().cpu().numpy())
                    
                # (number of concepts, batch size, dimension/channels) -> (batch size, number of concepts, dimension / channels)
                x_repeat = x.expand(30, x.size()[0], 30).transpose(0,1)
                centers_repeat = self.centers.expand(x.size()[0], 30, 30)
                
                # Here we encourage points close to other centers to drift away from them based off of inverse distance. Maybe explore negative?
                dists = 1 / torch.norm(x_repeat - centers_repeat, dim=2)
                dists[dists > 5] = 5
                
                # On the other hand, points will move closer to their own centers based off of square root. Also multiply by number of concepts and a relative beta term
                # TODO: Add option to affect formula clustering loss uses (for both cluster and explosion)
                for i in range(x.size()[0]):
                    dists[i][center_inds[i]] = (1 / dists[i][center_inds[i]]) ** (1 / 2) * 30 * self.cl_beta
                
                cluster_loss = torch.mean(dists) * self.cl_alpha * self.cl_rate
                
            if self.opt.model_type == "expandingCluster":
                # Encourages clusters to move outwards - expanding cluster training 
                # (NEEDS REWRITING, currently pushes clusters outwards but collapses clusters at boundary)
                cluster_inds = self.kmeans.predict(x.detach().cpu().numpy())
                x_clusters = self.centers[cluster_inds]
                dists = torch.sum((x - x_clusters) ** 2 / x.size()[1], dim=1)
                
                cluster_loss = torch.mean(dists) * self.cl_alpha * self.cl_rate
                
            if self.opt.model_type == "shiftingCluster":
                # Encourages clusters to move to predefined 'thin' vectors - shifting cluster training
                x_clusters = self.id_mat[self.assignment[self.kmeans.predict(x.detach().cpu().numpy())]]
                dists = torch.sum((x - x_clusters) ** 2 / x.size()[1], dim=1)
                
                cluster_loss = torch.mean(dists) * self.cl_alpha * self.cl_rate
            
        x = F.relu(x)
        x = self.fc3(x)
    
        return x, cluster_loss, hidden_reps
    
    
    def classExplodeClusterLoss(self, hidden_reps, labels):
        if self.centers is None:
            return 0
        
        # 10 here is the number of classes
        x_repeat = hidden_reps.expand(10, hidden_reps.size()[0], 30).transpose(0,1)
        centers_repeat = self.centers.expand(hidden_reps.size()[0], 10, 30)
        
        raw_dists = torch.norm(x_repeat - centers_repeat, dim=2)
        dists = math.e ** ((raw_dists / 4) ** 2 * -1)
        new_dists = dists.clone()
        new_dists[dists < 1e-20] = 1e-20
        
        # TODO: Add option to affect formula clustering loss uses (for both cluster and explosion)
        for j in range(hidden_reps.size()[0]):
            new_dists[j][labels[j]] = raw_dists[j][labels[j]] ** (1/2) * hidden_reps.size()[1] * self.cl_beta
        
        cluster_loss = torch.mean(new_dists) * self.cl_alpha * self.cl_rate
        
        cluster_loss = torch.mean(dists) * self.cl_alpha * self.cl_rate
        return cluster_loss
    
    
    def updateCenters(self):
        if self.cl_rate < 1:
            self.cl_rate += self.cl_rate_speed
        
        num_inps = self.batch_size * self.super_batch_size
        to_cluster = self.stats[:num_inps, :self.hidden_rep_dim].numpy()
        self.kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, random_state=0, batch_size=1600, max_iter = 4, n_init='auto').fit(to_cluster)
        if self.opt.model_type == "classCluster":
            self.centers = torch.zeros(10, self.hidden_rep_dim).cuda()
            for j in range(10):
                    self.centers[j] = torch.mean(self.stats[:num_inps][self.stats[:num_inps, self.hidden_rep_dim] == j][:, :self.hidden_rep_dim], axis=0).cuda()
                    
        if self.opt.model_type == "simpleCluster" or self.opt.model_type == "explodingCluster":
            self.centers = torch.from_numpy(self.kmeans.cluster_centers_).cuda()
            
        if self.opt.model_type == "expandingCluster":
            self.centers = F.normalize(torch.from_numpy(self.kmeans.cluster_centers_).cuda())
            
        if self.opt.model_type == "shiftingCluster":
            C = cdist(self.kmeans.cluster_centers_, self.id_mat.cpu())
            _, self.assignment = linear_sum_assignment(C)

# Old Projection testing code
class ProjNet(nn.Module):
    def __init__(self, ver):
        super().__init__()
        # trng_state = torch.random.get_rng_state();
        # torch.manual_seed(3)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.fc1 = nn.Linear(160, 80)
        self.fc2 = nn.Linear(80, 30)
        self.fc3 = nn.Linear(30, 10)
        # torch.random.set_rng_state(trng_state)
        
        self.cvecs = torch.from_numpy(np.load(str(ver) + '_cvecs.npy')).type(torch.FloatTensor)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = x @ self.cvecs.T

        x = self.fc3(x)
        return x
    
    # Where stats is the hidden reps of another trained net 
    # cvecs = []
    # for j in range(10):
    #     cvec = np.mean(stats[colors == j], axis=0)
    #     cvecs.append(cvec / np.linalg.norm(cvec))
        
    # np.save(str(i) + '_cvecs.npy', np.array(cvecs))