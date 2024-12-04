import torch
import torch.nn as nn  
import torch.nn.functional as F

class ClusterNet(nn.Module):
    def __init__(self, opt):
        self.opt = opt
        self.stats = torch.zeros([10000, opt.hidden_rep_dim + 1])
        self.index = 0
        
        self.kmeans = None
        self.centers = None
        
        self.assignment = None
        self.id_mat = None
        
        self.cl_alpha = opt.cl_alpha
        self.cl_beta = opt.cl_beta
        self.cl_rate = opt.initial_cl_rate
        self.cl_rate_speed = opt.cl_rate_speed
        
        if opt.dataset == 'MNIST':
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 10, 5)
            self.fc1 = nn.Linear(160, 120)
            
            self.fc2 = nn.Linear(120, opt.hidden_rep_dim)
            self.fc3 = nn.Linear(opt.hidden_rep_dim, 10)
            
            if opt.batch_norm:
                self.bn = nn.BatchNorm1d(opt.hidden_rep_dim)
            
        elif opt.dataset == 'CIFAR10':
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16*5*5, 120)
            
            self.fc2 = nn.Linear(120, opt.hidden_rep_dim)
            self.fc3 = nn.Linear(opt.hidden_rep_dim, 10)
            
            if opt.batch_norm:
                self.bn = nn.BatchNorm1d(opt.hidden_rep_dim)
                
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
            
        self.stats[self.index:self.index+self.opt.batch_size, :self.opt.hidden_rep_dim] = x.detach().clone().cpu()
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
            
        x = F.relu(x)
        x = self.fc3(x)
    
        return x, cluster_loss, hidden_reps