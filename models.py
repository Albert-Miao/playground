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
        self.stats = torch.zeros([50000, opt.hidden_rep_dim + 1])
        self.stat_index = 0
        
        self.hidden_rep_dim = opt.hidden_rep_dim
        self.batch_size = opt.batch_size
        self.super_batch_size = opt.super_batch_size
        
        self.criterion = nn.CrossEntropyLoss()
        
        if opt.dataset == 'MNIST':
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 10, 5)
            
            self.drop = nn.Dropout(0.3)
            
            self.fc1 = nn.Linear(160, 120)
            
            self.fc2 = nn.Linear(120, self.hidden_rep_dim)
            self.fc3 = nn.Linear(self.hidden_rep_dim, 20)
            
            self.fc4 = nn.Linear(20, 10)
            
            if opt.batch_norm:
                self.bn = nn.BatchNorm1d(self.hidden_rep_dim)
            
        elif opt.dataset == 'CIFAR10':
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            
            self.drop = nn.Dropout(0.3)
            
            self.fc1 = nn.Linear(16*5*5, 120)
            
            self.fc2 = nn.Linear(120, self.hidden_rep_dim)
            self.fc3 = nn.Linear(self.hidden_rep_dim, 20)
            
            self.fc4 = nn.Linear(20, 10)
            
            if opt.batch_norm:
                self.bn = nn.BatchNorm1d(self.hidden_rep_dim)
                
        self.optimizer = optim.SGD(self.parameters(), lr=opt.lr, momentum=opt.momentum)
        
    def forward(self, x):
        hidden_reps = 0
        cluster_loss = 0
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        
        x = self.drop(x)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        if self.opt.batch_norm:
            x = self.bn(x)
        
        self.stats[self.stat_index:self.stat_index+x.size()[0], :self.hidden_rep_dim] = x.detach().clone().cpu()

        x = F.relu(x)
        x = F.relu(self.fc3(x))
        
        x = self.fc4(x)
    
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

class FeatureNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        
        self.hidden_rep_dim = opt.hidden_rep_dim
        self.sae_dim = opt.num_clusters
        self.batch_size = opt.batch_size
        self.super_batch_size = opt.super_batch_size
        self.cl_alpha = opt.cl_alpha
        self.cl_beta = opt.cl_beta
        
        self.stats = torch.zeros([50000, self.sae_dim + 1])
        self.stat_index = 0

        self.stage = 0
        self.tracking_neurons = False
        self.neuron_record = torch.empty(0, self.sae_dim).cuda()
        self.feature_loss_record = torch.empty(0).cuda()
        self.input_record = torch.empty(0, 20).cuda()
        self.residual_record = torch.empty(0, 20).cuda()
        
        self.criterion = nn.CrossEntropyLoss()
        
        if opt.dataset == 'MNIST':
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 10, 5)
            self.fc1 = nn.Linear(160, 120)
            
            self.fc2 = nn.Linear(120, self.hidden_rep_dim)
            self.fc3 = nn.Linear(self.hidden_rep_dim, 20)
            
            self.sae1 = nn.Linear(20, self.sae_dim)
            self.sae2 = nn.Linear(self.sae_dim, 20)
            
            self.ln1 = nn.LayerNorm(20)
            self.ln2 = nn.LayerNorm(20)
            
            self.q = nn.Linear(self.sae_dim, 20)
            self.k = nn.Linear(20, 20)
            self.v = nn.Linear(self.sae_dim, 20)
            
            self.fc4 = nn.Linear(20, 10)
            
            
            self.bn = nn.BatchNorm1d(self.hidden_rep_dim)
            
        elif opt.dataset == 'CIFAR10':
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            
            self.drop = nn.Dropout(0.3)
            
            self.fc1 = nn.Linear(16*5*5, 120)
            
            self.fc2 = nn.Linear(120, self.hidden_rep_dim)
            self.fc3 = nn.Linear(self.hidden_rep_dim, 20)
            
            self.sae1 = nn.Linear(20, self.sae_dim)
            self.sae2 = nn.Linear(self.sae_dim, 20)
            
            self.ln1 = nn.LayerNorm(20)
            self.ln2 = nn.LayerNorm(20)
            self.ln3 = nn.LayerNorm((self.sae_dim, 20))
            self.ln4 = nn.LayerNorm(20)
            self.ln5 = nn.LayerNorm(20)
            
            self.q = nn.Linear(20, 20, bias=False)
            self.k = nn.Linear(20, 20, bias=False)
            self.v = nn.Linear(20, 20, bias=False)
            
            self.fc4 = nn.Linear(20, 40)
            self.fc5 = nn.Linear(40, 20, bias=False)
            
            self.w0 = nn.Linear(20, 20)
            
            self.fc6 = nn.Linear(20, 10)
            
            self.bn = nn.BatchNorm1d(self.hidden_rep_dim)
                
        self.optimizer = optim.SGD(self.parameters(), lr=opt.lr, momentum=opt.momentum)
        
    def upstage(self):
        # Stage 0: No SAE
        # Stage 1: SAE training + resample
        # Stage 2: SAE training + resample
        # Stage 4: SAE training
        # Stage 5: Transformer training
        # Stage 6: All training
        
        # Without resample
        # Stage 0: No SAE
        # Stage 1: SAE training
        # Stage 2: Transformer training
        # Stage 3: All training
        
        self.stage += 1
        
        if self.stage % 3 == 1: 
            self.conv1.requires_grad_(False)
            self.conv2.requires_grad_(False)
            self.pool.requires_grad_(False)
            self.bn.requires_grad_(False)
            
            self.fc1.requires_grad_(False)
            self.fc2.requires_grad_(False)
            self.fc3.requires_grad_(False)
            self.ln1.requires_grad_(False)
            self.fc6.requires_grad_(False)
            
            # self.optimizer = optim.Adam(self.parameters(), lr=self.opt.lr / (self.stage ** (1/2)), momentum=self.opt.momentum)
            self.optimizer = optim.Adam(self.parameters(), lr=self.opt.lr / (self.stage ** (1/2)))
        
        elif self.stage % 3 == 2:
            self.conv1.requires_grad_(True)
            self.conv2.requires_grad_(True)
            self.pool.requires_grad_(True)
            self.bn.requires_grad_(True)
            
            self.fc1.requires_grad_(True)
            self.fc2.requires_grad_(True)
            self.fc3.requires_grad_(True)
            self.ln1.requires_grad_(True)
            self.fc6.requires_grad_(True)
                
            
            self.sae1.requires_grad_(False)
            self.sae2.requires_grad_(False)
            
            self.optimizer = optim.SGD(self.parameters(), lr=self.opt.lr / (self.stage - 1), momentum=self.opt.momentum)
            
        elif self.stage % 3 == 0:
            self.sae1.requires_grad_(True)
            self.sae2.requires_grad_(True)
            
            # nn.init.xavier_normal_(self.sae1.weight)
            # nn.init.normal_(self.sae1.bias)
            # nn.init.xavier_normal_(self.sae2.weight)
            # nn.init.normal_(self.sae2.bias)        
        
        # if self.stage == 1:
        #     self.conv1.requires_grad_(False)
        #     self.conv2.requires_grad_(False)
        #     self.pool.requires_grad_(False)
        #     self.bn.requires_grad_(False)
            
        #     self.fc1.requires_grad_(False)
        #     self.fc2.requires_grad_(False)
        #     self.fc3.requires_grad_(False)
        #     self.ln1.requires_grad_(False)
        #     self.fc6.requires_grad_(False)
            
        #     self.optimizer = optim.SGD(self.parameters(), lr=self.opt.lr, momentum=self.opt.momentum)
        
        # elif self.stage == 2:
        #     self.sae1.requires_grad_(False)
        #     self.sae2.requires_grad_(False)
            
        #     self.optimizer = optim.SGD(self.parameters(), lr=self.opt.lr, momentum=self.opt.momentum)
        # elif self.stage == 3:
        #     self.conv1.requires_grad_(True)
        #     self.conv2.requires_grad_(True)
        #     self.pool.requires_grad_(True)
        #     self.bn.requires_grad_(True)
            
        #     self.fc1.requires_grad_(True)
        #     self.fc2.requires_grad_(True)
        #     self.fc3.requires_grad_(True)
        #     self.ln1.requires_grad_(True)
            
        #     self.sae1.requires_grad_(True)
        #     self.sae2.requires_grad_(True)
            
        #     self.fc6.requires_grad_(True)
            
            # self.optimizer = optim.SGD(self.parameters(), lr=self.opt.lr / 2, momentum=self.opt.momentum)
        
    def eval(self):
        super().eval()
        # self.stage = 0
        
    def track_neurons(self):
        self.tracking_neurons = True
        
    def forward(self, x):
        with torch.no_grad():
            self.sae2.weight.copy_(F.normalize(self.sae2.weight, dim=1))
        
        f = 0
        feature_loss = 0
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        if self.opt.batch_norm:
            x = self.bn(x)

        x = F.relu(x)
        x = F.relu(self.fc3(x))
        # x = self.ln1(x)
        
        if self.stage % 3 != 0:
            _x = x - self.sae2.bias
            input_x = _x
            f = F.relu(self.sae1(_x))
            _x = self.sae2(f)

            self.stats[self.stat_index:self.stat_index+x.size()[0], :self.sae_dim] = f.detach().clone().cpu()

            if self.stage % 3 == 1:
                recon_loss_arr = self.cl_alpha * torch.linalg.vector_norm(x - _x, dim=1)
                recon_loss = torch.sum(recon_loss_arr)
                
                l1_loss = torch.sum(self.cl_beta * torch.linalg.vector_norm(f, ord=1, dim=1)) 
                
                # feature_loss_arr = self.cl_alpha * torch.linalg.vector_norm(x - _x, dim=1) + self.cl_beta * torch.linalg.vector_norm(f, ord=1, dim=1)
                # feature_loss = torch.sum(feature_loss_arr)
                
                feature_loss = (recon_loss, l1_loss)
                
                true_x = x
            else:
                x = _x
                
        # elif self.stage >= 2:
            # feature_loss = 0
            # for _ in range(1):
            #     prev_x = x
            #     #diagembed to get array of relevant dictionary vectors
            #     if self.stage == 3:
            #         _x = x - self.sae2.bias
            #         f = F.relu(self.sae1(_x))
            #         _x = self.sae2(f)
                    
            #         _, inds = f.sort(dim=1, descending=True)
            #         sparse_f = self.sae2(torch.diag_embed(f))
            #         decoded_f = torch.zeros(sparse_f.size()).cuda()
            #         for i in range(sparse_f.size(0)):
            #             decoded_f[i] = sparse_f[i][inds[i]]
            #     else: 
            #         _x = x - self.sae2.bias.detach()
            #         f = F.relu(F.linear(_x, self.sae1.weight.detach(), self.sae1.bias.detach()))
            #         _x = F.linear(f, self.sae2.weight.detach(), self.sae2.bias.detach())
                    
            #         _, inds = f.sort(dim=1, descending=True)
            #         sparse_f = F.linear(torch.diag_embed(f), self.sae2.weight.detach(), self.sae2.bias.detach())
            #         decoded_f = torch.zeros(sparse_f.size()).cuda()
            #         for i in range(sparse_f.size(0)):
            #             decoded_f[i] = sparse_f[i][inds[i]]
                    
                    
            #     feature_loss += self.cl_alpha * torch.linalg.vector_norm(x - _x) + self.cl_beta * torch.linalg.vector_norm(f, ord=1)
                
                # f = F.relu(self.f_to_trans(f))
                # res = x - _x
                
                # for i in range(decoded_f.size(0)):
                #     iso_f = decoded_f[i][f[i][inds[i]] != 0]
                #     num_f = iso_f.size(0)
                    
                #     iso_f = iso_f + res[i].unsqueeze(0).repeat(num_f, 1)
                    
                #     q_f = self.q(iso_f)
                #     k_f = self.k(iso_f)
                #     v_f = self.v(iso_f)
                    
                #     iso_f = F.softmax(q_f @ k_f.T * 10, dim=1) @ v_f
                #     iso_f = self.w0(iso_f)
                #     x[i] = torch.sum(iso_f, dim=0) / (num_f ** (1/2))
                
                
                # x = x + prev_x
                # x = self.ln4(x)
                # prev_x = x
                
                # x = F.relu(self.fc4(x))
                # x = self.fc5(x)
                
                # x = x + prev_x
                # x = self.ln5(x)
            
            # feature_loss = feature_loss / 1
        
        x = self.fc6(x)
        
        if self.tracking_neurons and self.training:
            self.neuron_record = torch.cat((self.neuron_record, f), dim=0)
            # self.feature_loss_record = torch.cat((self.feature_loss_record, feature_loss_arr), dim=0)
            self.feature_loss_record = torch.cat((self.feature_loss_record, recon_loss_arr), dim=0)
            self.input_record = torch.cat((self.input_record, input_x), dim=0)
            self.residual_record = torch.cat((self.residual_record, true_x - _x), dim=0)
        
        return x, feature_loss, f
    
    def resample_dead_neurons(self):
        print(torch.mean(torch.sum(self.neuron_record != 0, dim=1).double()))
        neuron_record = torch.sum(self.neuron_record, dim=0)
        dead_inds = (neuron_record == 0).nonzero()[:, 0]
        alive_inds = (neuron_record != 0).nonzero()[:, 0]
        
        if dead_inds.size(0) != 0:
            with torch.no_grad():
                print(dead_inds)
                input_probs = self.feature_loss_record ** 2
                new_vecs = F.normalize(self.input_record[torch.multinomial(input_probs, dead_inds.size(0))])
                # new_vecs = F.normalize(self.residual_record[torch.multinomial(input_probs, dead_inds.size(0))])
                new_norm = torch.mean(torch.norm(self.sae1.weight[alive_inds],dim=1)) * 0.2
                
                self.sae1.weight[dead_inds] = new_vecs * new_norm
                self.sae1.bias[dead_inds] = torch.zeros(dead_inds.size(0)).cuda()
                
                self.sae2.weight[:, dead_inds] = new_vecs.T
                
                # opt_param_size = self.optimizer.param_groups[0]['params'][10].size()
                
                # if opt_param_size[0] == 80 and opt_param_size[1] == 20:
                #     self.optimizer.param_groups[0]['params'][10][dead_inds] = torch.ones(new_vecs.size()).cuda()
                #     self.optimizer.param_groups[0]['params'][11][dead_inds] = torch.zeros(new_vecs.size(0)).cuda()
                #     self.optimizer.param_groups[0]['params'][12][:, dead_inds] = torch.ones(new_vecs.size()).cuda().T
                # else:
                #     print("Couldn't find head of SAE in optmizer params! Did you modify the layer order you silly bean?")
        
        self.tracking_neurons = False
        self.neuron_record = torch.empty(0, self.sae_dim).cuda()
        self.feature_loss_record = torch.empty(0).cuda()
        self.input_record = torch.empty(0, 20).cuda()
        self.residual_record = torch.empty(0, 20).cuda()

# stage0                       stage1        stage2        stage0        stage1        stage2        stage0
# ['53%', '60%', '65%', '69%', '68%', '68%', '70%', '71%', '74%', '74%', '75%', '75%', '77%', '79%', '79%', '80%']
# ['52%', '57%', '61%', '63%', '63%', '63%', '63%', '65%', '66%', '66%', '66%', '66%', '68%', '68%', '69%', '68%']

# control
# ['50%', '61%', '67%', '69%', '72%', '72%', '74%', '74%', '77%', '77%', '77%', '78%', '78%', '79%', '80%', '79%']
# ['49%', '58%', '63%', '63%', '66%', '65%', '66%', '67%', '67%', '67%', '67%', '67%', '67%', '67%', '67%', '67%']

# ['61%',        '70%',        '73%',        '75%',        '80%',        '80%',        '82%',        '83%']
# ['58%',        '64%',        '66%',        '66%',        '68%',        '68%',        '68%',        '69%'

#0.04
# stage0                       stage1                                    stage2        stage0        stage1                                    stage2        stage0        stage1                                    stage2        stage0       
# ['52%', '61%', '65%', '67%', '67%', '67%', '67%', '68%', '68%', '67%', '70%', '71%', '74%', '74%', '74%', '75%', '74%', '74%', '74%', '74%', '77%', '77%', '81%', '80%', '80%', '80%', '81%', '80%', '81%', '80%', '81%', '81%', '82%', '82%']
# ['50%', '59%', '62%', '63%', '63%', '63%', '63%', '63%', '63%', '63%', '65%', '65%', '65%', '66%', '66%', '66%', '66%', '66%', '66%', '66%', '67%', '67%', '68%', '68%', '68%', '68%', '68%', '68%', '68%', '68%', '68%', '68%', '69%', '68%']

# 0.01
# ['60%', '69%', '70%', '70%', '70%', '70%', '69%', '70%', '71%', '76%', '74%', '75%', '75%', '74%', '75%', '74%', '79%', '80%', '80%', '81%', '81%', '80%', '80%', '81%', '81%', '83%']
# ['59%', '65%', '64%', '64%', '64%', '64%', '64%', '64%', '65%', '66%', '66%', '66%', '66%', '66%', '66%', '66%', '68%', '68%', '69%', '69%', '69%', '69%', '69%', '69%', '68%', '69%']

# stage0         stage1                                    stage2 stage0 stage1                                    stage2 stage0 stage1                                    stage2 stage0
# ['61%', '70%', '67%', '68%', '68%', '68%', '68%', '67%', '73%', '75%', '75%', '76%', '76%', '76%', '76%', '76%', '80%', '80%', '81%', '82%', '81%', '81%', '81%', '81%', '82%', '83%']
# ['58%', '64%', '62%', '62%', '62%', '62%', '62%', '62%', '66%', '66%', '67%', '67%', '67%', '67%', '67%', '67%', '68%', '68%', '68%', '68%', '68%', '68%', '68%', '68%', '68%', '69%']

# ['54%', '63%', '62%', '62%', '62%', '63%', '63%', '62%', '65%', '68%', '66%', '67%', '66%', '65%', '65%', '67%', '70%', '70%', '70%', '71%', '71%', '70%', '71%', '71%', '71%', '72%']
# ['54%', '62%', '63%', '63%', '63%', '63%', '63%', '63%', '64%', '66%', '65%', '65%', '65%', '65%', '65%', '65%', '68%', '68%', '68%', '68%', '68%', '68%', '68%', '68%', '69%', '69%']

# ['53%', '57%', '60%', '63%', '64%', '65%', '65%', '65%']
# ['55%', '60%', '61%', '64%', '65%', '66%', '66%', '67%']

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