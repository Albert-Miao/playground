import torch
import torch.nn as nn  
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import MiniBatchKMeans



import math

from options import PlaygroundOptions
from datasets import generate_data_loaders
from models import ClusterNet, NaiveNet
    
def test_proj(ver):

    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
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

            if flag:
                stats[count, :10] = x[0]

            x = self.fc3(x)
            return x

    PATH = './MNIST.pth'

    net = Net(ver)
    net.load_state_dict(torch.load(PATH, weights_only=True))
    net.cuda()

    count = 0
    flag = True
    
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            stats[count, 10] = labels[0]
            # calculate outputs by running images through the network
            outputs = net(images)
            count += 1
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def main():
    options = PlaygroundOptions()
    opt = options.parse()
    torch.cuda.set_device(opt.gpu)
    
    trainloader, testloader = generate_data_loaders(opt)
    net = None
    if opt.model_type == "control":
        net = NaiveNet(opt)
    else:
        net = ClusterNet(opt)
    
    
    for epoch in range(opt.num_epochs):
        hidden_reps = None
        cluster_loss = 0
        running_loss = 0.0
        running_cl = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].cuda(0), data[1].cuda()
            if opt.model_type != "control":
                net.stats[net.stat_index:net.stat_index+opt.batch_size, opt.hidden_rep_dim] = labels

            # zero the parameter gradients
            net.optimizer.zero_grad()
            
            outputs, cluster_loss, hidden_reps = net(inputs)
            loss = net.criterion(outputs, labels)
            
            if opt.model_type == 'explodingCluster':
                cluster_loss = net.classExplodeClusterLoss(hidden_reps, labels)
            
            running_loss += loss.item()
            running_cl += cluster_loss
            loss += cluster_loss
            
            loss.backward()
            net.optimizer.step()
            
            net.stat_index += opt.batch_size
            
            if i % opt.super_batch_size == opt.super_batch_size - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / opt.super_batch_size:.3f} 
                      closs: {running_cl / opt.super_batch_size:.3f}')
                
                running_loss = 0.0
                running_cl = 0.0
                
                if opt.model_type != "control":
                    net.stat_index = 0
                    net.updateCenters()
                    
    print('Finished Training')
    
    # TODO: Improve model saving process - dates, parameters, performance, final image, etc
    PATH = './MNIST.pth'
    torch.save(net.state_dict(), PATH)
    
    net.eval()
    net.stat_index = 0
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in trainloader:
            images, labels = data[0].cuda(), data[1].cuda()
            # calculate outputs by running images through the network
            outputs, _, _ = net(images)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total > 10000:
                break
        print(f'Accuracy of the network on the first 10000 train images: {100 * correct // total} %')
        
        correct = 0
        total = 0            
        for data in testloader:
            images, labels = data[0].cuda(), data[1].cuda()
            net.stats[net.stat_index, 30] = labels[0]
            # calculate outputs by running images through the network
            outputs, _, _ = net(images)
            net.stat_index += 1
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total > 10000:
                break

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    

# TODO: write code to evaluate the linear seperability of images
# TODO: implement own self generating pipeline
if __name__ == "__main__":
    
    # for i in range(1):
    #     stats = torch.zeros([10000, 31])
    #     cluster_train()
    #     np.save(str(i) + '.npy', stats.numpy())
    
    for i in range(1):
        stats = np.zeros([10000, 31])
        run()
        np.save(str(i) + '.npy', stats)
    
    # for i in range(6):
    #     stats = np.zeros([10000, 11])
    #     test_proj(i)
    #     np.save(str(i) + '_proj.npy', stats)

    fig, axs = plt.subplots(3, 5)
    for i in range(15):
        print('running')
        stats = np.load(str(0)+'.npy')
        
        colors = stats[:, 30].astype('int')

        stats = stats[:, :30]
        stats = stats[np.logical_or(colors == 4, colors == 9)]
        
        colors = colors[np.logical_or(colors == 4, colors == 9)]
        
        # U, s, Vt = np.linalg.svd(stats, full_matrices=False)
        # V = Vt.T
        # S = np.diag(s)
        c_arr = ['r', 'g', 'b', 'c', 'gray', 'orange', 'purple', 'black', 'yellow', 'pink']
        c = [c_arr[x] for x in colors]
        x = (stats[:500, 0] - np.mean(stats[:500, 0])) / np.std(stats[:500, 0])
        y = (stats[:500, 1] - np.mean(stats[:500, 1])) / np.std(stats[:500, 1])
        axs[i % 3, i // 3].scatter(stats[:500, 2 * i], stats[:500, 2 * i + 1], c=c[:500])
        
        # cvecs = []
        # for j in range(10):
        #     cvec = np.mean(stats[colors == j], axis=0)
        #     cvecs.append(cvec / np.linalg.norm(cvec))
            
        # np.save(str(i) + '_cvecs.npy', np.array(cvecs))
        
    plt.show()