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
from model_pipelines import trainNet


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
    
    trainNet(trainloader, net, opt)
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