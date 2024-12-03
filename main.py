import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 32

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import MiniBatchKMeans

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# functions to show an image

import math


def fibonacci_sphere(samples=30):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


global stats
global flag
global count
global kmeans
global cluster_loss
global cl_alpha
global cl_beta
global cl_ramp
global id_mat

global hidden_reps

global assignment

def run():
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            # trng_state = torch.random.get_rng_state();
            # torch.manual_seed(3)
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16*5*5, 120)
            self.fc2 = nn.Linear(120, 30)
            self.bn = nn.BatchNorm1d(30)
            self.fc3 = nn.Linear(30, 10)
            # torch.random.set_rng_state(trng_state)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = self.bn(x)
            if flag:
                stats[count:count+x.size()[0], :30] = x.detach().clone().cpu()
            x = F.relu(x)
            # x = F.normalize(x)
            
            x = self.fc3(x)
            return x

    flag = False
    net = Net()
    net.to(device)

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.004, momentum=0.9)

    for epoch in range(3):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % (8000 / batch_size) == 8000 / batch_size - 1:   # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 8000 * batch_size:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './MNIST.pth'
    torch.save(net.state_dict(), PATH)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images

    net = Net()
    net.load_state_dict(torch.load(PATH, weights_only=True))
    net.to(device)
    net.eval()

    count = 0
    flag = True
    
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            stats[count, 30] = data[1]
            # calculate outputs by running images through the network
            outputs = net(images)
            count += 1
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    # correct_pred = {classname: 0 for classname in classes}
    # total_pred = {classname: 0 for classname in classes}

    # # again no gradients needed
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predictions = torch.max(outputs, 1)
    #         # collect the correct predictions for each class
    #         for label, prediction in zip(labels, predictions):
    #             if label == prediction:
    #                 correct_pred[classes[label]] += 1
    #             total_pred[classes[label]] += 1


    # print accuracy for each class
    # for classname, correct_count in correct_pred.items():
    #     accuracy = 100 * float(correct_count) / total_pred[classname]
    #     print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
    
def cluster_train():
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            # trng_state = torch.random.get_rng_state();
            # torch.manual_seed(3)
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 10, 5)
            self.fc1 = nn.Linear(160, 120)
            
            
            self.fc2 = nn.Linear(120, 30)
            self.bn = nn.BatchNorm1d(30)
            self.fc3 = nn.Linear(30, 10)
            # torch.random.set_rng_state(trng_state)

        def forward(self, x):
            cluster_loss = 0
            hidden_reps = 0
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            
            
            x = self.fc2(x)
            
            # x = F.normalize(x)
            # x = self.bn(x)
                        
            if flag:
                stats[count:count+x.size()[0], :30] = x.detach().clone().cpu()
                hidden_reps = x.clone()

                # if not kmeans is None:
                #     # Encourages similar points to cluster together - simple cluster training
                #     # x_clusters = kmeans.cluster_centers_[kmeans.predict(x.detach().cpu().numpy())]
                #     # dists = torch.sum((x - torch.from_numpy(x_clusters).to(device)) ** 2 / x.size()[1], dim=1)
                    
                #     # Exploding clusters - encourages points to move away from all clusters except their own, but also stick to own cluster
                #     center_inds = kmeans.predict(x.detach().cpu().numpy())
                    
                #     # (number of concepts, batch size, dimension/channels) -> (batch size, number of concepts, dimension / channels)
                #     x_repeat = x.expand(30, x.size()[0], 30).transpose(0,1)
                #     centers_repeat = centers.expand(x.size()[0], 30, 30)
                    
                #     # Here we encourage points close to other centers to drift away from them based off of inverse distance. Maybe explore negative?
                #     dists = 1 / torch.norm(x_repeat - centers_repeat, dim=2)
                #     dists[dists > 5] = 5
                    
                #     # On the other hand, points will move closer to their own centers based off of square root. Also multiply by number of concepts and a relative beta term
                #     for i in range(x.size()[0]):
                #         dists[i][center_inds[i]] = (1 / dists[i][center_inds[i]]) ** (1 / 2) * 30 * cl_beta
                    
                #     # Encourages clusters to move outwards - expanding cluster training 
                #     # (NEEDS REWRITING, currently pushes clusters outwards but collapses clusters at boundary)
                #     # cluster_inds = kmeans.predict(x.detach().cpu().numpy())
                #     # x_clusters = centers[cluster_inds]
                #     # dists = torch.sum((x - x_clusters) ** 2 / x.size()[1], dim=1)
                    
                #     # Encourages clusters to move to predefined 'thin' vectors - shifting cluster training
                #     # x_clusters = id_mat[assignment[kmeans.predict(x.detach().cpu().numpy())]]
                #     # dists = torch.sum((x - x_clusters) ** 2 / x.size()[1], dim=1)
                #     cluster_loss = torch.mean(dists) * cl_alpha * cl_rate
                    
            
            x = F.relu(x)
            x = self.fc3(x)
            return x, cluster_loss, hidden_reps

    flag = True
    cl_alpha = 100
    cl_beta = 0.001
    cl_rate = 0
    cluster_loss = 0
    kmeans = None
    assignment = None
    net = Net()
    net.to(device)
    
    centers = None
    
    class_centers = torch.zeros(10, 30).to(device)
    
    # id_mat = torch.eye(30).to(device).to(int)
    # test = torch.tensor([((1 - (0.8) ** 2) / 29) ** (1/2), 0.8]).to(device)
    # id_mat = test[id_mat]
    

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.004, momentum=0.9)

    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(3):  # loop over the dataset multiple times
        count = 0
        running_loss = 0.0
        running_cl = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, cluster_loss, hidden_reps = net(inputs)
            loss = criterion(outputs, labels)
            
            # Testing class-based expanding clustering
            stats[count:count+labels.size()[0], 30] = labels
            if not kmeans is None:
                x_repeat = hidden_reps.expand(10, hidden_reps.size()[0], 30).transpose(0,1)
                centers_repeat = class_centers.expand(hidden_reps.size()[0], 10, 30)
                
                
                raw_dists = torch.norm(x_repeat - centers_repeat, dim=2)
                dists = math.e ** ((raw_dists / 4) ** 2 * -1)
                new_dists = dists.clone()
                new_dists[dists < 1e-20] = 1e-20
                
                for j in range(hidden_reps.size()[0]):
                    new_dists[j][labels[j]] = raw_dists[j][labels[j]] ** (1/2) * 10 * cl_beta
                
                cluster_loss = torch.mean(new_dists) * cl_alpha * cl_rate
                
            
            
            running_loss += loss.item()
            if cl_rate > 0:
                running_cl += cluster_loss
            loss += cluster_loss
            
            
            loss.backward()
            optimizer.step()

            count += batch_size

            # print statistics
            if i % (8000 / batch_size) == 8000 / batch_size - 1:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 8000 * batch_size:.3f} closs: {running_cl / 8000 * batch_size:.3f}')
                count = 0
                if cl_rate < 1:
                    cl_rate += 0.17
                to_cluster = stats[:8000, :30].numpy()
                kmeans = MiniBatchKMeans(n_clusters=30, random_state=0, batch_size=256, max_iter = 4, n_init='auto').fit(to_cluster)
                
                # Class cluster center identification
                for j in range(10):
                    class_centers[j] = torch.mean(stats[:8000][stats[:8000, 30] == j][:, :30], axis=0).to(device)
                
                # For exploding or simple clustering
                # CHECK TO SEE IF CENTERS IS ORIENTED PROPERLY (IE (NUMBER OF CLUSTERS X DIMENSIONS) VS (DIMENSIONS X NUMBER OF CLUSTERS))
                # centers = torch.from_numpy(kmeans.cluster_centers_).to(device)
                
                # expanding centers: cube rt to magnify centers
                # centers = F.normalize(torch.from_numpy(kmeans.cluster_centers_).to(device))
                
                # Or shifting centers
                # C = cdist(kmeans.cluster_centers_, id_mat.cpu())
                # _, assignment = linear_sum_assignment(C)
                running_loss = 0.0
                running_cl = 0.0

    print('Finished Training')

    PATH = './MNIST.pth'
    torch.save(net.state_dict(), PATH)
    
    kmeans = None
    # print images

    net = Net()
    net.load_state_dict(torch.load(PATH, weights_only=True))
    net.to(device)
    net.eval()

    count = 0
    flag = True
    
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            # stats[count, 30] = labels[0]
            # calculate outputs by running images through the network
            outputs, _, _ = net(images)
            # count += 1
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total > 10000:
                break
        print(f'Accuracy of the network on the 10000 train images: {100 * correct // total} %')
        correct = 0
        total = 0        
            
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            stats[count, 30] = labels[0]
            # calculate outputs by running images through the network
            outputs, _, _ = net(images)
            count += 1
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total > 10000:
                break

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    
    
def test_proj(ver):
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

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
    net.to(device)

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

# TODO: write code to evaluate the linear seperability of images
# TODO: implement own self generating pipeline
if __name__ == "__main__":
    
    # for i in range(1):
    #     stats = torch.zeros([10000, 31])
    #     cluster_train()
    #     np.save(str(i) + '.npy', stats.numpy())
    
    # for i in range(1):
    #     stats = np.zeros([10000, 31])
    #     run()
    #     np.save(str(i) + '.npy', stats)
    
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