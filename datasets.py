import torchvision
import torchvision.transforms as transforms

import torch

def generate_data_loaders(opt):
    
    if opt.dataset == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5), (0.5))])
        
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                                  shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                             download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                shuffle=False, num_workers=2)
        
        return trainloader, testloader
    
    elif opt.dataset == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                                  shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                shuffle=False, num_workers=2)
        
        return trainloader, testloader
    
    else:
        raise Exception("Only MNIST and CIFAR 10 currently supported")
    