import torchvision
import torchvision.transforms as transforms

import torch

def generate_data_loaders(opt):
    
    if opt.dataset == 'MNIST':
        
        
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5), (0.5))])
            
        trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                            download=True, transform=transform)

        testset = torchvision.datasets.MNIST(root='../data', train=False,
                                            download=True, transform=transform)
        
            
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                        shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                        shuffle=False, num_workers=2)
        
        return trainloader, testloader
    
    elif opt.dataset == 'CIFAR10':
        transform_train = transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
                                      transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                      transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                    #   transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                      transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
                               ])
        
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                                  shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                shuffle=False, num_workers=2)
        
        return trainloader, testloader
    
    else:
        raise Exception("Only MNIST and CIFAR 10 currently supported")
    