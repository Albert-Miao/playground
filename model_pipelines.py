import torch

def trainNet(trainloader, net, opt):
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
                    
    # TODO: Improve model saving process - dates, parameters, performance, final image, etc
    PATH = './MNIST.pth'
    torch.save(net.state_dict(), PATH)
                    
    return net


def evalNet(trainloader, testloader, net, opt):
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
