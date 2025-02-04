import torch

def trainNet(trainloader, testloader, net, opt):
    train_accs = []
    test_accs = []
    
    upstage_track, neuron_track, resample_track = stage_planner(1, 6, 4, 2, 12, 6)
    
    for epoch in range(opt.num_epochs):
        hidden_reps = None
        cluster_loss = 0
        running_loss = 0.0
        running_cl = 0.0
        net.stat_index = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].cuda(0), data[1].cuda()
            if opt.model_type != "control" and opt.model_type != "feature":
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
            
            if opt.model_type == "feature" and net.stage == 1:
                loss = cluster_loss
            
            loss.backward()
            net.optimizer.step()
            
            net.stat_index += opt.batch_size
            
            if i % opt.super_batch_size == opt.super_batch_size - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / opt.super_batch_size:.3f} ' + 
                      f'closs: {running_cl / opt.super_batch_size:.3f}')
                
                running_loss = 0.0
                running_cl = 0.0
                net.stat_index = 0
                
                if opt.model_type != "control" and opt.model_type != "feature":
                    net.updateCenters()
        
        if epoch % 6 == 5:
            if opt.model_type == "feature":
                stage = net.stage
                net.stage = 0

            train_acc, test_acc = evalNet(trainloader, testloader, net, opt)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            if opt.model_type == "feature":
                net.stage = stage
        
        # [12, 48, 54, 60, 96, 102, 108, 144, 150, 156]
        # [23, 35, 71, 83, 119, 131]
        # [24, 36, 72, 84, 120, 132]
        
        # if opt.model_type == "feature" and epoch in [2, 4, 20]:
        # if opt.model_type == "feature" and epoch in [11, 23, 29]:
        # if opt.model_type == "feature" and epoch in [11, 29, 35, 41, 59, 65, 71, 89, 95]:
        if opt.model_type == "feature" and epoch in upstage_track:
            net.upstage()
        # if opt.model_type == "feature" and epoch in []:
        # if opt.model_type == "feature" and epoch in [13, 16, 19]:
        # if opt.model_type == "feature" and epoch in [15, 22, 45, 52, 75, 82]:
        if opt.model_type == "feature" and epoch in neuron_track:
            net.track_neurons()
        # elif opt.model_type == "feature" and epoch in []:
        # elif opt.model_type == "feature" and epoch in [14, 17, 20]:
        # if opt.model_type == "feature" and epoch in [16, 23, 46, 53, 76, 83]:
        if opt.model_type == "feature" and epoch in resample_track:
            net.resample_dead_neurons()
                    
    # TODO: Improve model saving process - dates, parameters, performance, final image, etc
    PATH = './MNIST.pth'
    torch.save(net.state_dict(), PATH)

    print(train_accs)
    print(test_accs)
                    
    return net

def stage_planner(initial_e, e_per_0, num_0, num_resamples, e_per_resample, e_per_2):
    curr_epoch = initial_e
    upstage = []
    neuron_track = []
    resample = []
    
    upstage.append(curr_epoch)
    for _i in range(num_0 - 1):
        for _j in range(num_resamples):
            curr_epoch += e_per_resample - 1
            neuron_track.append(curr_epoch)
            curr_epoch += 1
            resample.append(curr_epoch)
        curr_epoch += e_per_resample
        upstage.append(curr_epoch)
        curr_epoch += e_per_2
        upstage.append(curr_epoch)
        curr_epoch += e_per_0
        upstage.append(curr_epoch)
        
    return upstage, neuron_track, resample


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
        train_acc = str(100 * correct // total) + '%'
        
        correct = 0
        total = 0            
        for data in testloader:
            images, labels = data[0].cuda(), data[1].cuda()
            # calculate outputs by running images through the network
            outputs, _, _ = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total > 10000:
                break

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        test_acc = str(100 * correct // total) + '%'

    net.train()
    return train_acc, test_acc