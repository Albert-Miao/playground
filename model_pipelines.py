import torch

def trainNet(trainloader, testloader, net, opt):
    train_accs = []
    test_accs = []
    
    upstage_track, neuron_track, resample_track = stage_planner(12, 6, 10, 2, 12, 6)
    if opt.debug_load_pth == True:
        upstage_track, neuron_track, resample_track = stage_planner(0, 6, 1, 1, 3, 6)
    
    for epoch in range(opt.num_epochs):
        hidden_reps = None
        cluster_loss = 0
        running_loss = 0.0
        running_cl = 0.0
        if opt.model_type == "feature":
            running_l1 = 0.0
        net.stat_index = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].cuda(0), data[1].cuda()
            if opt.model_type != "control" and opt.model_type != "feature":
                net.stats[net.stat_index:net.stat_index+opt.batch_size, opt.hidden_rep_dim] = labels
            if opt.model_type == "feature" and net.stage % 3 == 2:
                # Num clusters here is sae dim, I know, I know, I promise if this project gets approval I'll rewrite.
                net.stats[net.stat_index:net.stat_index+opt.batch_size, opt.num_clusters] = labels

            # zero the parameter gradients
            net.optimizer.zero_grad()
            
            outputs, cluster_loss, hidden_reps = net(inputs)
            loss = net.criterion(outputs, labels)
            
            if opt.model_type == 'classCluster':
                cluster_loss = net.classExplodeClusterLoss(hidden_reps, labels)
            
            running_loss += loss.item()
            
            if opt.model_type == "feature" and net.stage == 1:
                running_cl += cluster_loss[0]
                running_l1 += cluster_loss[1]
                loss += cluster_loss[0]
                loss += cluster_loss[1]
            else:
                running_cl += cluster_loss
                loss += cluster_loss
            
            loss.backward()
            net.optimizer.step()
            
            net.stat_index += opt.batch_size
            
            if i % opt.super_batch_size == opt.super_batch_size - 1:
                
                if opt.model_type == "feature":
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / opt.super_batch_size:.3f} ' + 
                      f'reconstruction loss: {running_cl / opt.super_batch_size:.3f} ' + 
                      f'l1 loss: {running_l1 / opt.super_batch_size:.3f}')
                else:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / opt.super_batch_size:.3f} ' + 
                        f'closs: {running_cl / opt.super_batch_size:.3f}')
                    
                running_loss = 0.0
                running_cl = 0.0
                running_l1 = 0.0
                
                if opt.model_type != "control" and opt.model_type != "feature":
                    net.stat_index = 0
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
        

        if opt.model_type == "feature" and epoch in upstage_track:
            net.upstage()
        if opt.model_type == "feature" and epoch in neuron_track:
            net.track_neurons()
        if opt.model_type == "feature" and epoch in resample_track:
            net.resample_dead_neurons()
                    
    # TODO: Improve model saving process - dates, parameters, performance, final image, etc
    PATH = './MNIST.pth'
    torch.save(net.state_dict(), PATH)

    print(train_accs)
    print(test_accs)
                    
    return net

def stage_planner(initial_e, e_per_0, num_0, num_resamples, e_per_resample, e_per_2):
    curr_epoch = initial_e - 1
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