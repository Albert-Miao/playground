import torch
import numpy as np

from options import PlaygroundOptions
from datasets import generate_data_loaders
from models import ClusterNet, NaiveNet, FeatureNet
from model_pipelines import trainNet, evalNet
from visualizations import viewHiddenReps, classSeparabilityEval

def train(opt):
    torch.cuda.set_device(opt.gpu)
    print(torch.cuda.get_device_name(0))
    net = None
    if opt.model_type == "control":
        net = NaiveNet(opt).cuda()
        if opt.debug_load_pth == True:
            net.load_state_dict(torch.load("control_long.pth", weights_only=True))
    elif opt.model_type == "feature":
        net = FeatureNet(opt).cuda()
        if opt.debug_load_pth == True:
            net.load_state_dict(torch.load("sae_test.pth", weights_only=True))
    else:
        net = ClusterNet(opt).cuda()
        
    trainloader, testloader = generate_data_loaders(opt)
    trainNet(trainloader, testloader, net, opt)
    evalNet(trainloader, testloader, net, opt)
    
    # np.save('stats/' + opt.stats_fn + '.npy', net.stats.numpy())
    

def main():
    options = PlaygroundOptions()
    opt = options.parse()
    
    if opt.train:
        train(opt)
        
    if opt.print_class_sep:
        classSeparabilityEval(opt)
        
    if opt.visualize_hidden_reps:
        viewHiddenReps(opt)


# TODO: Actually evaluate linear separability of existing pipelines
#       Maybe write huge testing code??
# TODO: implement own self generating pipeline
# TODO: comment thoroughly
if __name__ == "__main__":
    main()