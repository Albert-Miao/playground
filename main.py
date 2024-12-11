import torch
import numpy as np

from options import PlaygroundOptions
from datasets import generate_data_loaders
from models import ClusterNet, NaiveNet
from model_pipelines import trainNet, evalNet
from visualizations import viewHiddenReps, classSeparabilityEval

def train(opt):
    torch.cuda.set_device(opt.gpu)
    print(torch.cuda.get_device_name(0))
    net = None
    if opt.model_type == "control":
        net = NaiveNet(opt).cuda()
    else:
        net = ClusterNet(opt).cuda()
        
    trainloader, testloader = generate_data_loaders(opt)
    trainNet(trainloader, net, opt)
    evalNet(trainloader, testloader, net, opt)
    
    np.save('stats/' + opt.stats_fn + '.npy', net.stats.numpy())
    

def main():
    options = PlaygroundOptions()
    opt = options.parse()
    
    if opt.train:
        train(opt)
        
    if opt.print_class_sep:
        classSeparabilityEval(opt)
        
    if opt.visualize_hidden_reps:
        viewHiddenReps(opt)


# TODO: write code to evaluate the linear seperability of images
# TODO: implement own self generating pipeline
# TODO: comment thoroughly
if __name__ == "__main__":
    main()