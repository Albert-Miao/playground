import argparse

class PlaygroundOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='options')
        
        self.parser.add_argument("--train",
                                 default=True)
        self.parser.add_argument("--print_class-sep",
                                 default=False)
        self.parser.add_argument("--visualize_hidden_reps",
                                 default=False)
        
        self.parser.add_argument("--gpu",
                                 default=0,
                                 type=int)
        self.parser.add_argument("--dataset",
                                 default="CIFAR10",
                                 help="supports CIFAR10 and MNIST currently")
        self.parser.add_argument("--batch_size",
                                 default=32,
                                 type=int)
        self.parser.add_argument("--lr",
                                 default=0.004,
                                 type=float)
        self.parser.add_argument("--momentum",
                                 default=0.9,
                                 type=float)
        self.parser.add_argument("--num_epochs",
                                 default=156) #3
        
        # TODO: Add more helpful descriptions of the various clustering algorithms
        self.parser.add_argument("--model_type",
                                 default="feature",
                                 help="""
                                         Currently testing control, simpleCluster, 
                                         explodingCluster, expandingCluster, shiftingCluster, classCluster, feature
                                      """)
        self.parser.add_argument("--batch_norm",
                                 default=False)
        self.parser.add_argument("--hidden_rep_dim",
                                 default=50,
                                 type=int)
        self.parser.add_argument("--cl_alpha",
                                 default=1,
                                 type=int, #10
                                 help='How much to value clustering over control loss')
        self.parser.add_argument("--cl_beta",
                                 default=0.2,
                                 type=float,#0.001
                                 help='How much to value cluster over explosion (explodingCluster)')
        self.parser.add_argument("--initial_cl_rate",
                                 default=0,
                                 type=float,
                                 help='Initial proportion of cl to l')
        self.parser.add_argument("--cl_rate_speed",
                                 default=1/6,
                                 type=float,
                                 help='Amount to increase cl_rate by per mega_batch')
        self.parser.add_argument("--super_batch_size",
                                 default=250,
                                 type=int,
                                 help="Number of batches until centers are recalculated")
        self.parser.add_argument("--num_clusters",
                                 default=80,
                                 type=int,
                                 help='Number of clusters for kmeans clustering methods')
        
        self.parser.add_argument("--stats_fn",
                                 default='0',
                                 help='Filepath to save stats')
        
        self.parser.add_argument("--debug_load_pth",
                                 default=False)
        
        
        
    def parse(self, args=None):
        self.options = self.parser.parse_args(args)
        return self.options