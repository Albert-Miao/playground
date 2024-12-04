import argparse

class PlaygroundOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='options')
        
        self.parser.add_argument("--dataset",
                                 default="MNIST",
                                 help="supports CIFAR10 and MNIST currently")
        self.parser.add_argument("--batch_size",
                                 default=32)
        
        self.parser.add_argument("--model_type",
                                 default="control",
                                 help="""
                                         Currently testing control, simpleCluster, 
                                         explodingCluster, expandingCluster, shiftingCluster
                                      """)
        self.parser.add_argument("--batch_norm",
                                 default=True)
        self.parser.add_argument("--hidden_rep_dim",
                                 default=30)
        self.parser.add_argument("--cl_alpha",
                                 default=10,
                                 help='How much to value clustering over control loss')
        self.parser.add_argument("--cl_beta",
                                 default=0.001,
                                 default='How much to value cluster over explosion (explodingCluster)')
        self.parser.add_argument("--initial_cl_rate",
                                 default=0,
                                 help='Initial proportion of cl to l')
        self.parser.add_argument("--cl_rate_speed",
                                 default=1/6,
                                 help='Amount to increase cl_rate by per mega_batch')
        self.parser.add_argument("--super_batch_size",
                                 default=250,
                                 help="Number of batches until centers are recalculated")
        
        
        
    def parse(self, args=None):
        self.options = self.parser.parse_args
        return self.options