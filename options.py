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
        self.parser.add_argument("--return_hidden_rep_info",
                                 default=True)
        
    def parse(self, args=None):
        self.options = self.parser.parse_args
        return self.options