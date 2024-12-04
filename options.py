import argparse

class playgroundOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='options')
        
        self.parser.add_argument("--dataset",
                                 default="MNIST",
                                 help="supports CIFAR10 and MNIST currently")
        self.parser.add_argument("--batch_size",
                                 default=32)
        
    def parse(self, args=None):
        self.options = self.parser.parse_args
        return self.options