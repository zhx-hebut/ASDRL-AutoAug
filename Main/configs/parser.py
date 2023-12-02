import yaml
import argparse


class BaseOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.config = None
        self.opts = self.initialize()

    def parse_config(self, d, opts=None):
        if opts is None:
            opts = type('new', (object,), d)
        seqs = tuple, list, set, frozenset
        for i, j in d.items():
            if isinstance(j, dict): 
                setattr(opts, i, self.parse_config(j))
            elif isinstance(j, seqs):
                setattr(opts, i, type(j)(self.parse_config(sj) if isinstance(sj, dict) else sj for sj in j))
            else:
                setattr(opts, i, j)
        return opts

    def initialize(self):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser = argparse.ArgumentParser('DeepAugNet in Kidney')
        parser.add_argument(
            "--cfg",
            default='ASDRL-AutoAug-public/Main/configs/kidney_2d.yaml',
            metavar="FILE", 
            help="path to config file",
            type=str,
        )
        opts = parser.parse_args()
        self.cfg = get_config(opts.cfg)
        opts = self.parse_config(self.cfg, opts)
        return opts


def get_config(config_yaml):
    with open(config_yaml, 'r') as stream:
        config = yaml.safe_load(stream)
        return config