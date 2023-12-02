import itertools
import warnings

from yacs.config import CfgNode
from configs.parser import load_cfg
from engine.trainer import Trainer


class Task(object):
    """
    An object to provide cfg and target values,
    so that user can create a task easier
    """
    def __init__(self, cfg_name):
        self.cfg_name = cfg_name
        self.target_values = {}

    def add_values(self, key, value):
        if not isinstance(value, (list, tuple)):
            warnings.warn('value should be list, put it into list automatically')
            value = [value]
        self.target_values[key] = value


def change_value(cfg:CfgNode, key:str, value):
    """ Change cfg value with key and value """
    parent_key, child_key = key.split('.')
    cfg[parent_key][child_key] = value
    return cfg


def create_cfgs(task:Task):
    """ Create cfg according to task"""
    origin_cfg = load_cfg(task.cfg_name)

    keys = []
    values = []
    for k, v in task.target_values.items():
        keys.append(k)
        values.append(v)

    # get combinations of all params
    param_combinations = itertools.product(*values)

    # overwrite params in provided cfg
    cfgs = []
    param_strings = []
    for params in param_combinations:
        param_string = ''
        new_cfg = origin_cfg.clone()
        for k, v in zip(keys, params):
            print('Key:{}, value:{}'.format(k, v))
            new_cfg = change_value(new_cfg, k, v)
            param_string += k + '_' + str(v) + '_'
        cfgs.append(new_cfg)
        param_strings.append(param_string)

    # if no cfg generated, use default cfg
    if not cfgs:
        cfgs.append(origin_cfg)
        param_strings.append('Default')
    return cfgs, param_strings


def do_one_task(task:Task):
    """ Train networks with cfgs generated from task"""
    cfgs, param_strs = create_cfgs(task)
    for cfg, p_str in zip(cfgs, param_strs):
        print('Trying... {}'.format(p_str))
        trainer = Trainer(cfg)
        trainer.train()


def do_tasks(tasks:[Task]):
    """ Do specified task sequentially. """
    if isinstance(tasks, list):
        for task in tasks:
            do_one_task(task)
    else:
        do_one_task(tasks)

