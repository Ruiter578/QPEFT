from .lora_cfgs import *
from .quant_cfgs import *
from .train_cfgs import *

import importlib
from easydict import EasyDict as edict
import sys


def parse_configs(name):
    module = importlib.import_module(name)
    cfgs = getattr(module, 'cfg')
    # import pdb; pdb.set_trace()
    return cfgs

def merge_configs(cfgs_list):
    configs = edict()

    for c in cfgs_list:
        for k, v in vars(c).items():
            assert not k in configs.keys(), f"{k} is repeated in configs!"
            # configs.update({k: v})
            # import pdb; pdb.set_trace()
            setattr(configs, k, v)
    return configs

def load_configs(args):
    cfgs_list = []
    if hasattr(args, 'train_cfgs'):
        if not args.train_cfgs is None:
            sys.path.append('./configs/train_cfgs')
            cfgs_list.append(parse_configs(args.train_cfgs))
            sys.path.pop()

    if hasattr(args, 'quant_cfgs'):
        if not args.quant_cfgs is None:
            sys.path.append('./configs/quant_cfgs')
            cfgs_list.append(parse_configs(args.quant_cfgs))
            sys.path.pop()
    if hasattr(args, "lora_cfgs"):
        if not args.lora_cfgs is None:
            sys.path.append('./configs/lora_cfgs')
            cfgs_list.append(parse_configs(args.lora_cfgs))
            sys.path.pop()
    if len(cfgs_list) == 0:
        raise ValueError
    return merge_configs(cfgs_list)