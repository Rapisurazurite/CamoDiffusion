import argparse
from random import random

import numpy as np
import omegaconf
import torch
from omegaconf import OmegaConf

def add_args(parser: argparse.ArgumentParser or argparse.Namespace) -> omegaconf.dictconfig.DictConfig:
    """
        Add arguments to the parser
        1. Read the config file and add the parameters to the parser
        2. Support using '__base__' to inherit the parameters from the base config file
        3. Override the parameters in the config file with the command line parameters
        4. Override the parameters in the config file with '--set' parameters, e.g. --set train.batch_size=4
    """
    if isinstance(parser, argparse.ArgumentParser):
        parser.add_argument('-c', '--config', type=str, help='config file path', default='configs/default.yaml')
        parser.add_argument('--set', nargs='+', type=str, help="override config file settings", default=[])
        args = parser.parse_args()
    elif isinstance(parser, argparse.Namespace):
        args = parser
    else:
        raise TypeError(f'parser must be argparse.ArgumentParser or argparse.Namespace, but got {type(parser)}')
    # read config file
    if args.config is not None:
        config = OmegaConf.load(args.config)
    else:
        config = OmegaConf.create()

    __base__ = config.get('__base__', [])
    config.__base__ = []
    # load config file and it's base config file
    while len(__base__) > 0:
        base_config = OmegaConf.load(__base__.pop(0))
        config = OmegaConf.merge(base_config, config)
        __base__ += base_config.get('__base__', [])
        config.__base__ = []
    # override config file settings
    for k, v in args.__dict__.items():
        cfg_v = config.get(k, None)
        config[k] = v if v is not None else cfg_v
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.set)) if len(args.set) > 0 else config
    return config


def config_pretty(d: omegaconf.dictconfig.DictConfig, indent=0):
    for key, value in d.items():
        print('\n' + '\t' * indent + str(key) + ":", end='')
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            config_pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value), end='')
