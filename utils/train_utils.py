import glob
import logging
import os
from typing import Dict, Any, List
import torch
import argparse
import random
import numpy as np
from collections import deque


def load_state_dict(model: torch.nn.Module, state_dict: Dict[str, Any]) -> torch.nn.Module:
    """
        Load state dict to model.
        args:
            model: model to load state dict
            state_dict: state dict to load
    """
    # make sure all keys are in state_dict
    model_state_dict = model.state_dict()
    missing_keys = set(model_state_dict.keys()) - set(state_dict.keys())
    unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())
    error_msg = []
    if len(missing_keys) > 0:
        error_msg.append(f"Missing key(s) in state_dict: {missing_keys}.")
    if len(unexpected_keys) > 0:
        error_msg.append(f"Unexpected key(s) in state_dict: {unexpected_keys}.")
    assert len(error_msg) == 0, " ".join(error_msg)
    model.load_state_dict(state_dict, strict=True)
    return model

def checkpoint_state(model=None, optimizer=None, scheduler=None, epoch=None, it=None) -> Dict[str, Any]:
    """
        Return a checkpoint state dict.
    """
    optim_state = optimizer.state_dict() if optimizer is not None else None
    sched_state = scheduler.state_dict() if scheduler is not None else None
    if model is not None:
        model_state = model.state_dict()
    else:
        model_state = None
    return {
        "epoch": epoch,
        "it": it,
        "model_state": model_state,
        "optimizer_state": optim_state,
        "scheduler_state": sched_state
    }


def save_checkpoint(state: Dict[str, Any], epoch: int or str, save_path="./checkpoint", max_checkpoints=0) -> None:
    """
        Save checkpoint to disk, and remove old checkpoints if needed.
        The checkpoint file name is "checkpoint_epoch_{epoch}.pth" in the checkpoint directory.
        args:
            state: checkpoint state dict
            epoch: current epoch
            save_path: checkpoint directory
            max_checkpoints: max number of checkpoints to keep. If 0, keep all checkpoints.
    """
    filepath = os.path.join(save_path, f"checkpoint_epoch_{epoch}.pth")
    torch.save(state, filepath)
    checkpoint_files = glob.glob(os.path.join(save_path, "checkpoint_epoch_*.pth"))
    checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]))

    # remove old checkpoints if number of checkpoints exceed max_checkpoints
    if len(checkpoint_files) > max_checkpoints and max_checkpoints > 0:
        for f in checkpoint_files[:-max_checkpoints]:
            os.remove(f)


def load_checkpoint(model=None, optimizer=None, scheduler=None, ckpt_dir="./checkpoint", logger: logging.Logger = None):
    """
        Load checkpoint from disk.
        args:
            model: model to load checkpoint
            optimizer: optimizer to load checkpoint
            ckpt_dir: checkpoint directory or checkpoint file. If it is a directory, load the latest checkpoint, otherwise load the checkpoint file.
            logger: logger
        return:
            start_epoch: start epoch, 0 if specified checkpoint file.
            start_it: start iteration, 0 if specified checkpoint file.
    """
    # if logger is None, then set logger.info to print
    if logger is None:
        logger = argparse.Namespace()
        logger.info = print

    # if specified the ckpt file
    if os.path.isfile(ckpt_dir):
        logger.info("Loading checkpoint from %s", ckpt_dir)
        state_dict = torch.load(ckpt_dir, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict["model_state"])
        return 0, 0

    # or specified the ckpt dir
    checkpoint_files = glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pth"))
    if len(checkpoint_files) == 0:
        raise FileNotFoundError("No checkpoint found in %s" % ckpt_dir)
    checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]))
    last_ckpt_file = checkpoint_files[-1]
    logger.info("Loading checkpoint from %s", last_ckpt_file)
    state_dict = torch.load(last_ckpt_file, map_location=torch.device("cpu"))
    cur_epoch, cur_it = state_dict["epoch"] + 1, state_dict["it"]  # +1 because we want to start from next epoch
    model.load_state_dict(state_dict["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(state_dict["scheduler_state"])
    return cur_epoch, cur_it


def freeze_params_contain_keyword(model, keywords: List[str], logger: logging.Logger = None):
    """
        Freeze parameters that contain keywords.
    """
    # if logger is None, then set logger.info to print
    if logger is None:
        logger = argparse.Namespace()
        logger.info = print

    if keywords is None or len(keywords) == 0:
        return

    logger.info("Freezing params containing keywords: %s", keywords)
    for name, param in model.named_parameters():
        for keyword in keywords:
            if keyword in name:
                param.requires_grad = False
                logger.info("Freeze parameter %s", name)


def set_random_seed(seed=0, determin=False, benchmark=False):
    """
        set random seed.
        if seed is 0, then force torch conv to use a deterministic algorithm.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # if seed == 0:
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    if determin:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def reset(self):
        self.deque.clear()
        self.count = 0
        self.total = 0.0

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)
