import logging
import os
import time
from logging import Logger

import accelerate
import wandb

from accelerate import Accelerator
import tqdm


def create_logger(log_file=None, rank=0, log_level=logging.INFO) -> logging.Logger:
    """
        Create a logger.
        args:
            log_file: log file path.
            rank: rank of the process.
            log_level: log level.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def create_url_shortcut_of_wandb(wandb: wandb.sdk.wandb_run.Run = None,
                                 accelerator: accelerate.Accelerator = None) -> None:
    """
        Create a url shortcut of wandb.
        args:
            wandb: wandb object.
    """
    if wandb is None:
        try:
            wandb = accelerator.get_tracker("wandb", unwrap=True)
            url = wandb.get_url()
            run_dir = wandb.dir
            run_name = wandb.name
            shortcut_file = os.path.join(run_dir, run_name + ".url")
            with open(shortcut_file, "w") as f:
                f.write("[InternetShortcut]\n")
                f.write(f"URL={url}")
        except Exception as e:
            # print("wandb is not initialized. No url shortcut is created.")
            return


def create_logger_of_wandb(wandb: wandb.sdk.wandb_run.Run = None, accelerator: accelerate.Accelerator = None,
                           **kwargs) -> Logger:
    """
        Create a logger of wandb.
        args:
            wandb: wandb object.
            rank: rank of the process.
            log_level: log level.
    """
    if wandb is None:
        try:
            wandb = accelerator.get_tracker("wandb", unwrap=True)
            run_dir = wandb.dir
            log_file = os.path.join(run_dir, "log.txt")
        except Exception as e:
            # print("wandb is not initialized. No log file is created.")
            log_file = None
    logger = create_logger(log_file=log_file, **kwargs)
    return logger


if __name__ == '__main__':
    accelerator = Accelerator(log_with='wandb')
    accelerator.init_trackers("my_tracker", config={"a": 1, "b": 2})
    create_url_shortcut_of_wandb(accelerator=accelerator)
    logger = create_logger_of_wandb(accelerator=accelerator, rank=not accelerator.is_main_process)
    with tqdm.tqdm(initial=0, total=200, disable=not accelerator.is_main_process) as pbar:
        for i in range(200):
            accelerator.log({"loss": i}, step=i)
            accelerator.wait_for_everyone()
            time.sleep(0.01)
            if i % 10 == 0:
                logger.info(msg=f"loss hhh: {i}")
            pbar.set_description(f'loss: 12454)')
            pbar.update(1)

    accelerator.end_training()
