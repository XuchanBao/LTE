import os
import argparse

import torch

from spaghettini import load
from src.utils.seed import set_seed


if __name__ == "__main__":
    # Get the config.
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str,
                        help="The path to the configuration file")
    args = parser.parse_args()

    # Load the config.
    cfg = load(args.cfg)

    # Set the seed.
    set_seed(cfg.seed)

    # Get directory where results will be saved.
    save_dir = os.path.split(args.cfg)[0]

    # Get the system, experiment (for logging) and trainer.
    system = cfg.system
    logger = cfg.logger(save_dir=save_dir)
    if torch.cuda.is_available():
        trainer = cfg.trainer(logger=logger, gpus=1, default_save_path=save_dir)
    else:
        trainer = cfg.trainer(logger=logger)

    # Train.
    trainer.fit(system)
