import os
import argparse

import torch

from spaghettini import load
from src.utils.seed import set_seed
from src.utils.misc import set_hyperparams

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


if __name__ == "__main__":
    # Get the config.
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str,
                        help="The path to the configuration file")
    args = parser.parse_args()

    # Load the config.
    cfg = load(args.cfg, record_config=False)

    # Set the seed.
    set_seed(cfg.seed)

    # Get directory where results will be saved.
    save_dir = os.path.split(args.cfg)[0]

    # Get the system, experiment (for logging) and trainer.
    system = cfg.system
    logger = cfg.logger(save_dir=save_dir)

    set_hyperparams(args.cfg, logger)

    ckpt_callback = cfg.manage_checkpoint(
        model_type=system.model.name,
        voting_rule=system.train_loader.dataset.voting_rule.__name__,
        template_path=args.cfg,
        utility_distribution=system.train_loader.dataset.utility_distribution
    )

    if torch.cuda.is_available():
        trainer = cfg.trainer(logger=logger, gpus=1, callbacks=[ckpt_callback])
    else:
        trainer = cfg.trainer(logger=logger, callbacks=[ckpt_callback])

    # Train.
    trainer.fit(system)
