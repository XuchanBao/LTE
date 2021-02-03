import os
import torch
import argparse
from spaghettini import load
from src.utils.seed import set_seed
from src.utils.saving_loading import save_result_to_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str,
                        help="The path to the configuration file")
    args = parser.parse_args()

    # Load the config.
    cfg = load(args.cfg, record_config=False)

    # Set the seed.
    set_seed(cfg.seed)

    system = cfg.system
    test_loader = cfg.test_loader

    # Get directory where results will be saved.
    save_dir = os.path.split(args.cfg)[0]
    test_results_save_dir = cfg.manage_save_test_results(
        model_type=system.model.name,
        voting_rule=test_loader.dataset.voting_rule.__name__,
        template_path=args.cfg,
        utility_distribution=test_loader.dataset.utility_distribution
    )

    logger = cfg.logger(save_dir=save_dir)

    ckpt_callback = cfg.manage_checkpoint(
        model_type=system.model.name,
        voting_rule=test_loader.dataset.voting_rule.__name__,
        template_path=args.cfg,
        utility_distribution=test_loader.dataset.utility_distribution
    )

    if torch.cuda.is_available():
        trainer = cfg.trainer(logger=logger, gpus=1, callbacks=[ckpt_callback])
    else:
        trainer = cfg.trainer(logger=logger, callbacks=[ckpt_callback])

    # Test
    test_result = trainer.test(system, test_loader)

    save_result_to_yaml(test_result, test_results_save_dir, "test_result", ckpt_callback.load_path)
