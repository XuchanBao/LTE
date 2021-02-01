import torch
import datetime
import os
import yaml
from shutil import copyfile

import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from spaghettini import quick_register


def save_result_to_yaml(np_list, save_dir, filename):
    yml_list = []
    for np_dict in np_list:
        yml_dict = dict()
        for k, v in np_dict.items():
            if "group" in k:
                group_name = k.split('/')[-2]
            else:
                group_name = 'overall'

            if group_name not in yml_dict:
                yml_dict[group_name] = {}

            yml_dict[group_name][k] = float(v)
        yml_list.append(yml_dict)

    with open(f'{save_dir}/{filename}.yaml', 'w') as yaml_file:
        yaml.dump(yml_list, yaml_file, default_flow_style=False)
    print(f"Saved to {save_dir}/{filename}.yaml")


@quick_register
def load_numpy_array():
    def load_npy(path):
        return np.load(path)

    return load_npy


@quick_register
def manage_checkpoint(root_path, experiment_name, log_utility_distribution=False,
                      load_version=None, abs_load_path=False,
                      save_checkpoint=True):
    def get_full_path_callback(model_type, voting_rule, template_path, utility_distribution):
        if log_utility_distribution:    # for social welfare experiments
            exp_path = f"{root_path}/{experiment_name}/{utility_distribution}/{model_type}/{voting_rule}"
        else:                           # for mimicking experiments
            exp_path = f"{root_path}/{experiment_name}/{model_type}/{voting_rule}"

        if load_version is None:
            load_path = None
        else:
            if abs_load_path:
                load_path = load_version
            else:
                load_path= f"{exp_path}/{load_version}"

        if save_checkpoint:
            # new checkpoint directory
            time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
            new_ckpt_dir = f"{exp_path}/{time_str}"
            os.makedirs(new_ckpt_dir, exist_ok=True)
            print(f"checkpoint directory set to {os.path.abspath(new_ckpt_dir)}")

            # copy the template file to the new checkpoint directory
            copyfile(template_path, f"{new_ckpt_dir}/template.yaml")
        else:
            new_ckpt_dir = None

        return CustomCheckpointCallback(load_path=load_path, dirpath=new_ckpt_dir)
    return get_full_path_callback


class CustomCheckpointCallback(ModelCheckpoint):
    def __init__(self, load_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_path = load_path

    def _load_from_checkpoint(self, trainer, pl_module):
        if self.load_path is None:
            return

        print(f"<<< loading checkpoint from {self.load_path}")
        ckpt = torch.load(self.load_path)
        trainer.global_step = ckpt['global_step']
        trainer.current_epoch = ckpt['epoch']

        # restore the optimizers
        optimizer_states = ckpt['optimizer_states']
        for optimizer, opt_state in zip(trainer.optimizers, optimizer_states):
            optimizer.load_state_dict(opt_state)

        # restore the lr schedulers
        lr_schedulers = ckpt['lr_schedulers']
        for scheduler, lrs_state in zip(trainer.lr_schedulers, lr_schedulers):
            scheduler.load_state_dict(lrs_state)

        pl_module.load_state_dict(ckpt['state_dict'])

    def on_fit_start(self, trainer, pl_module):
        # should be called by both trainer.fit() and trainer.test()
        self._load_from_checkpoint(trainer, pl_module)
