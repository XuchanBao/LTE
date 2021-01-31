import torch
import datetime
import os
from shutil import copyfile

import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from spaghettini import quick_register


@quick_register
def load_numpy_array():
    def load_npy(path):
        return np.load(path)

    return load_npy


@quick_register
def manage_checkpoint(root_path, experiment_name, load_version=None):
    def get_full_path_callback(model_type, voting_rule, template_path):
        exp_path = f"{root_path}/{experiment_name}/{model_type}/{voting_rule}"
        if load_version is None:
            load_path = None
        else:
            load_path= f"{exp_path}/{load_version}"

        # new checkpoint directory
        time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        new_ckpt_dir = f"{exp_path}/{time_str}"
        os.makedirs(new_ckpt_dir, exist_ok=True)
        print(f"checkpoint directory set to {os.path.abspath(new_ckpt_dir)}")

        # copy the template file to the new checkpoint directory
        copyfile(template_path, f"{new_ckpt_dir}/template.yaml")

        return CustomCheckpointCallback(load_path=load_path, filepath=new_ckpt_dir)
    return get_full_path_callback


class CustomCheckpointCallback(ModelCheckpoint):
    def __init__(self, load_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_path = load_path

    def on_fit_start(self, trainer, pl_module):
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
