import torch
import datetime
import os
import yaml
from shutil import copyfile

import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from spaghettini import quick_register

DATETIME_STR_FORMAT = "%Y-%m-%d-%H%M%S"


def save_result_to_yaml(np_list, save_dir, filename, ckpt_path):
    yml_list = [{"ckpt_path": os.path.abspath(ckpt_path)}]
    inv_dist_ratio_dict = dict()

    assert len(np_list) == 1, f"Should test for only 1 epoch. Found len(np_list) == {len(np_list)}."
    np_dict = np_list[0]

    yml_dict = dict()
    for k, v in np_dict.items():
        if "group" in k:
            group_name = k.split('/')[-2]
        else:
            group_name = 'overall'

        if group_name not in yml_dict:
            yml_dict[group_name] = {}

        if np.size(v) == 1:
            yml_dict[group_name][k] = float(v)
        else:   # inv distortion ratios
            inv_dist_ratio_dict[k] = v
    yml_list.append(yml_dict)

    with open(f'{save_dir}/{filename}.yaml', 'w') as yaml_file:
        yaml.dump(yml_list, yaml_file, default_flow_style=False)
    print(f"Metric data saved to {save_dir}/{filename}.yaml")

    np.savez(f'{save_dir}/{filename}.npz', **inv_dist_ratio_dict)
    print(f"Inverse distortion ratio data saved to {save_dir}/{filename}.npz")


@quick_register
def load_numpy_array():
    def load_npy(path):
        return np.load(path)

    return load_npy


@quick_register
def manage_save_test_results(save_test_results_root, experiment_name, include_dataset_filename=False, model_suffix=""):
    def get_test_results_save_dir(model_type, voting_rule, template_path, utility_distribution, dataset_filename=None):
        if include_dataset_filename:
            save_dir = f"{save_test_results_root}/{experiment_name}/{utility_distribution}/" \
                       f"{model_type}{model_suffix}/{voting_rule}/{dataset_filename}"
        else:
            save_dir = f"{save_test_results_root}/{experiment_name}/{utility_distribution}/" \
                       f"{model_type}{model_suffix}/{voting_rule}"
        os.makedirs(save_dir, exist_ok=True)
        print(f">>> Saving test results to {os.path.abspath(save_dir)}.")

        # copy the template file to the saving directory
        copyfile(template_path, f"{save_dir}/template.yaml")

        return save_dir
    return get_test_results_save_dir


@quick_register
def manage_checkpoint(root_path, experiment_name, log_utility_distribution=False,
                      load_version=None, abs_load_path=False,
                      save_checkpoint=True, ckpt_frequency=500, additional_path_name=None, model_suffix=""):
    def get_full_path_callback(model_type, voting_rule, template_path, utility_distribution):
        if log_utility_distribution:    # for social welfare experiments
            exp_path = f"{root_path}/{experiment_name}/{utility_distribution}/{model_type}{model_suffix}/{voting_rule}"
        else:                           # for mimicking experiments
            exp_path = f"{root_path}/{experiment_name}/{model_type}{model_suffix}/{voting_rule}"

        if additional_path_name is not None:
            exp_path = f"{exp_path}/{additional_path_name}"

        if load_version is None:
            load_path = None
        else:
            if abs_load_path:
                load_path = load_version
            else:
                if load_version == "only":
                    version_list = [path for path in os.listdir(exp_path)]
                    assert len(version_list) == 1, \
                        f"When load_version == 'only', there must be exactly 1 version under the experiment dir. " \
                        f"Found {len(version_list)} directories under {os.path.abspath(exp_path)}."
                    load_path = f"{exp_path}/{version_list[0]}"

                elif load_version == "auto_resume":  # used for continue training after preemption

                    # load the latest checkpoint if exists, otherwise skip
                    if os.path.isdir(exp_path):
                        try:
                            all_versions_datetime = []
                            for date_str in os.listdir(exp_path):
                                all_versions_datetime.append(datetime.datetime.strptime(date_str, DATETIME_STR_FORMAT))
                        except:
                            raise ValueError(f"Failed converting directories to datetime objects. "
                                             f"Directories = {os.listdir(exp_path)}")
                        if len(all_versions_datetime) > 0:
                            latest_time = max(all_versions_datetime)
                            latest_dirname = latest_time.strftime(DATETIME_STR_FORMAT)
                            load_path = f"{exp_path}/{latest_dirname}"
                        else:
                            load_path = None
                    else:
                        load_path = None

                else:   # load_version is a directory
                    version_list = [path for path in os.listdir(exp_path) if load_version in path]
                    assert len(version_list) == 1, \
                        f"There must be exactly 1 version under the experiment dir that contains '{load_version}'. " \
                        f"Found {len(version_list)} directories under {os.path.abspath(exp_path)}."
                    load_path = f"{exp_path}/{version_list[0]}"

            if load_path is not None and os.path.isdir(load_path):
                ckpt_list = [file for file in os.listdir(load_path) if file.endswith('.ckpt')]
                assert len(ckpt_list) == 1, \
                    f"There should only be exactly 1 *.ckpt file in the load directory. " \
                    f"Found {len(ckpt_list)} in {os.path.abspath(load_path)}."
                load_path = os.path.join(load_path, ckpt_list[0])

        if save_checkpoint:
            # new checkpoint directory
            time_str = datetime.datetime.now().strftime(DATETIME_STR_FORMAT)
            new_ckpt_dir = f"{exp_path}/{time_str}"
            os.makedirs(new_ckpt_dir, exist_ok=True)
            print(f"checkpoint directory set to {os.path.abspath(new_ckpt_dir)}")

            # copy the template file to the new checkpoint directory
            copyfile(template_path, f"{new_ckpt_dir}/template.yaml")
        else:
            new_ckpt_dir = None

        return CustomCheckpointCallback(load_path=load_path, dirpath=new_ckpt_dir, period=ckpt_frequency)
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
            scheduler['scheduler'].load_state_dict(lrs_state)

        pl_module.load_state_dict(ckpt['state_dict'])

    def on_fit_start(self, trainer, pl_module):
        # should be called by both trainer.fit() and trainer.test()
        self._load_from_checkpoint(trainer, pl_module)
