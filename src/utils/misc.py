import os
import oyaml as yaml
from pprint import pprint
import numpy as np
import random as random
import torch

USE_GPU = torch.cuda.is_available()


def to_cuda(xs):
    if type(xs) is not list and type(xs) is not tuple:
        return xs.cuda() if USE_GPU else xs
    items = list()
    for curr_item in xs:
        curr_item = curr_item.cuda() if USE_GPU else curr_item
        items.append(curr_item)

    return items


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def average_values_in_list_of_dicts(list_of_dicts):
    averaged_values_dict = dict()
    for curr_output_dict in list_of_dicts:
        for k, v in curr_output_dict.items():
            if k not in averaged_values_dict:
                averaged_values_dict[k] = [v]
            else:
                averaged_values_dict[k].append(v)
    for k, v in averaged_values_dict.items():
        averaged_values_dict[k] = np.array(v).mean()

    return averaged_values_dict


def concatenate_values_in_list_of_dicts(list_of_dicts):
    concat_values_dict = dict()
    for k, v in list_of_dicts[0].items():
        curr_metrics_list = list()
        try:
            for curr_output_dict in list_of_dicts:
                curr_metrics_list += list(curr_output_dict[k])
            concat_values_dict[k] = np.array(curr_metrics_list)
        except Exception as e:
            continue

    return concat_values_dict


def prepend_string_to_dict_keys(prepend_key, dictinary):
    return {"{}{}".format(prepend_key, k): v for k, v in dictinary.items()}


def print_experiment_config(path="."):
    yaml_path = os.path.join(path, "template.yaml")
    config_dict = yaml.safe_load(open(yaml_path))
    pprint(config_dict)


def sendline_and_get_response(s, line):
    s.sendline(line)
    s.prompt()
    reply = str(s.before.decode("utf-8"))
    pprint(reply)


def set_hyperparams(path, wandb_logger):
    with open(path, 'r') as f:
        x = yaml.safe_load(f)
        wandb_logger.log_hyperparams(x)


def timeit(method):
    import time

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f"Time taken: {(te - ts) * 1000}")
        return result

    return timed
