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
    for k, v in list_of_dicts[0].items():
        curr_metric_list = list()
        try:
            for curr_output_dict in list_of_dicts:
                curr_metric_list.append(float(curr_output_dict[k]))
            averaged_values_dict[k] = np.array(curr_metric_list).mean()
        except Exception as e:
            continue

    return averaged_values_dict


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