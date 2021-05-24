import os
import yaml
import argparse
import numpy as np

model_order_list = ['SetTransformer', 'GIN', 'DeepSetOriginal', 'FullyConnectedSmall']
voting_rule_order_list = ['plurality', 'borda', 'copeland', 'maximin', 'kemeny', 'utilitarian', 'egalitarian']
baseline_order_list = ['plurality', 'borda', 'copeland', 'maximin']

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='/scratch/hdd001/home/jennybao/projects/LTE/results')
parser.add_argument('--task', default='mimic')
parser.add_argument('--distribution')
parser.add_argument('--model', default='all')
parser.add_argument('--voting_rule', default='all')
parser.add_argument('--additional_path_name', default='neurips/v1')
parser.add_argument('--acc_source', default='model')
parser.add_argument('--result_filename', default='test_result')
parser.add_argument('--verbose_result', default=True)
args = parser.parse_args()

exp_dir = f"{args.root_dir}/{args.task}/{args.distribution}"

if args.model == 'all':
    model_list = [model_type for model_type in os.listdir(exp_dir)]
else:
    model_list = [args.model]

acc_table_dict = {}
for model_type in model_list:
    model_dir = f"{exp_dir}/{model_type}"
    acc_table_dict[model_type] = {}

    if args.voting_rule == 'all':
        voting_rule_list = [v for v in os.listdir(model_dir)]
    else:
        voting_rule_list = [args.voting_rule]

    for voting_rule in voting_rule_list:
        vote_dir = f"{model_dir}/{voting_rule}"

        if args.additional_path_name is not None:
            vote_dir = f"{vote_dir}/{args.additional_path_name}"

            if not os.path.isdir(vote_dir):
                print(f"Skipping {vote_dir} as it doesn't exist.")
                continue

        assert os.path.isdir(vote_dir), f"Directory {vote_dir} does not exist!"
        if not os.path.exists(f"{vote_dir}/{args.result_filename}.yaml"):
            continue

        with open(f"{vote_dir}/{args.result_filename}.yaml", 'r') as yaml_f:
            results = yaml.safe_load(yaml_f)
            if args.acc_source == 'model':
                if args.verbose_result is True:
                    overall_dict = results[1]['overall']
                    acc = overall_dict['test/model/acc']
                else:
                    acc = results['test/model/acc']
            else:
                assert args.acc_source == 'baselines', "acc_source needs to be either 'model' or 'baselines'."
                acc = {baseline_rule: results[1]['overall'][f'test/{baseline_rule}/acc']
                       for baseline_rule in baseline_order_list}

        acc_table_dict[model_type][voting_rule] = acc


if args.acc_source == 'baselines':
    baseline_acc_dict = {}
    for baseline_rule in baseline_order_list:
        baseline_acc_dict[baseline_rule] = {}
        for model_type, model_acc_dict in acc_table_dict.items():
            for voting_rule, accs in model_acc_dict.items():
                if voting_rule not in baseline_acc_dict[baseline_rule]:
                    baseline_acc_dict[baseline_rule][voting_rule] = []

                baseline_acc_dict[baseline_rule][voting_rule].append(accs[baseline_rule])

        for voting_rule in baseline_acc_dict[baseline_rule]:
            baseline_acc_dict[baseline_rule][voting_rule] = np.mean(baseline_acc_dict[baseline_rule][voting_rule])

    acc_table_dict = baseline_acc_dict
    model_order_list = baseline_order_list


# print in tabular format
table_strs = []
for model_type in model_order_list:
    if model_type not in acc_table_dict.keys():
        print(f"[Warning] Skipping model {model_type}")
        continue

    print(f"{model_type}")
    model_table_strs = [f"{model_type}"]

    for voting_rule in voting_rule_order_list:
        if voting_rule in acc_table_dict[model_type]:
            print(f"\t{voting_rule}")
            model_table_strs.append("{:.3f}".format(acc_table_dict[model_type][voting_rule]))
        else:
            model_table_strs.append("")

    table_strs.append(" & ".join(model_table_strs))

all_str = "\n".join(table_strs)
print(all_str)
