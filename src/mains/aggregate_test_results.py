import os
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='/scratch/hdd001/home/jennybao/projects/LTE/results')
parser.add_argument('--task', default='mimic')
parser.add_argument('--distribution', default='real_data')
parser.add_argument('--model', default='all')
parser.add_argument('--voting_rule', default='all')
parser.add_argument('--dataset_name_filter', default='')
parser.add_argument('--result_filename', default='test_result')
args = parser.parse_args()

exp_dir = f"{args.root_dir}/{args.task}/{args.distribution}"

if args.model == 'all':
    model_list = [model_type for model_type in os.listdir(exp_dir)]
else:
    model_list = [args.model]

for model_type in model_list:
    model_dir = f"{exp_dir}/{model_type}"

    if args.voting_rule == 'all':
        voting_rule_list = [v for v in os.listdir(model_dir)]
    else:
        voting_rule_list = [args.voting_rule]

    for voting_rule in voting_rule_list:
        vote_dir = f"{model_dir}/{voting_rule}"

        weight_sum_acc = 0.0
        total_samples = 0.0
        for dataset_name in os.listdir(vote_dir):
            data_dir = f"{vote_dir}/{dataset_name}"
            if not os.path.isdir(data_dir) or not args.dataset_name_filter in data_dir:
                continue

            with open(f"{data_dir}/{args.result_filename}.yaml", 'r') as yaml_f:
                results_list = yaml.safe_load(yaml_f)
                overall_dict = results_list[1]['overall']
                weight_sum_acc += overall_dict['test/model/acc'] * overall_dict['total_test_sample_size']
                total_samples += overall_dict['total_test_sample_size']

        if total_samples > 0.0:
            agg_acc = {
                'test/model/acc': weight_sum_acc / total_samples,
                'total_test_sample_size': total_samples
            }
            with open(f"{vote_dir}/{args.result_filename}{args.dataset_name_filter}.yaml", 'w') as yaml_f:
                yaml.dump(agg_acc, yaml_f, default_flow_style=False)
                print(f"Saved to {vote_dir}/{args.result_filename}{args.dataset_name_filter}")
        else:
            print(f"No sub-level results found in {vote_dir}, skipped.")

