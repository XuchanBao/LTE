import os
import yaml
import argparse

model_order_list = ['SetTransformer', 'GIN', 'DeepSetOriginal', 'FullyConnectedSmall']
generalization_order_list = ['_49', '50_99', '100_149', '150_199']

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='/scratch/hdd001/home/jennybao/projects/LTE/results')
parser.add_argument('--task', default='mimic_generalization')
parser.add_argument('--distribution', default='uniform')
parser.add_argument('--model', default='all')
parser.add_argument('--voting_rule')
parser.add_argument('--additional_path_name', default='neurips/v1')
parser.add_argument('--result_filename', default='test_result')
args = parser.parse_args()

task_dir = f"{args.root_dir}/{args.task}"

acc_table_dict = {}

for sub_task in os.listdir(task_dir):
    exp_dir = f"{task_dir}/{sub_task}/{args.distribution}"
    print(f"sub_task = {sub_task}")
    if args.model == 'all':
        model_list = [model_type for model_type in os.listdir(exp_dir)]
    else:
        model_list = [args.model]

    acc_table_dict[sub_task] = {}
    for model_type in model_list:
        model_dir = f"{exp_dir}/{model_type}"

        voting_rule = args.voting_rule

        vote_dir = f"{model_dir}/{voting_rule}"

        if args.additional_path_name is not None:
            vote_dir = f"{vote_dir}/{args.additional_path_name}"

            if not os.path.isdir(vote_dir):
                print(f"Skipping {vote_dir} as it doesn't exist.")
                continue

        assert os.path.isdir(vote_dir), f"Directory {vote_dir} does not exist!"
        if not os.path.exists(f"{vote_dir}/{args.result_filename}.yaml"):
            continue
        print(f"voting_dir = {vote_dir}")

        with open(f"{vote_dir}/{args.result_filename}.yaml", 'r') as yaml_f:
            results = yaml.safe_load(yaml_f)
            overall_dict = results[1]['overall']
            acc = overall_dict['test/model/acc']

        acc_table_dict[sub_task][model_type] = acc

print(f"\nacc_table_dict = {acc_table_dict}\n")
# print in tabular format
table_strs = []


for model_type in model_order_list:
    print(f"{model_type}")
    model_table_strs = [f"{model_type}"]

    for task_kwd in generalization_order_list:
        for sub_task in acc_table_dict.keys():
            if task_kwd in sub_task:

                if model_type not in acc_table_dict[sub_task].keys():
                    print(f"[Warning] Skipping model {model_type} and keyword {task_kwd}.")
                    continue

                print(f"\t{sub_task}")
                break
        if task_kwd in sub_task:
            model_table_strs.append("{:.3f}".format(acc_table_dict[sub_task][model_type]))
        else:
            model_table_strs.append("")

    table_strs.append(" & ".join(model_table_strs))

all_str = "\n".join(table_strs)
print(all_str)
