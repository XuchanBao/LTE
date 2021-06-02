import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

baseline_list = ['plurality', 'borda', 'copeland', 'maximin']

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='/scratch/hdd001/home/jennybao/projects/LTE/results')
parser.add_argument('--task', default='social-welfare')
parser.add_argument('--distribution')
parser.add_argument('--model', default='all')
parser.add_argument('--voting_rule')
parser.add_argument('--additional_path_name', default='neurips/v1')
parser.add_argument('--result_filename', default='test_result')
args = parser.parse_args()

exp_dir = f"{args.root_dir}/{args.task}/{args.distribution}"
if args.model == 'all':
    model_list = [model_type for model_type in os.listdir(exp_dir)]
else:
    model_list = [args.model]

voting_rule = args.voting_rule
sns.set_palette("muted")
sns.set_theme()
fig, ax = plt.subplots(figsize=(6, 4))

baseline_data = {b: [] for b in baseline_list}

for model_type in model_list:
    model_dir = f"{exp_dir}/{model_type}"
    vote_dir = f"{model_dir}/{voting_rule}"
    if args.additional_path_name is not None:
        vote_dir = f"{vote_dir}/{args.additional_path_name}"

    if not os.path.exists(f"{vote_dir}/{args.result_filename}.npz"):
        print(f"Skipping {vote_dir}/{args.result_filename}.npz as it doesn't exist.")
        continue
    test_result = np.load(f"{vote_dir}/{args.result_filename}.npz")

    inv_dist_ratios = test_result['test/model/inv_dist_ratios']
    for b in baseline_list:
        baseline_data[b].extend(test_result[f'test/{b}/inv_dist_ratios'])

    count, bins = np.histogram(inv_dist_ratios, bins=200, density=True)

    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    label = "MLP" if model_type == "FullyConnectedSmall" else f"{model_type}"
    plt.plot(bin_centers, count, label=label)

for b in baseline_list:
    count, bins = np.histogram(baseline_data[b], bins=200, density=True)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    plt.plot(bin_centers, count, label=f"{b}", alpha=0.6)

plt.yscale('log')
plt.legend()
plt.xlabel('$\\frac{sw(\\hat{a},\\vec{u})}{max_a sw(a, \\vec{u})}$', fontsize=16)
plt.ylabel('Density', fontsize=14)
plt.tight_layout()
plt.savefig(f"{exp_dir}/{args.distribution}-{voting_rule}-histogram.pdf")
