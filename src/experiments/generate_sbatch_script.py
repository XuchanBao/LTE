import argparse
import os

CONDA_ENV = "lte"
ROOT = "/Users/cemanil/Workspace/LTE"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generating sbatch script for slurm. ")
    parser.add_argument("--dir", help="The path to the folder of interest")
    parser.add_argument(
        "--partition", "-p", help="Partition on slurm. ", default="p100"
    )
    parser.add_argument(
        "--cpu", "-c", help="# of cpus", default=4
    )
    parser.add_argument(
        "--mem", help="amount of memory", default="12G"
    )
    parser.add_argument(
        "--gres", help="gres", default="gpu:1"
    )
    parser.add_argument(
        "--limit", help="limit on number of jobs", default=36
    )
    parser.add_argument("--pattern",
                        help="Find all files that ends with this in the folder",
                        default="cfg.yaml")
    parser.add_argument("--conda_env",
                        help="Which conda environment to activate. ",
                        default=CONDA_ENV)
    parser.add_argument("--out",
                        help="output directory relative to dir",
                        default="batch_run.sh")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    lst = []
    print(
        "=========== Searching for {} ==================".format(args.pattern))
    for dir_name, subdir_list, file_list in os.walk(args.dir):
        for fname in file_list:
            if args.pattern == fname:
                fn = os.path.join(dir_name, fname)
                print('  %s' % (fn))
                lst.append("\"python -um src.mains.train --cfg " +
                           fn +
                           (" --resume" if args.resume else "") +
                           (" --test" if args.test else "") +
                           "\"")
    print("Found {} experiments".format(len(lst)))

    print("Generating sbatch script....")
    out_dir = os.path.join(args.dir, args.out)
    print(out_dir)
    with open(out_dir, "w") as f:
        f.write("""#!/bin/bash
#SBATCH --partition={}
#SBATCH --qos="deadline"
#SBATCH --account="deadline"
#SBATCH --gres={}
#SBATCH --mem={}
#SBATCH --array=0-{}%{}
#SBATCH -c {}

source ~/.bashrc
conda activate {}
nvidia-smi
pwd
cd {}

list=(
    {}
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${{list[SLURM_ARRAY_TASK_ID]}}"
eval ${{list[SLURM_ARRAY_TASK_ID]}}\n""".format(args.partition,
                                                args.gres,
                                                args.mem,
                                                len(lst) - 1,
                                                args.limit,
                                                args.cpu,
                                                args.conda_env,
                                                ROOT,
                                                "\n    ".join(lst)))
    print("Done.")
