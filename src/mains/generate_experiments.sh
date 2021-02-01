template=$1
partition=$2
python -m src.experiments.generate_runs_from_yaml --dir $template
python -m src.experiments.generate_sbatch_script --dir $template --partition $partition
#python -m src.experiments.generate_sbatch_script --dir $template --resume --test --out batch_run_resume_test.sh
#python -m src.experiments.generate_sbatch_script --dir $template --resume --out batch_run_resume.sh
#python -m src.experiments.generate_sbatch_script --dir $template --test --out batch_run_test.sh
