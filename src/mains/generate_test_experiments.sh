template=$1
partition=$2
python -m src.experiments.generate_runs_from_yaml --dir $template --template test_template.yaml --results_dir test_runs --pattern test_cfg.yaml
python -m src.experiments.generate_sbatch_script --test --dir $template --partition $partition --pattern test_cfg.yaml --out test_batch_run.sh
