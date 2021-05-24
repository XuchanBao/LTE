# Mimicking (synthetic data)
python -m src.mains.generate_result_table --distribution=uniform

# Mimicking (real data)
python -m src.mains.aggregate_test_results --dataset_name_filter=ED-00004
python -m src.mains.aggregate_test_results --dataset_name_filter=ED-00014
python -m src.mains.aggregate_test_results --dataset_name_filter=ED-00025
python -m src.mains.generate_result_table --distribution=real_data --result_filename=test_resultED-00004 --verbose_result=False
python -m src.mains.generate_result_table --distribution=real_data --result_filename=test_resultED-00014 --verbose_result=False
python -m src.mains.generate_result_table --distribution=real_data --result_filename=test_resultED-00025 --verbose_result=False

# Social welfare
python -m src.mains.generate_result_table --task='social-welfare' --distribution=uniform
python -m src.mains.generate_result_table --task='social-welfare' --distribution=polarized
python -m src.mains.generate_result_table --task='social-welfare' --distribution=indecisive
# social welfare baselines
python -m src.mains.generate_result_table --task='social-welfare' --acc_source=baselines --distribution=uniform
python -m src.mains.generate_result_table --task='social-welfare' --acc_source=baselines --distribution=polarized
python -m src.mains.generate_result_table --task='social-welfare' --acc_source=baselines --distribution=indecisive

# Generalization to unseen voters
python -m src.mains.generate_generalization_result_table --voting_rule=plurality
python -m src.mains.generate_generalization_result_table --voting_rule=borda
python -m src.mains.generate_generalization_result_table --voting_rule=copeland
python -m src.mains.generate_generalization_result_table --voting_rule=maximin
python -m src.mains.generate_generalization_result_table --voting_rule=kemeny

# histograms
python -m scripts.generate_histogram --distribution=uniform --voting_rule=utilitarian
python -m scripts.generate_histogram --distribution=uniform --voting_rule=egalitarian
python -m scripts.generate_histogram --distribution=indecisive --voting_rule=utilitarian
python -m scripts.generate_histogram --distribution=indecisive --voting_rule=egalitarian
python -m scripts.generate_histogram --distribution=polarized --voting_rule=utilitarian
python -m scripts.generate_histogram --distribution=polarized --voting_rule=egalitarian
