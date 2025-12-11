#!/bin/bash

python3 hyperparameter_search_optuna.py --trials 100 --data_dir ./data --num_epochs 400 --train_fraction 0.8 --random_seed 42 --log_file optuna_trials_history.tsv | tee log
