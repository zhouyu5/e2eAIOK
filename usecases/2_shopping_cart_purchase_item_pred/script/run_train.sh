# Data Process
#! /bin/bash
python src/dataprep/data_prepare.py --data_dir dataset/dressipi_recsys2022_dataset --feat_dir feat --save_dir output --train

# # augment data for pretrain
# python src/dataprep/data_prepare.py --data_dir output/processed --augment

# Model Train
python -u src/train/cli.py --dataset-dir output/processed/  --feat-dir feat/ --save-path output/saved_models --epochs 1
# python -u src/train/cli.py --dataset-dir output/processed/  --feat-dir feat/ --save-path output/saved_models
