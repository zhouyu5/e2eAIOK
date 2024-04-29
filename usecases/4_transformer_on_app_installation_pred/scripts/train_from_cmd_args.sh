#!/bin/bash

# warmup_steps: 1000, 2000
# per_device_train_batch_size: 2048, 4096
# use_ipex: true, false
# use_hpu: true, false

use_hpu=false
dev_mode=true

time python -m recsys_kit.train \
    --dev_mode $dev_mode \
    --model_config_name_or_path config/model_config.json \
    --preprocess_dataset_path processed_data/sharechat_local \
    --metrics roc_auc \
    --output_dir checkpoint/tmp \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --warmup_steps 1000 \
    --per_device_train_batch_size 2048 \
    --per_device_eval_batch_size 4096 \
    --do_train \
    --do_eval \
    --lr_scheduler_type cosine \
    --weight_decay 0 \
    --save_strategy steps \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --dataloader_num_workers 8 \
    --use_hpu $use_hpu \
    --gaudi_config_name_or_path config/gaudi_config.json 
