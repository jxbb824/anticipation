#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

# mkdir -p /home/xiruij/anticipation/checkpoints_subset_large
# mkdir -p /home/xiruij/anticipation/logs

# 运行多个任务（0到29）
for TASK_ID in {0..29}; do
    echo "Starting task $TASK_ID"
    
    python finetune.py \
        --output_dir /home/xiruij/anticipation/checkpoints_subset_large/${TASK_ID} \
        --train_file /home/xiruij/anticipation/datasets/finetune/train_v2.txt \
        --valid_file /home/xiruij/anticipation/datasets/finetune/test_v2.txt \
        --subset_ratio 0.5 \
        --seed ${TASK_ID} \
        --epochs 5 \
        --batch_size 16 \
    
    echo "Task $TASK_ID completed"
done