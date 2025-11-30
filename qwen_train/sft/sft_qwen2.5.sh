#!/bin/bash

# Set paths
PROJECT_DIR="/research/d7/gds/yztian25/EmotionTransfer"
LLAMA_FACTORY_DIR="${PROJECT_DIR}/LLaMA-Factory"
MODEL_PATH="${PROJECT_DIR}/models/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR="${PROJECT_DIR}/models/Qwen2.5-0.5B-Instruct-sft"

cd ${LLAMA_FACTORY_DIR}

# Number of GPUs
NUM_GPUS=2  # Adjust based on your setup

torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_GPUS} src/train.py \
    --stage sft \
    --do_train \
    --finetuning_type full \
    --flash_attn fa2 \
    --model_name_or_path ${MODEL_PATH} \
    --dataset emotion_transfer_train \
    --template qwen \
    --cutoff_len 512 \
    --overwrite_cache \
    --preprocessing_num_workers 8 \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps 10 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --plot_loss \
    --overwrite_output_dir \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000 \
    --val_size 0.1 \
    --per_device_eval_batch_size 16 \
    > ${PROJECT_DIR}/qwen_train/sft/train_log.txt 2>&1