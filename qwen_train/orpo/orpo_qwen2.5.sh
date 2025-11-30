#!/bin/bash

# Set paths
PROJECT_DIR="/research/d7/gds/yztian25/EmotionTransfer"
LLAMA_FACTORY_DIR="${PROJECT_DIR}/LLaMA-Factory"
MODEL_PATH="${PROJECT_DIR}/models/Qwen2.5-0.5B-Instruct-sft"  # Or use DPO checkpoint
OUTPUT_DIR="${PROJECT_DIR}/models/Qwen2.5-0.5B-Instruct-orpo"

cd ${LLAMA_FACTORY_DIR}

# Number of GPUs
NUM_GPUS=4

torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_GPUS} src/train.py \
    --stage dpo \
    --pref_loss orpo \
    --do_train \
    --finetuning_type full \
    --flash_attn fa2 \
    --model_name_or_path ${MODEL_PATH} \
    --dataset emotion_transfer_dpo \
    --template qwen \
    --cutoff_len 512 \
    --overwrite_cache \
    --preprocessing_num_workers 8 \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps 5 \
    --save_strategy epoch \
    --plot_loss \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --bf16 \
    --ddp_timeout 180000000 \
    --val_size 0.0 \
    --per_device_eval_batch_size 32 \
    --eval_strategy no \
    > ${PROJECT_DIR}/qwen_train/orpo/train_log.txt 2>&1