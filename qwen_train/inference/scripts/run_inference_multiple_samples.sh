#!/bin/bash

# Set default parameters
PROJECT_DIR="/research/d7/gds/yztian25/EmotionTransfer"
DATA_PATH="${PROJECT_DIR}/dataset/yelp_800.csv"
MODEL_PATH="${PROJECT_DIR}/models/Qwen2.5-0.5B-Instruct-sft"
OUTPUT_FILE="${PROJECT_DIR}/qwen_train/inference/outputs/emotion_conversion_multiple_samples.json"
TEMPERATURE=0.9
TOP_P=0.9
N_SAMPLES=10
BATCH_SIZE=10
NUM_WORKERS=6  # Adjust based on your GPU memory and CPU cores
DEVICE="cuda:0"

cd ${PROJECT_DIR}/qwen_train/inference

# Run Python script
python3 run_qwen_inference_multiple_samples.py \
  --data_path "$DATA_PATH" \
  --model_path "$MODEL_PATH" \
  --output_file "$OUTPUT_FILE" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --n_samples "$N_SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE" 