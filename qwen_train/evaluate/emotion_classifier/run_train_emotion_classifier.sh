# 设置使用的GPU数量
NUM_GPUS=2

# 设置超参数
torchrun --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12345 \
    train_emotion_classifier.py \
    --model_name_or_path /research/d7/gds/yztian25/EmotionTransfer/models/bge-large-en-v1.5 \
    --train_data_file /research/d7/gds/yztian25/EmotionTransfer/dataset/train.csv \
    --test_data_file /research/d7/gds/yztian25/EmotionTransfer/dataset/test.csv \
    --output_dir /research/d7/gds/yztian25/EmotionTransfer/models/bge-large-en-v1.5-emotion-score \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_seq_length 256 \
    --do_train \
    --logging_dir ./logs \
    --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model f1 \
    --bf16 \
    --gradient_accumulation_steps 1 \
    > train_log.txt 2>&1
