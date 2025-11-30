
# 设置默认参数
PROJECT_DIR="/research/d7/gds/yztian25/EmotionTransfer"
BERT_MODEL_PATH="bert-base-uncased"
EMOTION_CLASSIFIER_PATH="${PROJECT_DIR}/models/bge-large-en-v1.5-emotion-score"
DEVICE="cuda:0"
BATCH_SIZE=64
INPUT_JSON="${PROJECT_DIR}/qwen_train/inference/outputs/emotion_conversion_multiple_samples.json"
OUTPUT_JSON="${PROJECT_DIR}/qwen_train/evaluate/evaluation_results_samples.json"

# 运行评估脚本
python evaluate_samples.py \
  --input_json ${INPUT_JSON} \
  --output_json ${OUTPUT_JSON} \
  --bert_model_path ${BERT_MODEL_PATH} \
  --emotion_classifier_path ${EMOTION_CLASSIFIER_PATH} \
  --device ${DEVICE} \
  --batch_size ${BATCH_SIZE}

echo "评估完成，结果保存在: ${OUTPUT_JSON}" 