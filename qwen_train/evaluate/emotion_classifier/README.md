# BGE情感分类器

这是一个基于BGE模型的情感分类器，可以将文本分类为四种情感：喜悦(joy)、失望(disappointment)、平静(peacefulness)和愤怒(anger)。

## 目录结构

```
emotion_classifier/
├── train_emotion_classifier.py  # 训练脚本
├── inference_emotion_classifier.py  # 单文本推理脚本
├── batch_inference.py  # 批量推理脚本
├── api_server.py  # API服务器
└── README.md  # 使用说明
```

## 安装依赖

```bash
pip install transformers datasets torch pandas flask tqdm sklearn
```

## 训练模型

### 数据格式

训练和测试数据应为CSV格式，包含两列：`text` 和 `label`。标签必须是以下四种之一：joy、disappointment、peacefulness、anger。

示例：

```csv
text,label
我今天很开心,joy
这个结果让我很失望,disappointment
看着平静的湖面我感到心情平和,peacefulness
他的话让我非常生气,anger
```

### 训练命令

```bash
python train_emotion_classifier.py \
  --model_name_or_path "BAAI/bge-base-zh" \
  --train_data_file "./data/train.csv" \
  --test_data_file "./data/test.csv" \
  --max_seq_length 256 \
  --output_dir "./output" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --fp16 \
  --do_train \
  --do_eval \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --load_best_model_at_end \
  --metric_for_best_model "f1"
```

## 推理

### 单文本推理

```bash
python inference_emotion_classifier.py \
  --model_path "./output" \
  --input_text "这个结果让我非常开心" \
  --device cuda \
  --fp16
```

### 批量推理

```bash
python batch_inference.py \
  --model_path "./output" \
  --input_file "./data/to_predict.csv" \
  --output_file "./data/predictions.csv" \
  --batch_size 32 \
  --device cuda \
  --fp16
```

输入文件必须是CSV格式，并包含`text`列。输出文件将包含原始文本、预测的情感标签和每个情感类别的概率。



