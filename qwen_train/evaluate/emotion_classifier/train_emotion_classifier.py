# train.py
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorWithPadding
)
from transformers import BertForSequenceClassification
import json
from datasets import load_dataset
import numpy as np
from dataclasses import dataclass, field
import os
import torch
from sklearn.metrics import accuracy_score, f1_score



# 定义参数类
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="BAAI/bge-base-zh",
        metadata={"help": "Pretrained model name or path"}
    )

@dataclass
class DataArguments:
    train_data_file: str = field(
        default="./data/train.csv",
        metadata={"help": "Training file"}
    )
    test_data_file: str = field(
        default="./data/test.csv",
        metadata={"help": "Testing file"}
    )
    max_seq_length: int = field(
        default=256,
        metadata={"help": "Maximum sequence length for tokenization"}
    )

def main():
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 标签映射
    label_list = ["joy", "disappointment", "peacefulness", "anger"]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # 设置模型为fp16
    if training_args.fp16:
        model.to(torch.float16)
    elif training_args.bf16:
        model.to(torch.bfloat16)
    

    # 加载数据集
    data_files = {
        "train": data_args.train_data_file,
        "test": data_args.test_data_file,
    }
    dataset = load_dataset("csv", data_files=data_files)

    # # convert label to id for train and test dataset
    # for split in ["train", "test"]:
    #     dataset[split]["label"] = [label2id[label] for label in dataset[split]["label"]]

    # 预处理函数
    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="longest",
            truncation=True,
            max_length=data_args.max_seq_length,
        )
        tokenized["label"] = [label2id[label] for label in examples["label"]]

        return tokenized

    processed_dataset = dataset.map(preprocess_function, batched=True)

    # 初始化Trainer
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="macro"),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset.get("test"),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )


    # 训练和评估
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
    
    if training_args.do_eval:
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
        
        # 保存评估结果和最佳模型信息
        import json
        best_checkpoint_info = {
            "eval_results": eval_results,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric
        }
        with open(os.path.join(training_args.output_dir, "best_model_info.json"), "w") as f:
            json.dump(best_checkpoint_info, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()