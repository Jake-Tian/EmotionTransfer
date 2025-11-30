# batch_inference.py
import argparse
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='情感分类器批量推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型保存路径')
    parser.add_argument('--input_file', type=str, required=True, help='输入CSV文件路径，需包含text列')
    parser.add_argument('--output_file', type=str, required=True, help='输出CSV文件路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='运行设备 (cuda/cpu)')
    parser.add_argument('--fp16', action='store_true', help='使用FP16精度')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 标签映射
    label_list = ["joy", "disappointment", "peacefulness", "anger"]
    id2label = {i: label for i, label in enumerate(label_list)}
    
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    
    # 设置设备和精度
    device = torch.device(args.device)
    model = model.to(device)
    
    if args.fp16:
        model = model.half()
    
    # 加载数据
    df = pd.read_csv(args.input_file)
    if 'text' not in df.columns:
        raise ValueError("输入CSV文件必须包含'text'列")
    
    texts = df['text'].tolist()
    
    # 结果列表
    results = []
    
    # 批量处理
    model.eval()
    for i in tqdm(range(0, len(texts), args.batch_size), desc="批量推理"):
        batch_texts = texts[i:i+args.batch_size]
        
        # 对输入文本进行处理
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, 
                          padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_ids = torch.argmax(probabilities, dim=-1).cpu().tolist()
        
        # 获取结果
        for j, text_id in enumerate(range(i, min(i+args.batch_size, len(texts)))):
            predicted_label = id2label[predicted_class_ids[j]]
            probs = probabilities[j].cpu().numpy()
            
            result = {
                'text': batch_texts[j],
                'predicted_label': predicted_label,
            }
            
            # 添加每个情感的概率
            for label_id, label in id2label.items():
                result[f'prob_{label}'] = float(probs[label_id])
            
            results.append(result)
    
    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv(args.output_file, index=False)
    print(f"处理完成，共处理 {len(texts)} 条文本，结果已保存到 {args.output_file}")

if __name__ == "__main__":
    main() 