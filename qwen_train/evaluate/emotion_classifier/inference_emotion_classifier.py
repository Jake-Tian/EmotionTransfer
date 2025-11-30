# inference.py
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser(description='情感分类器推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型保存路径')
    parser.add_argument('--input_text', type=str, required=True, help='要分类的文本')
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
    
    # 对输入文本进行处理
    inputs = tokenizer(args.input_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 推理
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    
    # 获取结果
    predicted_label = id2label[predicted_class_id]
    probability = probabilities[0][predicted_class_id].item()
    
    # 输出所有情感的概率
    all_probs = {id2label[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    
    print(f"输入文本: {args.input_text}")
    print(f"预测情感: {predicted_label}")
    print(f"置信度: {probability:.4f}")
    print("所有情感概率:")
    for emotion, prob in all_probs.items():
        print(f"  {emotion}: {prob:.4f}")

if __name__ == "__main__":
    main() 