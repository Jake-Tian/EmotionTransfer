# api_server.py
import argparse
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify

app = Flask(__name__)

# 全局变量存储模型和tokenizer
MODEL = None
TOKENIZER = None
DEVICE = None
ID2LABEL = None

def load_model(model_path, device='cuda', fp16=False):
    global MODEL, TOKENIZER, DEVICE, ID2LABEL
    
    # 标签映射
    label_list = ["joy", "disappointment", "peacefulness", "anger"]
    ID2LABEL = {i: label for i, label in enumerate(label_list)}
    
    # 加载模型和tokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(model_path)
    MODEL = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # 设置设备和精度
    DEVICE = torch.device(device)
    MODEL = MODEL.to(DEVICE)
    
    if fp16:
        MODEL = MODEL.half()
    
    # 设置为评估模式
    MODEL.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json or 'text' not in request.json:
        return jsonify({'error': '请求必须包含text字段'}), 400
    
    text = request.json['text']
    
    # 对输入文本进行处理
    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # 推理
    with torch.no_grad():
        outputs = MODEL(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    
    # 获取结果
    predicted_label = ID2LABEL[predicted_class_id]
    
    # 构建所有情感的概率字典
    all_probs = {ID2LABEL[i]: float(prob) for i, prob in enumerate(probabilities[0].cpu().numpy())}
    
    result = {
        'text': text,
        'predicted_emotion': predicted_label,
        'confidence': all_probs[predicted_label],
        'probabilities': all_probs
    }
    
    return jsonify(result)

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if not request.json or 'texts' not in request.json:
        return jsonify({'error': '请求必须包含texts字段'}), 400
    
    texts = request.json['texts']
    
    if not isinstance(texts, list):
        return jsonify({'error': 'texts字段必须是文本列表'}), 400
    
    # 对输入文本进行处理
    inputs = TOKENIZER(texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # 推理
    results = []
    with torch.no_grad():
        outputs = MODEL(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_ids = torch.argmax(probabilities, dim=-1).cpu().tolist()
    
    # 获取结果
    for i, text in enumerate(texts):
        predicted_label = ID2LABEL[predicted_class_ids[i]]
        all_probs = {ID2LABEL[j]: float(prob) for j, prob in enumerate(probabilities[i].cpu().numpy())}
        
        result = {
            'text': text,
            'predicted_emotion': predicted_label,
            'confidence': all_probs[predicted_label],
            'probabilities': all_probs
        }
        results.append(result)
    
    return jsonify({'results': results})

def parse_args():
    parser = argparse.ArgumentParser(description='情感分类器API服务')
    parser.add_argument('--model_path', type=str, required=True, help='模型保存路径')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='运行设备 (cuda/cpu)')
    parser.add_argument('--fp16', action='store_true', help='使用FP16精度')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"加载模型: {args.model_path}")
    load_model(args.model_path, device=args.device, fp16=args.fp16)
    print(f"启动服务器: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port) 