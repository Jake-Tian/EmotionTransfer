
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from bert_score import BERTScorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TextEmotionTransferEvaluator:
    def __init__(
        self, 
        bert_model_path, 
        emotion_classifier_path, 
        device = 'cuda:0', 
        batch_size=64
    ):

        # Default to English
        if 'bert-base-uncased' in bert_model_path or 'roberta-base' in bert_model_path or 'bert-base' in bert_model_path:
            num_layers = 8
            lang = 'en'
        elif 'bert-base-chinese' in bert_model_path:
            num_layers = 8
            lang = 'zh'
        else:
            # Default to English for other models
            num_layers = 8
            lang = 'en'
        
        self.bert_config = {'model_type': bert_model_path, 'num_layers': num_layers, 'lang': lang}


        self.bert_scorer = BERTScorer(
            model_type=self.bert_config['model_type'], 
            num_layers=self.bert_config['num_layers'], 
            lang=self.bert_config['lang'], 
            batch_size=batch_size,
            device=device
        )

        self.device = torch.device(device)
        self.emotion_classifier = AutoModelForSequenceClassification.from_pretrained(emotion_classifier_path).to(self.device).to(torch.float16)
        self.emotion_classifier.eval()
        self.emotion_classifier_tokenizer = AutoTokenizer.from_pretrained(emotion_classifier_path)
        
        self.batch_size = batch_size

    def calculate_bert_score(self, cands, refs):

        P, R, F1 = self.bert_scorer.score(cands, refs)

        content_scores = [max(s, 0) * 100 for s in F1.tolist()]
        content_scores = np.array(content_scores)
        return round(content_scores.mean(), 1), content_scores
    
    def calculate_emotion_score(self, texts, labels):
        label_list = ["joy", "disappointment", "peacefulness", "anger"]
        id2label = {i: label for i, label in enumerate(label_list)}
        
        # 结果列表
        predict_corrects = []
        
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i+self.batch_size]
            batch_labels = labels[i:i+self.batch_size]

            # 对输入文本进行处理
            inputs = self.emotion_classifier_tokenizer(
                batch_texts, 
                return_tensors="pt", 
                truncation=True, 
                padding='longest', 
                max_length=256
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.emotion_classifier(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class_ids = torch.argmax(probabilities, dim=-1).cpu().tolist()
            
            
            for j, text_id in enumerate(range(i, min(i+self.batch_size, len(texts)))):
                predicted_label = id2label[predicted_class_ids[j]]
                # probs = probabilities[j].cpu().numpy()

                predict_corrects.append(int(batch_labels[j] == predicted_label))
            
        predict_corrects = np.array(predict_corrects)
        return round(100 * predict_corrects.mean(), 1), predict_corrects

    
    def evaluate(self, source_texts, output_texts, labels):
        # calculate content preservation score
        print("evaluating content preservation score...")
        content_score, content_scores = self.calculate_bert_score(output_texts, source_texts)

        # calculate emotion score
        print("evaluating emotion score...")
        emotion_score, emotion_scores = self.calculate_emotion_score(output_texts, labels)

        # calculate joint scores
        print("evaluating joint score...")
        joint_scores = content_scores * emotion_scores
        joint_score = round(joint_scores.mean(), 1)

        return content_score, emotion_score, joint_score
    
    def evaluate_dataset(
        self, 
        data_path, 
        output_file
    ):

        dataset = pd.read_csv(data_path, encoding='utf-8')
        dataset = dataset.to_dict(orient='records')
        source_texts, target_texts = [], []
        labels = []

        emotions = ['joy', 'disappointment', 'peacefulness', 'anger']

        for data in dataset:
            original_text = data['original_text']
            for emotion in emotions:
                target_text = data[emotion]
                source_texts.append(original_text)
                target_texts.append(target_text)
                labels.append(emotion)
        print(f"evaluating {len(source_texts)} samples...")

        content_score, emotion_score, joint_score = self.evaluate(source_texts, target_texts, labels)
        
        result = {
            'content_score': content_score,
            'emotion_score': emotion_score,
            'joint_score': joint_score
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"content score: {content_score}, emotion score: {emotion_score}, joint score: {joint_score}")
        print(f"finished evaluating {len(source_texts)} samples, saved to {output_file}")

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/research/d7/gds/yztian25/EmotionTransfer/dataset/yelp_with_emotions.csv')
    parser.add_argument('--output_file', type=str, default='/research/d7/gds/yztian25/EmotionTransfer/qwen_train/evaluate/evaluation_results.json')
    parser.add_argument('--bert_model_path', type=str, default='bert-base-uncased')
    parser.add_argument('--emotion_classifier_path', type=str, default='/research/d7/gds/yztian25/EmotionTransfer/models/bge-large-en-v1.5-emotion-score')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    evaluator = TextEmotionTransferEvaluator(
        bert_model_path=args.bert_model_path,
        emotion_classifier_path=args.emotion_classifier_path,
        device=args.device,
        batch_size=args.batch_size
    )
    
    evaluator.evaluate_dataset(
        data_path=args.data_path,
        output_file=args.output_file
    )