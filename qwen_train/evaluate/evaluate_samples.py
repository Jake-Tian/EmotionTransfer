import json
import numpy as np
import torch
from tqdm import tqdm
import argparse
import os
import time
import re
from openai import OpenAI
from evaluate import TextEmotionTransferEvaluator  # 导入原有评估器类

class SampleLevelEvaluator(TextEmotionTransferEvaluator):
    """
    针对样本级别的评估器，继承自TextEmotionTransferEvaluator
    用于评估每个样本的内容保存得分和情感得分
    """
    
    def __init__(self, *args, openai_api_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.openai_client = OpenAI()
    
    def calculate_emotion_score_gpt(self, texts, labels):
        """
        使用GPT-4o-mini计算情感得分
        
        Args:
            texts (list): 待评估的文本列表
            labels (list): 目标情感标签列表
        
        Returns:
            tuple: (平均得分, 得分数组)
        """
        emotion_scores = []
        
        for text, label in tqdm(zip(texts, labels), total=len(texts), desc="Evaluating emotions with GPT-4o-mini"):
            prompt = f"""You are an emotion classification expert. Evaluate how well the following text expresses the target emotion.

Target emotion: {label}

Text to evaluate: "{text}"

Respond with ONLY a number between 0 and 1 (inclusive), where:
- 1.0 means the text clearly and strongly expresses the target emotion
- 0.0 means the text does not express the target emotion at all
- Values in between represent partial or moderate expression of the emotion

Output only the number (e.g., 0.8, 0.5, 1.0, 0.0) without any additional text or explanation."""
            
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a precise emotion classification evaluator. Respond with only a number between 0 and 1."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0,
                        max_tokens=10
                    )
                    
                    result = response.choices[0].message.content.strip()
                    # Extract float value between 0 and 1 from response
                    try:
                        # Try to extract a float from the response
                        # Look for a number (integer or float) in the response
                        numbers = re.findall(r'\d+\.?\d*', result)
                        if numbers:
                            score = float(numbers[0])
                            # Clamp to [0, 1] range
                            score = max(0.0, min(1.0, score))
                        else:
                            # If no number found, default to 0
                            print(f"Warning: No number found in GPT response: {result}, defaulting to 0")
                            score = 0.0
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse GPT response: {result}, defaulting to 0. Error: {e}")
                        score = 0.0
                    
                    emotion_scores.append(score)
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        print(f"Error calling OpenAI API after {max_retries} attempts: {e}")
                        # Default to 0 on error
                        emotion_scores.append(0.0)
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        emotion_scores = np.array(emotion_scores)
        return round(emotion_scores.mean(), 3), emotion_scores
    
    def evaluate_samples(self, original_text, emotion_texts, emotion):
        """
        评估指定情感的多个文本样本
        
        Args:
            original_text (str): 原始文本
            emotion_texts (list): 同一情感下的多个生成文本
            emotion (str): 目标情感
        
        Returns:
            list: 包含每个样本评估结果的列表
        """
        # 批量计算内容保存得分 (bert score)
        # 将原始文本复制多次以匹配生成文本数量
        refs = [original_text] * len(emotion_texts)
        _, content_scores = self.calculate_bert_score(emotion_texts, refs)

        
        # 使用GPT-4o-mini计算情感得分 (0到1之间的连续值)
        labels = [emotion] * len(emotion_texts)
        _, emotion_scores = self.calculate_emotion_score_gpt(emotion_texts, labels)

        # joint scores
        joint_scores = content_scores * emotion_scores

        content_scores = content_scores.tolist()
        emotion_scores = emotion_scores.tolist()
        joint_scores = joint_scores.tolist()

        # 整合结果
        results = []
        for i, text in enumerate(emotion_texts):
            results.append({
                'text': text,
                'content_preservation_score': round(float(content_scores[i]), 1),
                'style_score': round(float(emotion_scores[i]), 3),
                'joint_score': round(float(joint_scores[i]), 3)
            })
        
        return results
    
    def evaluate_json_data(self, input_json, output_json):
        """
        评估JSON格式的输入数据，并输出评估结果
        
        Args:
            input_json (str): 输入JSON文件的路径
            output_json (str): 输出JSON文件的路径
        """
        # 加载输入数据
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        with open(output_json, 'w', encoding='utf-8') as f:

            # 处理每个样本
            for item in data:
                original_text = item['original_text']
                
                # 处理每种情感的文本
                for emotion in ["joy", "disappointment", "peacefulness", "anger"]:
                    if emotion in item:
                        emotion_texts = item[emotion]
                        item[emotion] = self.evaluate_samples(original_text, emotion_texts, emotion)
                output_str = json.dumps(item, ensure_ascii=False)
                f.write(output_str + '\n')
        
        print(f"评估完成，结果已保存至 {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="样本级别的情感转换评估工具")
    parser.add_argument('--input_json', type=str, required=True, help='输入JSON文件的路径')
    parser.add_argument('--output_json', type=str, required=True, help='输出JSON文件的路径')
    parser.add_argument('--bert_model_path', type=str, default='bert-base-uncased', 
                        help='BERT模型路径')
    parser.add_argument('--emotion_classifier_path', type=str, 
                        default='/research/d7/gds/yztian25/EmotionTransfer/models/bge-large-en-v1.5-emotion-score', 
                        help='情感分类器路径（仅用于BERT score，情感评估使用GPT-4o-mini）')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument('--batch_size', type=int, default=64, help='批处理大小')
    parser.add_argument('--openai_api_key', type=str, default=None, 
                        help='OpenAI API key (也可以使用OPENAI_API_KEY环境变量)')
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = SampleLevelEvaluator(
        bert_model_path=args.bert_model_path,
        emotion_classifier_path=args.emotion_classifier_path,
        device=args.device,
        batch_size=args.batch_size,
        openai_api_key=args.openai_api_key
    )
    
    # 执行评估
    evaluator.evaluate_json_data(args.input_json, args.output_json) 