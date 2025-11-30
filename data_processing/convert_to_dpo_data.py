import json
import os
import random
from typing import Dict, List, Any

def convert_jsonl_to_dpo_data(input_file: str, output_file: str):
    """
    处理JSONL文件，将每行转换为指定的格式
    """
    emotion_en_zh = {
        'joy': '喜悦',
        'disappointment': '失望',
        'peacefulness': '平和',
        'anger': '愤怒'
    }
    outputs = []

    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            data = json.loads(line.strip())
            original_text = data.get('original_text')

            # 如果original_text为空，则跳过
            if not original_text:
                continue
            
            # 处理每个情感标签
            for emotion in data.keys():
                if emotion == 'original_text':
                    continue
                messages =[ 
                    {"role": "system", "content": "你是一个电商评论文本情绪转换大师，你的任务是把原本的文本转化为目标情绪。\n\n"},
                    {"role": "user", "content": f"[原文本]：\n{original_text}\n\n[目标情绪]：\n{emotion_en_zh[emotion]}\n\n转化后文本：\n"}
                ]
                examples = data[emotion]
                
                # 根据joint_score排序
                examples.sort(key=lambda x: float(x.get('joint_score')), reverse=True)

                # 如果最高分和最低分相同，则跳过
                if examples[0].get('joint_score') == examples[-1].get('joint_score'):
                    continue
                
                # 获取最高分和最低分
                highest_score_candidates = [ex for ex in examples if ex.get('joint_score') == examples[0].get('joint_score')]
                lowest_score_candidates = [ex for ex in examples if ex.get('joint_score') == examples[-1].get('joint_score')]
                
                chosen = random.choice(highest_score_candidates)['text']
                rejected = random.choice(lowest_score_candidates)['text']
                chosen = {"role": "assistant", "content": chosen}
                rejected = {"role": "assistant", "content": rejected}
                
                output_data = {
                    "messages": messages,
                    "chosen": chosen,
                    "rejected": rejected
                }
                
                outputs.append(output_data)
    
    # 打乱顺序
    random.shuffle(outputs)

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(outputs, f_out, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='../../data/rlhf/inference_results/scored_rlhf_train_data_samples.jsonl')
    parser.add_argument('--output_file', type=str, default='../../data/rlhf/rlhf_train_data.json')
    
    args = parser.parse_args()
    convert_jsonl_to_dpo_data(args.input_file, args.output_file)
    print(f"转换完成，结果已保存到 {args.output_file}") 