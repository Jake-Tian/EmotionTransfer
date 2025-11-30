import json
import os
import random
from typing import Dict, List, Any

def convert_jsonl_to_dpo_data(chosen_file: str, rejected_file: str, output_file: str):
    """
    处理JSONL文件，将每行转换为指定的格式
    """
    emotion_en_zh = {
        'joy': '喜悦',
        'disappointment': '失望',
        'peacefulness': '平和',
        'anger': '愤怒'
    }
    original_text_chosen_map = {}
    outputs = []

    with open(chosen_file, 'r', encoding='utf-8') as f_in:
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
                # messages =[ 
                #     {"role": "system", "content": "你是一个电商评论文本情绪转换大师，你的任务是把原本的文本转化为目标情绪。\n\n"},
                #     {"role": "user", "content": f"[原文本]：\n{original_text}\n\n[目标情绪]：\n{emotion_en_zh[emotion]}\n\n转化后文本：\n"}
                # ]
                examples = data[emotion]
                
                # 根据joint_score排序
                examples.sort(key=lambda x: float(x.get('joint_score')), reverse=True)
                
                if float(examples[0].get('joint_score')) == 0.0:
                    continue
                
                # 获取最高分
                highest_score_candidates = [ex for ex in examples if ex.get('joint_score') == examples[0].get('joint_score')]
                
                chosen = random.choice(highest_score_candidates)['text']
                # chosen = {"role": "assistant", "content": chosen}
                
                original_text_chosen_map[(original_text, emotion)] = chosen

    with open(rejected_file, 'r', encoding='utf-8') as f_in:
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
                # messages =[ 
                #     {"role": "system", "content": "你是一个电商评论文本情绪转换大师，你的任务是把原本的文本转化为目标情绪。\n\n"},
                #     {"role": "user", "content": f"[原文本]：\n{original_text}\n\n[目标情绪]：\n{emotion_en_zh[emotion]}\n\n转化后文本：\n"}
                # ]
                examples = data[emotion]
                
                # 根据joint_score排序
                examples.sort(key=lambda x: float(x.get('joint_score')), reverse=True)

                
                # 获取最低分
                if examples[-1].get('joint_score') > 53.0:
                    continue
                lowest_score_candidates = [ex for ex in examples if ex.get('joint_score') == examples[-1].get('joint_score')]
                rejected = random.choice(lowest_score_candidates)['text']
                # if (original_text, emotion) not in original_text_chosen_map:
                #     continue
                chosen = {"role": "assistant", "content": original_text_chosen_map[(original_text, emotion)]}
                rejected = {"role": "assistant", "content": rejected}
                messages = [
                    {"role": "system", "content": "你是一个电商评论文本情绪转换大师，你的任务是把原本的文本转化为目标情绪。\n\n"},
                    {"role": "user", "content": f"[原文本]：\n{original_text}\n\n[目标情绪]：\n{emotion_en_zh[emotion]}\n\n转化后文本：\n"}
                ]
                output = {
                    "messages": messages,
                    "chosen": chosen,
                    "rejected": rejected
                }
                outputs.append(output)

    
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
    parser.add_argument('--chosen_file', type=str, default='/home/yewenxuan/rl_text_style_transfer/data/rlhf/20240426/strongly_rejected/b/scored_strongly_rejected_b_sft_samples.jsonl')
    parser.add_argument('--rejected_file', type=str, default='/home/yewenxuan/rl_text_style_transfer/data/rlhf/20240426/strongly_rejected/b/scored_strongly_rejected_b_dpo_samples.jsonl')
    parser.add_argument('--output_file', type=str, default='/home/yewenxuan/rl_text_style_transfer/data/rlhf/20240426/strongly_rejected/b/dpo_data_from_two_models.json')
    
    args = parser.parse_args()
    convert_jsonl_to_dpo_data(args.chosen_file, args.rejected_file, args.output_file)
    print(f"转换完成，结果已保存到 {args.output_file}") 