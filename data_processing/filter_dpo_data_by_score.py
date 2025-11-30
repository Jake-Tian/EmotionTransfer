import json
import os
import random
import math
from typing import Dict, List, Any

def filter_and_convert_jsonl_to_dpo_data(
        input_file: str, 
        output_file: str,
        strongly_rejected_file: str = None
    ):
    """
    Process JSONL file, calculate average of highest and lowest joint_score,
    then only keep samples that meet the criteria: highest >= score_highest_avg && lowest <= score_lowest_avg
    Also filters out any examples with joint_score == 0
    """
    emotion_names = {
        'joy': 'joy',
        'disappointment': 'disappointment',
        'peacefulness': 'peacefulness',
        'anger': 'anger'
    }
    
    # Store all sample information for later processing
    all_samples = []
    highest_scores = []
    lowest_scores = []
    total_samples = 0

    original_highest_map = {}
    
    # Read file once
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            data = json.loads(line.strip())
            original_text = data.get('original_text')

            # Skip if original_text is empty
            if not original_text:
                continue
            
            # Process each emotion label
            for emotion in data.keys():
                if emotion == 'original_text':
                    continue
                
                examples = data[emotion]
                
                # Filter out examples with joint_score == 0
                examples = [ex for ex in examples if float(ex.get('joint_score', 0)) > 0]
                
                # Skip if no valid examples remain
                if len(examples) == 0:
                    continue
                
                total_samples += 1
                
                # Sort by joint_score
                examples.sort(key=lambda x: float(x.get('joint_score')), reverse=True)

                # Skip if highest and lowest scores are the same
                if examples[0].get('joint_score') == examples[-1].get('joint_score'):
                    continue
                
                highest_score = float(examples[0].get('joint_score'))
                lowest_score = float(examples[-1].get('joint_score'))
                # Get highest and lowest score candidates
                highest_score_candidates = [ex for ex in examples if float(ex.get('joint_score')) == highest_score]
                lowest_score_candidates = [ex for ex in examples if float(ex.get('joint_score')) == lowest_score]
                
                # Collect highest and lowest scores
                highest_scores.append(highest_score)
                lowest_scores.append(lowest_score)

                original_highest_map[original_text] = random.choice(highest_score_candidates)['text']
                
                # Store sample information
                all_samples.append({
                    "original_text": original_text,
                    "examples": examples,
                    "highest_score": highest_score,
                    "lowest_score": lowest_score,
                    "emotion": emotion
                })
    
    # Calculate averages
    score_highest_avg = sum(highest_scores) / len(highest_scores) if highest_scores else 0
    score_lowest_avg = sum(lowest_scores) / len(lowest_scores) if lowest_scores else 0

    # Calculate standard deviations
    score_highest_std = math.sqrt(sum((score - score_highest_avg) ** 2 for score in highest_scores) / len(highest_scores)) if highest_scores else 0
    score_lowest_std = math.sqrt(sum((score - score_lowest_avg) ** 2 for score in lowest_scores) / len(lowest_scores)) if lowest_scores else 0
    
    print(f"Average highest score: {score_highest_avg:.3f}")
    print(f"Average lowest score: {score_lowest_avg:.3f}")
    
    # Filter samples based on criteria
    filtered_samples = []
    filtered_count = 0
    highest_score_threshold = score_highest_avg - score_highest_std
    lowest_score_threshold = score_lowest_avg

    print(f"Highest score threshold: {highest_score_threshold:.3f}")
    print(f"Lowest score threshold: {lowest_score_threshold:.3f}")
    for sample in all_samples:
        # Filter condition: highest >= score_highest_avg && lowest <= score_lowest_avg

        if sample["highest_score"] >= highest_score_threshold and sample["lowest_score"] <= lowest_score_threshold:
            filtered_count += 1
            
            # Get highest and lowest score candidates
            highest_score_candidates = [ex for ex in sample["examples"] if float(ex.get('joint_score')) == sample["highest_score"]]
            lowest_score_candidates = [ex for ex in sample["examples"] if float(ex.get('joint_score')) == sample["lowest_score"]]
            
            chosen = random.choice(highest_score_candidates)['text']
            rejected = random.choice(lowest_score_candidates)['text']

            original_highest_map[sample['original_text']] = chosen
            messages = [
                {"role": "system", "content": "You are an expert in emotion transfer for customer reviews. Your task is to rewrite the original text to express the target emotion.\n\n"},
                {"role": "user", "content": f"Original Review:\n{sample['original_text']}\n\nTarget Emotion: {emotion_names[sample['emotion']]}\n\nPlease rewrite the review to express {emotion_names[sample['emotion']]} while maintaining the core facts and details:\n"}
            ]
            output_data = {
                "messages": messages,
                "chosen": {"role": "assistant", "content": chosen},
                "rejected": {"role": "assistant", "content": rejected}
            }
            
            filtered_samples.append(output_data)

    
    print(f"Total samples: {total_samples}")
    print(f"Filtered samples: {filtered_count}")
    print(f"Retention rate: {filtered_count/total_samples:.2%}")
    
    # Shuffle order
    random.shuffle(filtered_samples)

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(filtered_samples, f_out, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, 
                        default='/research/d7/gds/yztian25/EmotionTransfer/qwen_train/evaluate/evaluation_results_samples.json')
    parser.add_argument('--output_file', type=str, 
                        default='/research/d7/gds/yztian25/EmotionTransfer/data_processing/filtered_dpo_data.json')
    
    args = parser.parse_args()
    filter_and_convert_jsonl_to_dpo_data(args.input_file, args.output_file)
    print(f"Conversion completed. Results saved to {args.output_file}") 