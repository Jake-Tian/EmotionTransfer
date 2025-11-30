import json
import os

def clean_dpo_data(input_file: str, output_file: str):
    """
    Remove samples where chosen response contains instruction text
    """
    instruction_keywords = [
        'hints:', 'hint:', 'instructions:', 'instruction:', 
        'understand the key points', 'consider', 'express specific examples',
        'use words and phrases', 'rewritten review', 'rewrite the review',
        'target emotion', 'original review', 'please rewrite', '____'
    ]
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    cleaned_data = []
    removed_count = 0
    
    for sample in data:
        chosen = sample['chosen']['content']
        chosen_lower = chosen.lower()
        
        # Check if chosen contains instruction text
        has_instruction = False
        for keyword in instruction_keywords:
            if keyword in chosen_lower:
                has_instruction = True
                removed_count += 1
                break
        
        if not has_instruction:
            cleaned_data.append(sample)
    
    # Save cleaned data
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"Original samples: {original_count}")
    print(f"Removed samples (chosen with instruction text): {removed_count} ({removed_count/original_count*100:.1f}%)")
    print(f"Remaining samples: {len(cleaned_data)} ({len(cleaned_data)/original_count*100:.1f}%)")
    print(f"Cleaned dataset saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, 
                        default='/research/d7/gds/yztian25/EmotionTransfer/data_processing/filtered_dpo_data.json')
    parser.add_argument('--output_file', type=str, 
                        default='/research/d7/gds/yztian25/EmotionTransfer/data_processing/filtered_dpo_data_cleaned.json')
    
    args = parser.parse_args()
    clean_dpo_data(args.input_file, args.output_file)

