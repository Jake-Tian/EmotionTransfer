import json
import random

def convert_to_kto_format(input_data):
    kto_data = []
    
    for item in input_data:
        # 创建正样本（chosen）
        chosen_sample = {
            "messages": item["messages"] + [{"role": "assistant", "content": item["chosen"]["content"]}],
            "kto_tag": True
        }
        
        # 创建负样本（rejected）
        rejected_sample = {
            "messages": item["messages"] + [{"role": "assistant", "content": item["rejected"]["content"]}],
            "kto_tag": False
        }
        
        # 添加到结果列表
        kto_data.append(chosen_sample)
        kto_data.append(rejected_sample)
    
    return kto_data

def convert_to_kto_format_from_file(file_path, output_file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    output_data = convert_to_kto_format(input_data)
    random.shuffle(output_data)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert data to KTO format')
    parser.add_argument('--input_file', type=str, default="/home/yewenxuan/rl_text_style_transfer/data/rlhf/20240426/filtered_dpo_train_data.json")
    parser.add_argument('--output_file', type=str, default="/home/yewenxuan/rl_text_style_transfer/data/rlhf/20240426/filtered_kto_train_data.json")
    args = parser.parse_args()

    convert_to_kto_format_from_file(args.input_file, args.output_file)
