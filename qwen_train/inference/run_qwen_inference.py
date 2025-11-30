import os
import torch
from tqdm import tqdm
from transformers import Qwen2Tokenizer, Qwen2ForCausalLM
import pandas as pd

from utils import inference_text


def inference_dataset(data_path, output_file, model_path, temperature, top_p):
    # 加载模型和分词器
    model = Qwen2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    ).to("cuda")

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)

    df = pd.read_csv(data_path)
    texts = df['original_text'].tolist()
    outputs = []
    print(f"已读取{len(texts)}条测试数据")

    emotions = ["joy", "disappointment", "peacefulness", "anger"]
    emotion_chinese = ["喜悦", "失望", "平和", "愤怒"]

    for text in tqdm(texts):
        output = {'original_text': text}
        for emotion, emotion_ch in zip(emotions, emotion_chinese):
            result = inference_text(
                text, 
                emotion_ch, 
                model, 
                tokenizer, 
                temperature, 
                top_p
            )
            output[emotion] = result
        outputs.append(output)

    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df = pd.DataFrame.from_records(outputs)
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f'已经完成{len(texts)}条数据的推理')
    print(f"结果已保存到 {output_file}") 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/research/d7/gds/yztian25/EmotionTransfer/dataset/test.csv")
    parser.add_argument("--model_path", type=str, default="/research/d7/gds/yztian25/EmotionTransfer/models/Qwen2.5-14B-Instruct")
    parser.add_argument("--output_file", type=str, default="/research/d7/gds/yztian25/EmotionTransfer/qwen_train/inference/outputs/emotion_conversion_test_outputs.csv")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    
    args = parser.parse_args()

    inference_dataset(
        args.data_path, 
        args.output_file, 
        args.model_path, 
        args.temperature, 
        args.top_p
    )

    