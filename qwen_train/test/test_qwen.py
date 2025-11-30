# from transformers import AutoModelForCausalLM, AutoTokenizer
## import qwenforcausalllm and qwentokenizer
from transformers import Qwen2Tokenizer, Qwen2ForCausalLM
import torch


# model_path = "/data1/yewenxuan/model_gallery/Qwen2.5-0.5B-Instruct"
# model_path = "/data1/yewenxuan/save_dir/Qwen2.5-1.5B-Instruct-sft/20250416/checkpoint-145"
model_path = "/data1/yewenxuan/save_dir/Qwen2.5-1.5B-Instruct-Tune/dpo/20250426_lr3e-6/checkpoint-162"

model = Qwen2ForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
).to("cuda")
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)

def generate_text(prompt, model, tokenizer, max_new_tokens=512, temperature=0.7, top_p=0.6):
    """
    生成文本的函数
    Args:
        prompt: 输入提示文本
        model: 加载的模型
        tokenizer: 分词器
        max_new_tokens: 最大生成token数
        temperature: 采样温度
        top_p: top-p采样参数
    Returns:
        生成的文本响应
    """
    messages = [
        {"role": "system", "content": "你是一个电商评论文本情绪转换大师，你的任务是把原本的文本转化为目标情绪。\n\n"},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # print(text)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=True,
        top_p=top_p,
        temperature=temperature
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

if __name__ == "__main__":
    original_text = "嘻嘻，很不错哦，一直都只买这一品牌的"
    target_emotion = "失望"
    prompt = f"""[原文本]：\n{original_text}\n\n[目标情绪]：\n{target_emotion}\n\n转化后文本：\n"""
    # print(f"prompt: {prompt}")
    
    print("response:")
    for _ in range(10):
        print(generate_text(prompt, model, tokenizer, max_new_tokens=512, temperature=0.7, top_p=0.7))
