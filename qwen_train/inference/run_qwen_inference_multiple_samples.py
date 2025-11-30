import random
import torch
from tqdm import tqdm
from transformers import Qwen2Tokenizer, Qwen2ForCausalLM
import pandas as pd
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_text_batch(prompts, model, tokenizer, max_new_tokens=512, temperature=0.7, top_p=0.6, batch_size=8):
    """
    Generate text for a batch of prompts
    Args:
        prompts: List of prompt texts
        model: Loaded model
        tokenizer: Tokenizer
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: top-p sampling parameter
        batch_size: Batch size for generation
    Returns:
        List of generated text responses
    """
    all_results = []
    
    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        messages_list = [[{"role": "user", "content": prompt}] for prompt in batch_prompts]
        
        # Apply chat template to all prompts in batch
        texts = [tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) for messages in messages_list]

        # Tokenize batch
        model_inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate for batch
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode only the generated parts
        input_lengths = model_inputs.input_ids.shape[1]
        generated_texts = tokenizer.batch_decode(
            [ids[input_lengths:] for ids in generated_ids],
            skip_special_tokens=True
        )
        
        all_results.extend(generated_texts)
    
    return all_results

def create_prompt(original_text, target_emotion):
    """Create prompt for emotion transfer"""
    return f"""You are an expert text rewriter. Your task is to transform the following customer review to express {target_emotion} emotion.

Original Review: {original_text}

Please rewrite the review to clearly express {target_emotion} emotion while keeping the core facts and details intact. Output only the rewritten review, no additional text."""

def process_single_review(args):
    """Process a single review with its selected emotions"""
    text, sampled_emotions, model, tokenizer, temperature, top_p, n_samples, batch_size = args
    
    output = {'original_text': text}
    
    for emotion in sampled_emotions:
        # Create n_samples prompts for this emotion
        prompts = [create_prompt(text, emotion) for _ in range(n_samples)]
        
        # Generate all samples in batch
        results = generate_text_batch(
            prompts, 
            model, 
            tokenizer, 
            max_new_tokens=512, 
            temperature=temperature, 
            top_p=top_p,
            batch_size=batch_size
        )
        
        output[emotion] = results
    
    return output

def inference_dataset_multiple_samples(data_path, output_file, model_path, temperature, top_p, device, n_samples=10, batch_size=8, num_workers=1):
    """
    For each original text in the dataset, randomly select 2 emotions and generate n_samples for each emotion
    
    Args:
        data_path: Input data path
        output_file: Output file path
        model_path: Model path
        temperature: Temperature parameter
        top_p: top-p parameter
        n_samples: Number of samples to generate per emotion
        batch_size: Batch size for generation
        num_workers: Number of parallel workers (1 = sequential, >1 = parallel)
    """
    # Load model and tokenizer
    device_obj = torch.device(device)
    model = Qwen2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Read data
    df = pd.read_csv(data_path)
    texts = df['original_text'].tolist()

    print(f"Loaded {len(texts)} test samples")
    print(f"Using batch_size={batch_size}, num_workers={num_workers}")

    emotions = ["joy", "disappointment", "peacefulness", "anger"]
    
    # Prepare tasks: (text, sampled_emotions, ...)
    tasks = []
    for text in texts:
        # Randomly select two emotions
        random_idx = random.sample(list(range(len(emotions))), 2)
        sampled_emotions = [emotions[idx] for idx in random_idx]
        tasks.append((text, sampled_emotions, model, tokenizer, temperature, top_p, n_samples, batch_size))

    results = []
    
    if num_workers > 1:
        # Parallel processing with ThreadPoolExecutor
        # Note: ThreadPoolExecutor works well here because most time is spent in GPU operations
        # which release the GIL, allowing parallel execution
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_review, task): task for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing reviews"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing review: {e}")
                    # Add empty result to maintain order
                    task = futures[future]
                    results.append({'original_text': task[0], 'error': str(e)})
    else:
        # Sequential processing
        for task in tqdm(tasks, desc="Processing reviews"):
            result = process_single_review(task)
            results.append(result)

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f'Completed inference for {len(texts)} samples, generated {n_samples} samples per emotion')
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/research/d7/gds/yztian25/EmotionTransfer/dataset/yelp_3000.csv")
    parser.add_argument("--model_path", type=str, default="/research/d7/gds/yztian25/EmotionTransfer/models/Qwen2.5-0.5B-Instruct-sft")
    parser.add_argument("--output_file", type=str, default="/research/d7/gds/yztian25/EmotionTransfer/qwen_train/inference/outputs/emotion_conversion_multiple_samples.json")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples per emotion")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers (1=sequential, >1=parallel)")

    args = parser.parse_args()

    inference_dataset_multiple_samples(
        data_path=args.data_path, 
        output_file=args.output_file, 
        model_path=args.model_path, 
        temperature=args.temperature, 
        top_p=args.top_p,
        device=args.device,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    ) 