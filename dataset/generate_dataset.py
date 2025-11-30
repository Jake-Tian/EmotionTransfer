from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pandas as pd
import torch
import re
from tqdm import tqdm

# Load Qwen2.5-14B-Instruct model and tokenizer
model_name = "models/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set pad_token if not already set (Qwen models may not have one by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
model.eval()  # Set to evaluation mode


def has_checkbox_or_validation(text):
    """Check if text contains checkbox markers or validation text."""
    if pd.isna(text):
        return False
    text_str = str(text).lower()
    checkbox_patterns = [
        r'-\s*\[[xX\s]\]',
        r'\[[xX\s]\]',
        r'the rewritten review is in',
        r'the length of the rewritten review',
        r'output expresses',
        r'core facts.*preserved',
        r'fluent.*natural.*coherent',
        r'no.*emoji',
        r'english only',
        r'does not contain.*chinese'
    ]
    return any(re.search(pattern, text_str) for pattern in checkbox_patterns)


def extract_clean_review(generated_text: str, prompt: str, full_output, input_length: int, tokenizer) -> str:
    """
    Extract a clean English review from generated text.
    First tries to parse JSON, then falls back to text extraction.
    """
    if not generated_text:
        # Fallback: decode full output
        full_text = tokenizer.decode(full_output, skip_special_tokens=True)
        if full_text.startswith(prompt):
            generated_text = full_text[len(prompt):].strip()
        else:
            generated_text = full_text.strip()
    
    # First, try to parse JSON
    try:
        # Find JSON object in the text (look for { ... })
        start_idx = generated_text.find('{')
        end_idx = generated_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = generated_text[start_idx:end_idx]
            result = json.loads(json_str)
            rewritten_review = result.get("rewritten_review", "").strip()
            
            # Validate the extracted review
            if rewritten_review and len(rewritten_review) > 10:
                # Check for Chinese or other invalid content
                if not re.search(r'[\u4e00-\u9fff]', rewritten_review):
                    if not has_checkbox_or_validation(rewritten_review):
                        return rewritten_review
    except (json.JSONDecodeError, KeyError, AttributeError):
        # JSON parsing failed, fall through to text extraction
        pass
    
    # Fallback: Extract from text (existing logic)
    # Split into lines
    lines = [line.strip() for line in generated_text.split("\n") if line.strip()]
    
    # Filter out invalid lines
    valid_lines = []
    for line in lines:
        # Skip lines with Chinese
        if re.search(r'[\u4e00-\u9fff]', line):
            continue
        # Skip checkbox/validation lines
        if has_checkbox_or_validation(line):
            continue
        # Skip instruction-like lines
        if any(keyword in line.lower() for keyword in [
            'task', 'instruction', 'requirement', '说明', '任务', '指令',
            'target emotion', 'original review', 'rewritten review'
        ]):
            continue
        # Skip JSON structure lines (if JSON parsing failed)
        if line.strip().startswith('{') or line.strip().startswith('}') or '"rewritten_review"' in line:
            continue
        # Skip very short lines (likely artifacts)
        if len(line.split()) < 3:
            continue
        valid_lines.append(line)
    
    # If we have valid lines, use the longest one (likely the actual review)
    if valid_lines:
        # Prefer lines that look like complete sentences
        complete_lines = [line for line in valid_lines 
                         if line.endswith('.') or line.endswith('!') or line.endswith('?')]
        if complete_lines:
            rewritten_review = max(complete_lines, key=len)
        else:
            rewritten_review = max(valid_lines, key=len)
    else:
        # No valid lines found, try to clean the full text
        cleaned = re.sub(r'[\u4e00-\u9fff]+', '', generated_text)
        cleaned = re.sub(r'-\s*\[[xX\s]\][^,]*[,]?', '', cleaned)
        cleaned = re.sub(r'\[[xX\s]\][^,]*[,]?', '', cleaned)
        cleaned = re.sub(r'(任务|说明|指令|target emotion|original review|rewritten review).*?:', '', cleaned, flags=re.IGNORECASE)
        # Remove JSON structure if present
        cleaned = re.sub(r'\{[^}]*"rewritten_review"[^}]*\}', '', cleaned, flags=re.DOTALL)
        cleaned_lines = [l.strip() for l in cleaned.split("\n") if l.strip() and len(l.strip()) > 10]
        if cleaned_lines:
            rewritten_review = max(cleaned_lines, key=len)
        else:
            rewritten_review = cleaned.strip()[:500]
    
    # Final cleanup: remove trailing incomplete words/sentences
    rewritten_review = rewritten_review.strip()
    rewritten_review = re.sub(r'\s+(will|but|and|the|is|are|was|were)\s*$', '', rewritten_review, flags=re.IGNORECASE)
    if rewritten_review and not rewritten_review[-1] in '.!?"':
        sentences = re.split(r'[.!?]', rewritten_review)
        if len(sentences) > 1:
            rewritten_review = '. '.join(sentences[:-1]).strip()
            if rewritten_review and not rewritten_review[-1] in '.!?':
                rewritten_review += '.'
    
    return rewritten_review if rewritten_review else generated_text[:500]


def create_prompt(original_review: str, target_emotion: str) -> str:
    """Create a prompt for emotion transfer."""
    return f"""You are an expert text rewriter. Your task is to transform the following customer review into the style of {target_emotion}.

Instructions:
1. Read the original review carefully.
2. Rewrite the review so that it clearly expresses the TARGET EMOTION.
3. Keep the core facts, details, and context intact (e.g., food quality, service speed, atmosphere).
4. Do not exaggerate, invent new details, or remove important information.
5. Ensure the rewritten text is fluent, natural, and coherent.
6. The length of the rewritten review should be similar to the original review.
7. Do NOT generate any emoji.
8. The rewritten review MUST be in English only (same language as original).

Target Emotion: {target_emotion}
Original Review: {original_review}

Output your response as a JSON object with the following format:
{{
    "rewritten_review": "your rewritten review here in English"
}}

Output only the JSON object, no additional text.
"""


def generate_batch(batch_prompts: list, max_new_tokens: int = 200):
    """
    Generate text for a batch of prompts efficiently.
    
    Args:
        batch_prompts: List of prompt strings (already batched)
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        List of generated texts
    """
    # Tokenize batch
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)
    
    # Generate with batching
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Extract only the generated part (remove prompt)
    rewritten_reviews = []
    input_ids = inputs['input_ids']
    
    for idx in range(len(batch_prompts)):
        # Get the actual input length for this prompt (excluding padding)
        input_length = (input_ids[idx] != tokenizer.pad_token_id).sum().item()
        
        # Extract only the generated tokens (skip input tokens)
        generated_ids = outputs[idx][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Use improved parsing to extract clean review
        rewritten_review = extract_clean_review(
            generated_text, 
            batch_prompts[idx], 
            outputs[idx], 
            input_length,
            tokenizer
        )
        
        rewritten_reviews.append(rewritten_review)
    
    return rewritten_reviews


def generate_dataset(dataset, batch_size: int = 20, max_new_tokens: int = 200):
    """
    Generate emotion-transferred reviews for the entire dataset using batching.
    
    Args:
        dataset: DataFrame with 'original_text' column
        batch_size: Number of reviews to process in parallel (adjust based on GPU memory)
        max_new_tokens: Maximum tokens to generate per review
    """
    emotions = ["joy", "disappointment", "peacefulness", "anger"]
    row_num = len(dataset)
    
    # Create columns for each emotion
    for emotion in emotions:
        dataset[emotion] = ""
    
    # Process each emotion
    for emotion in emotions:
        print(f"\nProcessing emotion: {emotion}")
        
        # Create all prompts for this emotion
        prompts = []
        for i in range(row_num):
            original_review = dataset.loc[i, "original_text"]
            prompt = create_prompt(original_review, emotion)
            prompts.append(prompt)
        
        # Generate in batches
        rewritten_reviews = []
        total_processed = 0
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating {emotion}"):
            batch_prompts = prompts[i:i + batch_size]
            # Remove newlines for cleaner processing
            batch_prompts_clean = [prompt.replace("\n", " ") for prompt in batch_prompts]
            batch_outputs = generate_batch(batch_prompts_clean, max_new_tokens=max_new_tokens)
            
            # Verify we got the right number of outputs
            assert len(batch_outputs) == len(batch_prompts), f"Mismatch: {len(batch_outputs)} outputs for {len(batch_prompts)} prompts"
            
            rewritten_reviews.extend(batch_outputs)
            
            # Indicator every 50 rows
            total_processed += len(batch_outputs)
            if total_processed % 50 == 0:
                print(f"\n[{emotion}] Generated {total_processed}/{row_num} rows")
        
        # Verify we have the correct number of results
        assert len(rewritten_reviews) == row_num, f"Mismatch: {len(rewritten_reviews)} results for {row_num} rows"
        
        # Assign results to dataset
        for i, rewritten_review in enumerate(rewritten_reviews):
            dataset.loc[i, emotion] = rewritten_review
    
    # Save results
    dataset.to_csv("dataset/yelp_with_emotions.csv", index=False)
    print("\nDataset saved to dataset/yelp_with_emotions.csv")
    print(dataset.head())


def main():
    dataset = pd.read_csv("dataset/yelp_1000.csv")
    # Adjust batch_size based on your GPU memory (larger = faster but uses more memory)
    generate_dataset(dataset, batch_size=20, max_new_tokens=200)


if __name__ == "__main__":
    main()