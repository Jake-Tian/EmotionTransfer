"""
Convert combined_train_test.csv to SFT format for LLaMA-Factory.
Output format: JSONL with messages structure
"""
import pandas as pd
import json
import os


def convert_to_sft_format(
    input_csv: str = 'combined_train_test.csv',
    output_jsonl: str = 'emotion_transfer_sft.jsonl',
    create_combined: bool = True
):
    """
    Convert CSV dataset to SFT format for LLaMA-Factory.
    
    Args:
        input_csv: Path to input CSV file
        output_jsonl: Path to output JSONL file
        create_combined: If True and input_csv doesn't exist, create it from train.csv and test.csv
    """
    # Check if combined file exists, if not create it
    if not os.path.exists(input_csv) and create_combined:
        print(f"{input_csv} not found. Creating from train.csv and test.csv...")
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        combined_df.to_csv(input_csv, index=False)
        print(f"Created {input_csv} with {len(combined_df)} rows")
    
    # Read the dataset
    print(f"Reading dataset from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Total rows: {len(df)}")
    print(f"Label distribution:")
    print(df['label'].value_counts())
    print()
    
    # Convert to SFT format
    print("Converting to SFT format...")
    output_data = []
    
    for idx, row in df.iterrows():
        original_text = str(row['original_text']).strip()
        target_text = str(row['text']).strip()
        label = str(row['label']).strip()
        
        # Skip if any field is empty
        if not original_text or not target_text or not label:
            continue
        
        # Create user prompt
        user_prompt = f"""You are an expert text rewriter. Your task is to transform the following customer review to express {label} emotion.

Original Review: {original_text}

Please rewrite the review to clearly express {label} emotion while keeping the core facts and details intact. Output only the rewritten review, no additional text."""
        
        # Create messages format
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": target_text}
        ]
        
        output_data.append({"messages": messages})
    
    # Write to JSONL file
    output_dir = os.path.dirname(output_jsonl)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Writing {len(output_data)} samples to: {output_jsonl}")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for data in output_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"✓ Successfully converted {len(output_data)} rows to SFT format")
    print(f"  Output file: {output_jsonl}")
    
    # Show sample
    print("\nSample output (first entry):")
    print(json.dumps(output_data[0], indent=2, ensure_ascii=False))
    
    return output_jsonl


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert combined_train_test.csv to SFT format for LLaMA-Factory'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='combined_train_test.csv',
        help='Path to input CSV file (default: combined_train_test.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='emotion_transfer_sft.jsonl',
        help='Path to output JSONL file (default: emotion_transfer_sft.jsonl)'
    )
    parser.add_argument(
        '--no-combine',
        action='store_true',
        help='Do not create combined file if input does not exist'
    )
    
    args = parser.parse_args()
    convert_to_sft_format(
        input_csv=args.input,
        output_jsonl=args.output,
        create_combined=not args.no_combine
    )


if __name__ == "__main__":
    main()

