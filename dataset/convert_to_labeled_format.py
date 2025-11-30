"""
Convert yelp_with_emotions.csv to text,label format.
Each original row becomes 4 rows (one for each emotion).
"""
import pandas as pd
import sys


def convert_to_labeled_format(input_file: str, output_file: str = None):
    """
    Convert emotion dataset to text,label format.
    
    Args:
        input_file: Path to input CSV file (yelp_with_emotions.csv)
        output_file: Path to output CSV file (default: dataset/yelp_labeled.csv)
    """
    if output_file is None:
        output_file = "dataset/yelp_labeled.csv"
    
    print(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    
    # Limit to 1000 rows to get exactly 4000 output rows (1000 × 4 emotions)
    max_rows = 1000
    if len(df) > max_rows:
        print(f"Limiting to first {max_rows} rows to get exactly 4000 output rows")
        df = df.head(max_rows)
    
    # Emotion columns
    emotions = ['joy', 'disappointment', 'peacefulness', 'anger']
    
    # Create new dataframe
    new_rows = []
    
    print(f"Processing {len(df)} rows...")
    for idx in df.index:
        for emotion in emotions:
            text = df.loc[idx, emotion]
            
            # Skip if text is empty, NaN, or contains instruction patterns
            if pd.isna(text) or not str(text).strip():
                continue
            
            # Skip if it looks like instruction text (starts with number and "Rewrite" etc.)
            text_str = str(text).strip()
            if text_str.startswith(('1.', '2.', '3.', '4.', '5.')) and any(
                keyword in text_str.lower() for keyword in ['rewrite', 'instruction', 'target emotion']
            ):
                continue
            
            new_rows.append({
                'text': text_str,
                'label': emotion
            })
    
    # Create new dataframe
    new_df = pd.DataFrame(new_rows)
    
    print(f"\nCreated {len(new_df)} rows")
    print(f"Saving to: {output_file}")
    new_df.to_csv(output_file, index=False)
    print("Done!")
    
    # Print label distribution
    print("\nLabel distribution:")
    print(new_df['label'].value_counts())
    
    return new_df


def main():
    """Main function to run from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert yelp_with_emotions.csv to text,label format'
    )
    parser.add_argument(
        'input_file',
        type=str,
        nargs='?',
        default='dataset/yelp_with_emotions.csv',
        help='Path to input CSV file (default: dataset/yelp_with_emotions.csv)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to output CSV file (default: dataset/yelp_labeled.csv)'
    )
    
    args = parser.parse_args()
    convert_to_labeled_format(args.input_file, args.output)


if __name__ == "__main__":
    main()

