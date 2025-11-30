"""
Remove specific invalid rows from the labeled dataset.
"""
import pandas as pd


def remove_invalid_rows(input_file: str, output_file: str = None):
    """
    Remove invalid rows from the dataset.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (default: overwrites input)
    """
    if output_file is None:
        output_file = input_file
    
    print(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Original dataset: {len(df)} rows")
    print(f"Label distribution:")
    print(df['label'].value_counts())
    print()
    
    # Invalid row indices to remove
    invalid_indices = [
        # Emoji rows
        289, 886, 2057, 2989,
        # Incomplete sentence rows
        838, 1574, 2044, 2258, 2591,
        # Too short rows
        264, 530, 828, 1859, 2471, 2473, 2553, 2554, 2968
    ]
    
    print(f"Removing {len(invalid_indices)} invalid rows")
    print(f"Indices: {invalid_indices}")
    
    # Check which indices actually exist
    existing_indices = [idx for idx in invalid_indices if idx in df.index]
    missing_indices = [idx for idx in invalid_indices if idx not in df.index]
    
    if missing_indices:
        print(f"Warning: {len(missing_indices)} indices not found in dataset: {missing_indices}")
    
    # Remove the rows
    df_cleaned = df.drop(existing_indices)
    
    print(f"\nAfter removing invalid rows: {len(df_cleaned)} rows")
    print("Label distribution:")
    print(df_cleaned['label'].value_counts())
    
    print(f"\nSaving to: {output_file}")
    df_cleaned.to_csv(output_file, index=False)
    print("Done!")
    
    return df_cleaned


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Remove invalid rows from labeled dataset'
    )
    parser.add_argument(
        'input_file',
        type=str,
        nargs='?',
        default='dataset/yelp_labeled.csv',
        help='Path to input CSV file (default: dataset/yelp_labeled.csv)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to output CSV file (default: overwrites input)'
    )
    
    args = parser.parse_args()
    remove_invalid_rows(args.input_file, args.output)


if __name__ == "__main__":
    main()

