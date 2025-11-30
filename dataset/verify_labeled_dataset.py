"""
Verify that every row in yelp_labeled.csv is valid.
"""
import pandas as pd
import re


def has_chinese(text):
    """Check if text contains Chinese characters."""
    if pd.isna(text):
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))


def has_emoji(text):
    """Check if text contains emoji."""
    if pd.isna(text):
        return False
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "]+",
        flags=re.UNICODE
    )
    return bool(emoji_pattern.search(str(text)))


def has_instruction_text(text):
    """Check if text contains instruction patterns."""
    if pd.isna(text):
        return False
    text_str = str(text)
    instruction_patterns = [
        r'^\d+\.\s*(Rewrite|Read|Keep|Do not|Ensure|The length|Do NOT|Output)',
        r'^Rewrite the review',
        r'^Maintain the core facts',
        r'^Do not exaggerate',
        r'^Target Emotion',
        r'^Original Review',
        r'^Output.*JSON',
        r'^Instructions?:',
    ]
    return any(re.search(pattern, text_str, re.IGNORECASE | re.MULTILINE) 
               for pattern in instruction_patterns)


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
    ]
    return any(re.search(pattern, text_str) for pattern in checkbox_patterns)


def is_incomplete_sentence(text):
    """Check if text ends with incomplete sentence."""
    if pd.isna(text) or not text.strip():
        return True
    text_str = str(text).strip()
    # Check if ends with incomplete patterns
    if re.search(r'\s+(will|but|and|the|is|are|was|were|to|for|with)\s*$', text_str, re.IGNORECASE):
        return True
    return False


def has_json_artifacts(text):
    """Check if text contains JSON structure artifacts."""
    if pd.isna(text):
        return False
    text_str = str(text)
    json_patterns = [
        r'\{"rewritten_review"',
        r'"rewritten_review"\s*:',
        r'```\s*json',
        r'```\s*```json',
    ]
    return any(re.search(pattern, text_str, re.IGNORECASE) for pattern in json_patterns)


def has_special_characters_only(text):
    """Check if text contains only special characters or formatting."""
    if pd.isna(text):
        return False
    text_str = str(text).strip()
    # Check if text is mostly special characters
    if len(text_str) > 0:
        alphanumeric_count = len(re.findall(r'[a-zA-Z0-9]', text_str))
        if alphanumeric_count < len(text_str) * 0.3:  # Less than 30% alphanumeric
            return True
    return False


def is_identical_to_original(text, original_texts):
    """Check if text is identical to any original text (should be rewritten)."""
    if pd.isna(text):
        return False
    text_str = str(text).strip()
    return text_str in [str(orig).strip() for orig in original_texts if not pd.isna(orig)]


def verify_dataset(input_file: str):
    """Verify all rows in the labeled dataset."""
    print(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Check required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        print("ERROR: Missing required columns 'text' or 'label'")
        return
    
    # Valid labels
    valid_labels = ['joy', 'disappointment', 'peacefulness', 'anger']
    
    # Track issues
    issues = {
        'empty_text': [],
        'invalid_label': [],
        'chinese': [],
        'emoji': [],
        'instruction_text': [],
        'checkbox_validation': [],
        'incomplete_sentence': [],
        'too_short': [],
        'json_artifacts': [],
        'special_characters_only': [],
        'exact_duplicates': [],
        'duplicate_text_different_labels': [],
    }
    
    # Get original texts if available (for checking if rewritten text is identical)
    original_texts = []
    if 'original_text' in df.columns:
        original_texts = df['original_text'].tolist()
    
    print("Verifying all rows...")
    for idx in df.index:
        text = df.loc[idx, 'text']
        label = df.loc[idx, 'label']
        
        # Check empty text
        if pd.isna(text) or not str(text).strip():
            issues['empty_text'].append(idx)
            continue
        
        text_str = str(text).strip()
        
        # Check text length
        if len(text_str) < 10:
            issues['too_short'].append(idx)
        
        # Check label validity
        if label not in valid_labels:
            issues['invalid_label'].append(idx)
        
        # Check for Chinese
        if has_chinese(text_str):
            issues['chinese'].append(idx)
        
        # Check for emoji
        if has_emoji(text_str):
            issues['emoji'].append(idx)
        
        # Check for instruction text
        if has_instruction_text(text_str):
            issues['instruction_text'].append(idx)
        
        # Check for checkbox/validation
        if has_checkbox_or_validation(text_str):
            issues['checkbox_validation'].append(idx)
        
        # Check for incomplete sentences
        if is_incomplete_sentence(text_str):
            issues['incomplete_sentence'].append(idx)
        
        # Check for JSON artifacts
        if has_json_artifacts(text_str):
            issues['json_artifacts'].append(idx)
        
        # Check for special characters only
        if has_special_characters_only(text_str):
            issues['special_characters_only'].append(idx)
    
    # Check for duplicate rows (exact duplicates)
    exact_duplicates = df[df.duplicated(subset=['text', 'label'], keep=False)]
    if len(exact_duplicates) > 0:
        # Get all indices of duplicates (including first occurrence)
        duplicate_groups = exact_duplicates.groupby(['text', 'label'])
        for (text, label), group in duplicate_groups:
            # Add all but first occurrence to issues
            indices = group.index.tolist()
            issues['exact_duplicates'].extend(indices[1:])  # Keep first, mark rest as duplicates
    
    # Check for duplicate text with different labels (might indicate issues)
    text_duplicates = df[df.duplicated(subset=['text'], keep=False)]
    if len(text_duplicates) > 0:
        # Group by text and check if same text has multiple labels
        for text, group in text_duplicates.groupby('text'):
            unique_labels = group['label'].unique()
            if len(unique_labels) > 1:
                # Same text with different labels - might be valid but worth flagging
                issues['duplicate_text_different_labels'].extend(group.index.tolist())
    
    # Print summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    total_issues = 0
    issue_counts = {}
    for issue_type, rows in issues.items():
        if rows:
            unique_rows = list(set(rows))  # Remove duplicates if a row has multiple issues
            issue_counts[issue_type] = len(unique_rows)
            total_issues += len(unique_rows)
            print(f"\n{issue_type.replace('_', ' ').title()}: {len(unique_rows)} rows")
            if len(unique_rows) <= 10:
                print(f"  Row indices: {unique_rows}")
            else:
                print(f"  Row indices (first 10): {unique_rows[:10]}...")
                print(f"  Row indices (last 10): ...{unique_rows[-10:]}")
    
    # Calculate unique invalid rows (rows that have at least one issue)
    all_invalid_indices = set()
    for rows in issues.values():
        all_invalid_indices.update(rows)
    
    if total_issues == 0:
        print("\n✓ All rows are valid!")
    else:
        print(f"\n✗ Found {total_issues} issues across {len(df)} rows")
        print(f"  Valid rows: {len(df) - len(all_invalid_indices)}")
        print(f"  Invalid rows: {len(all_invalid_indices)}")
        print(f"\nIssue breakdown:")
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {issue_type.replace('_', ' ').title()}: {count}")
    
    # Label distribution
    print("\n" + "=" * 80)
    print("LABEL DISTRIBUTION")
    print("=" * 80)
    print(df['label'].value_counts())
    
    return issues


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verify yelp_labeled.csv dataset'
    )
    parser.add_argument(
        'input_file',
        type=str,
        nargs='?',
        default='dataset/yelp_labeled.csv',
        help='Path to input CSV file (default: dataset/yelp_labeled.csv)'
    )
    
    args = parser.parse_args()
    verify_dataset(args.input_file)


if __name__ == "__main__":
    main()

