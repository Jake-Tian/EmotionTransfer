"""
Fix JSON artifacts by extracting only the review text.
"""
import pandas as pd
import re
import json


def extract_review_from_json_artifact(text):
    """
    Extract review text from JSON artifacts.
    Returns the extracted text, or original if no JSON found.
    """
    if pd.isna(text):
        return text
    
    text_str = str(text).strip()
    
    # Try to parse as complete JSON first
    try:
        # Find JSON object
        start_idx = text_str.find('{')
        end_idx = text_str.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = text_str[start_idx:end_idx]
            result = json.loads(json_str)
            review = result.get("rewritten_review", "").strip()
            if review:
                return review
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass
    
    # Try to extract from "rewritten_review": "text" pattern
    # Pattern 1: "rewritten_review": "text here"
    pattern1 = r'"rewritten_review"\s*:\s*"([^"]+)"'
    matches = re.findall(pattern1, text_str, re.DOTALL)
    if matches:
        # Get the longest match (likely the actual review)
        review = max(matches, key=len).strip()
        if review:
            return review
    
    # Pattern 2: "rewritten_review": "text here (may have escaped quotes)
    pattern2 = r'"rewritten_review"\s*:\s*"((?:[^"\\]|\\.)*)"'
    matches = re.findall(pattern2, text_str, re.DOTALL)
    if matches:
        review = max(matches, key=len).strip()
        if review:
            # Unescape common escape sequences
            review = review.replace('\\n', ' ').replace('\\t', ' ').replace('\\"', '"')
            return review
    
    # Pattern 3: Look for content after "rewritten_review": that's not JSON structure
    # Find the position after "rewritten_review":
    match = re.search(r'"rewritten_review"\s*:\s*"', text_str, re.IGNORECASE)
    if match:
        start_pos = match.end()
        # Find the closing quote or end of string
        # Look for the next unescaped quote
        remaining = text_str[start_pos:]
        # Try to find the closing quote
        quote_pos = -1
        i = 0
        while i < len(remaining):
            if remaining[i] == '"' and (i == 0 or remaining[i-1] != '\\'):
                quote_pos = i
                break
            i += 1
        
        if quote_pos > 0:
            review = remaining[:quote_pos].strip()
            if review:
                review = review.replace('\\n', ' ').replace('\\t', ' ').replace('\\"', '"')
                return review
        else:
            # No closing quote found, take everything after the colon
            review = remaining.strip()
            if review and len(review) > 5:
                review = review.replace('\\n', ' ').replace('\\t', ' ').replace('\\"', '"')
                # Remove trailing JSON structure if present
                review = re.sub(r'[}\],\s]*$', '', review)
                return review
    
    # Pattern 4: If text starts with "rewritten_review":, extract what follows
    if text_str.lower().startswith('"rewritten_review"'):
        # Remove the key part
        cleaned = re.sub(r'^"rewritten_review"\s*:\s*', '', text_str, flags=re.IGNORECASE)
        # Remove quotes if present
        cleaned = re.sub(r'^["\']|["\']$', '', cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            cleaned = cleaned.replace('\\n', ' ').replace('\\t', ' ').replace('\\"', '"')
            return cleaned
    
    # If no pattern matched, try to clean up common JSON artifacts
    # Remove JSON structure markers
    cleaned = text_str
    cleaned = re.sub(r'^.*?"rewritten_review"\s*:\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^.*?\{[^}]*"rewritten_review"\s*:\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'```\s*json\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'```\s*```json\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Rewritten Review:\s*', '', cleaned, flags=re.IGNORECASE)
    # Remove leading/trailing quotes and braces
    cleaned = re.sub(r'^["\'\{\[\s]+|["\'\}\]]+\s*$', '', cleaned)
    cleaned = cleaned.strip()
    
    # If we got something meaningful, return it
    if cleaned and len(cleaned) > 5:
        cleaned = cleaned.replace('\\n', ' ').replace('\\t', ' ').replace('\\"', '"')
        return cleaned
    
    # If nothing worked, return original
    return text_str


def fix_json_artifacts(input_file: str, output_file: str = None):
    """
    Fix JSON artifacts in the dataset by extracting review text.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (default: overwrites input)
    """
    if output_file is None:
        output_file = input_file
    
    print(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Original dataset: {len(df)} rows")
    print()
    
    # Check for JSON artifacts
    json_patterns = [
        r'\{"rewritten_review"',
        r'"rewritten_review"\s*:',
        r'```\s*json',
        r'```\s*```json',
    ]
    
    rows_fixed = 0
    for idx in df.index:
        text = df.loc[idx, 'text']
        if pd.isna(text):
            continue
        
        text_str = str(text)
        has_json = any(re.search(pattern, text_str, re.IGNORECASE) for pattern in json_patterns)
        
        if has_json:
            extracted = extract_review_from_json_artifact(text_str)
            if extracted != text_str:
                df.loc[idx, 'text'] = extracted
                rows_fixed += 1
    
    print(f"Fixed {rows_fixed} rows with JSON artifacts")
    print()
    print("Label distribution:")
    print(df['label'].value_counts())
    
    print(f"\nSaving to: {output_file}")
    df.to_csv(output_file, index=False)
    print("Done!")
    
    return df


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fix JSON artifacts by extracting review text'
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
    fix_json_artifacts(args.input_file, args.output)


if __name__ == "__main__":
    main()

