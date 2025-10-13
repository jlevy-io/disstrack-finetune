"""
Ultra-aggressive cleaning + conversion to LLaVA format v4
Strict visual grounding focus + system prompts + train/val split
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Optional, Tuple

def ultra_clean_roast(roast_text: str) -> Optional[str]:
    """
    Ultra-aggressive cleaning - return None if can't be salvaged
    Now with better artifact removal
    """
    
    original = roast_text
    
    # 1. Remove Edit/Award acknowledgments (very common)
    roast_text = re.sub(r'Edit:.*$', '', roast_text, flags=re.IGNORECASE | re.MULTILINE)
    roast_text = re.sub(r'EDIT:.*$', '', roast_text, flags=re.IGNORECASE | re.MULTILINE)
    roast_text = re.sub(r'Thanks? for the (gold|silver|platinum|award).*$', '', roast_text, flags=re.IGNORECASE | re.MULTILINE)
    roast_text = re.sub(r'Thank you (kind stranger|for the gold).*$', '', roast_text, flags=re.IGNORECASE)
    
    # 2. Remove GIF/image references
    roast_text = re.sub(r'!\[gif\]\(.*?\)', '', roast_text)
    roast_text = re.sub(r'!\[.*?\]\(.*?\)', '', roast_text)
    roast_text = re.sub(r'\[gif\]\(.*?\)', '', roast_text)
    
    # 3. Remove ALL markdown/links
    roast_text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', roast_text)  # [text](url) -> text
    roast_text = re.sub(r'https?://\S+', '', roast_text)  # Remove URLs
    
    # 4. Remove Reddit references
    roast_text = re.sub(r'r/\w+', '', roast_text)
    roast_text = re.sub(r'u/\w+', '', roast_text)
    
    # 5. Remove emojis (ALL of them)
    roast_text = re.sub(r'[\U00010000-\U0010ffff]', '', roast_text, flags=re.UNICODE)
    roast_text = re.sub(r'[\u2600-\u27BF]', '', roast_text)
    roast_text = re.sub(r'[\U0001F300-\U0001F9FF]', '', roast_text)
    
    # 6. Remove hashtags
    roast_text = re.sub(r'#\w+', '', roast_text)
    
    # 7. Remove markdown formatting
    roast_text = re.sub(r'[*_]{1,}', '', roast_text)
    roast_text = re.sub(r'#{1,6}\s', '', roast_text)
    
    # 8. Clean up brackets/parentheses
    roast_text = re.sub(r'\[\s*\]', '', roast_text)
    roast_text = re.sub(r'\(\s*\)', '', roast_text)
    
    # 9. Remove excessive punctuation
    roast_text = re.sub(r'([!?]){3,}', r'\1\1', roast_text)  # !!! -> !!
    roast_text = re.sub(r'\.{3,}', '...', roast_text)
    
    # 10. Clean whitespace
    roast_text = re.sub(r'\n+', ' ', roast_text)  # Remove ALL newlines
    roast_text = re.sub(r'\s+', ' ', roast_text)  # Collapse multiple spaces
    roast_text = roast_text.strip()
    
    # 11. Remove trailing artifacts
    roast_text = re.sub(r'\.\.\.\s*$', '', roast_text)
    roast_text = roast_text.strip()
    
    # 12. Final cleanup - remove if too much was removed
    if not roast_text or len(roast_text) < 20:
        return None
    
    # Check if we removed too much (sign of heavy artifacts)
    if len(roast_text) < len(original) * 0.3:
        return None
    
    return roast_text

def is_high_quality(roast_text: str, roast_score: int) -> Tuple[bool, str]:
    """
    STRICT quality check focused on visual grounding
    Returns: (is_valid, reason)
    """
    text_lower = roast_text.lower()
    
    # 1. Length check - STRICT
    if len(roast_text) < 25:
        return False, "too_short"
    if len(roast_text) > 150:
        return False, "too_long"
    
    # 2. Score check - HIGHER THRESHOLD
    if roast_score < 100:
        return False, "low_score"
    
    # 3. AUTO-REJECT patterns (non-visual content)
    reject_patterns = [
        r'\bkarma\b', r'\bsubscriber[s]?\b', r'\bharvard\b', r'\bcollege\b',
        r'\bi bet you\b', r'\byou probably\b', r'\bi heard\b',
        r'\bglad\b', r'\bcongrat[s]?\b', r'\bgood job\b', r'\bnice\b',
        r'\bperfectly valid\b', r'\bwe care\b',
        r'\bthis time\b', r'\bupdated\b', r'\bagain\b', r'\bstill\b',
        r'\bvirgin\b', r'\bgirlfriend\b', r'\bboyfriend\b',
        r'\bjob\b', r'\bmoney\b', r'\brich\b', r'\bpoor\b'
    ]
    
    for pattern in reject_patterns:
        if re.search(pattern, text_lower):
            return False, "non_visual_content"
    
    # 4. MUST have visual comparison OR multiple physical features
    visual_comparisons = [
        r'\blook like\b', r'\blooks like\b', r'\blooking like\b',
        r'\bremind[s]? me of\b', r'\bresemble[s]?\b',
        r'\bif .+ had a baby\b', r'\bif .+ and .+ had\b',
        r'\bknockoff\b', r'\bdollar store\b', r'\bwish\.com\b'
    ]
    
    has_comparison = any(re.search(pattern, text_lower) for pattern in visual_comparisons)
    
    # Physical features
    physical_features = [
        'face', 'forehead', 'fivehead', 'hair', 'hairline', 'receding',
        'eye', 'eyes', 'nose', 'mouth', 'teeth', 'smile', 'smiling',
        'chin', 'eyebrow', 'eyebrows', 'beard', 'mustache',
        'head', 'neck', 'cheek', 'cheeks', 'jaw', 'lips',
        'body', 'arm', 'arms', 'hand', 'hands'
    ]
    
    feature_count = sum(1 for f in physical_features if f in text_lower)
    
    # Need EITHER comparison OR 2+ features
    if not has_comparison and feature_count < 2:
        return False, "not_visual_enough"
    
    # 5. Check for remaining artifacts
    bad_patterns = [
        r'http', r'www\.', r'\.com', r'imgur', r'reddit',
        r'subreddit', r'upvote'
    ]
    
    for pattern in bad_patterns:
        if re.search(pattern, text_lower):
            return False, "has_artifact"
    
    return True, "valid"

def clean_and_convert(
    input_file: str = "data/raw/training_data.jsonl",
    output_file: str = "data/llava_format/train.json"
):
    """Clean and convert to LLaVA format with strict visual filtering"""
    
    print("\n" + "="*70)
    print("🧹 STRICT VISUAL FILTERING + CONVERSION v4.0")
    print("="*70 + "\n")
    
    # System prompt (matches production)
    SYSTEM_PROMPT = """You are a sharp-witted roast comedian performing in a consensual roast battle inspired by Reddit's r/RoastMe. Generate one savage, hilarious roast based ONLY on what you see in the photo.

Rules:
- Attack specific visual details: appearance, expression, outfit, setting
- Use clever comparisons and pop culture references  
- Keep it punchy: 1-2 sentences, under 100 characters ideal
- Be brutally funny - no holding back
- No hedging, no apologies, just pure roast

Deliver a devastating one-liner like a standup comedian."""
    
    # Load raw data
    samples = []
    with open(input_file) as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Raw samples: {len(samples)}")
    total_raw_roasts = sum(len(s['roasts']) for s in samples)
    print(f"Raw roasts: {total_raw_roasts}\n")
    
    # Clean and filter
    llava_data = []
    stats = Counter()
    
    for sample in samples:
        for roast_idx, (roast_text, roast_score) in enumerate(zip(sample['roasts'], sample['roast_scores'])):
            stats['total'] += 1
            
            # Clean
            cleaned = ultra_clean_roast(roast_text)
            
            if not cleaned:
                stats['cleaning_failed'] += 1
                continue
            
            # Quality check
            is_valid, reason = is_high_quality(cleaned, roast_score)
            
            if not is_valid:
                stats[reason] += 1
                continue
            
            # Convert to LLaVA format with system prompt
            llava_data.append({
                "id": f"{sample['id']}_r{roast_idx}",
                "image": sample['image_filename'],
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nRoast this person based on their appearance."
                    },
                    {
                        "from": "gpt",
                        "value": cleaned
                    }
                ]
            })
            
            stats['kept'] += 1
    
    # Split train/val
    import random
    random.seed(42)
    random.shuffle(llava_data)
    
    val_ratio = 0.1
    split_idx = int(len(llava_data) * (1 - val_ratio))
    train_data = llava_data[:split_idx]
    val_data = llava_data[split_idx:]
    
    # Save train
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    # Save val
    val_path = output_path.parent / "val.json"
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    # Stats
    print(f"\n{'='*70}")
    print(f"📊 CLEANING RESULTS")
    print(f"{'='*70}\n")
    
    survival_rate = (stats['kept']/stats['total']*100) if stats['total'] > 0 else 0
    print(f"✅ Kept: {stats['kept']}/{stats['total']} ({survival_rate:.1f}%)\n")
    
    print(f"❌ Removed breakdown:")
    removal_reasons = [(reason, count) for reason, count in stats.items() 
                       if reason not in ['total', 'kept']]
    removal_reasons.sort(key=lambda x: x[1], reverse=True)
    
    for reason, count in removal_reasons:
        pct = count/stats['total']*100
        print(f"   {reason:25s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\n📊 Split complete:")
    print(f"   Train: {len(train_data)} samples → {output_path}")
    print(f"   Val:   {len(val_data)} samples → {val_path}")
    
    # Show samples
    print(f"\n{'='*70}")
    print(f"📋 SAMPLE CLEANED ROASTS")
    print(f"{'='*70}\n")
    
    sample_count = min(15, len(train_data))
    for i, sample in enumerate(random.sample(train_data, sample_count), 1):
        roast = sample['conversations'][2]['value']
        print(f"{i}. {roast}\n")
    
    print(f"{'='*70}\n")
    
    # Assessment
    if stats['kept'] < 500:
        print("⚠️  WARNING: Less than 500 clean samples!")
        print("   You may need to:")
        print(f"   - Lower min_score threshold (currently 100)")
        print(f"   - Collect more data")
        print(f"   Current: {stats['kept']} samples")
    elif stats['kept'] < 800:
        print(f"⚠️  {stats['kept']} samples - Workable but could be better")
        print("   Consider collecting more data for optimal results")
    else:
        print(f"✅ {stats['kept']} samples - Good dataset size!")
        print("   Ready for training!")
    
    print()

if __name__ == "__main__":
    clean_and_convert()
