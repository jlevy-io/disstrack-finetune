"""
Ultra-aggressive cleaning + conversion to LLaVA format v3
Improved with expanded visual keywords and tiered scoring
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
    Improved quality check with tiered scoring and expanded visual keywords
    Returns: (is_valid, reason)
    """
    
    # Length check
    if len(roast_text) < 20:
        return False, "too_short"
    if len(roast_text) > 200:  # Increased from 150
        return False, "too_long"
    
    # TIERED SCORE CHECK (more lenient)
    if roast_score < 75:  # Lowered from 150
        return False, "low_score"
    
    # EXPANDED visual grounding keywords
    visual_keywords = [
        # Face features
        'hair', 'face', 'eye', 'eyes', 'nose', 'mouth', 'teeth', 'smile', 'smiling',
        'forehead', 'chin', 'eyebrow', 'eyebrows', 'beard', 'mustache', 'glasses', 'head',
        'ear', 'ears', 'cheek', 'cheeks', 'jaw', 'neck', 'skin', 'lips',
        
        # Body/appearance
        'wearing', 'shirt', 'outfit', 'clothes', 'look', 'looking', 'looks',
        'body', 'arm', 'arms', 'hand', 'hands', 'finger', 'fingers',
        'tall', 'short', 'fat', 'thin', 'skinny', 'big', 'small', 'ugly',
        
        # Hair descriptors
        'bald', 'balding', 'hairline', 'hairy', 'shaved', 'curly', 'greasy',
        
        # Physical comparisons (very common in roasts)
        'look like', 'looks like', 'looking like', 'looked like',
        'remind', 'reminds', 'resemble', 'resembles',
        
        # Visual descriptors
        'pretty', 'handsome', 'attractive', 'beautiful',
        'pale', 'dark', 'bright', 'color', 'colored',
        'shaped', 'round', 'square', 'long', 'wide',
        
        # Specific terms from common roasts
        'dude', 'guy', 'girl', 'man', 'woman', 'boy',
        'forehead', 'fivehead', 'receding',
        
        # Size/appearance
        'giant', 'tiny', 'huge', 'massive', 'small'
    ]
    
    text_lower = roast_text.lower()
    visual_count = sum(1 for kw in visual_keywords if kw in text_lower)
    
    # LOWERED from 2 to 1 for high-scoring roasts
    min_visual = 1 if roast_score >= 150 else 1  # Always require at least 1
    
    if visual_count < min_visual:
        return False, "not_visual"
    
    # Check for remaining artifacts
    bad_patterns = [
        r'http', r'www\.', r'\.com', r'imgur', r'reddit',
        r'edit:', r'thank', r'award',
        r'subreddit', r'upvote', r'#\w+'
    ]
    
    for pattern in bad_patterns:
        if re.search(pattern, text_lower):
            return False, f"has_artifact"
    
    # Check for non-visual personality roasts (less strict now)
    non_visual = [
        'you probably', 'i bet you', 'you must be',
        'you seem like', 'you\'re the type'
    ]
    
    # Only fail if it's low-scoring AND non-visual
    if roast_score < 100:
        if any(phrase in text_lower for phrase in non_visual):
            return False, "not_grounded"
    
    # Check for sympathy/encouragement
    sympathy = ['stay strong', 'you got this', 'good luck', 'hope you', 'prayers']
    if any(phrase in text_lower for phrase in sympathy):
        return False, "sympathy"
    
    return True, "valid"

def clean_and_convert(
    input_file: str = "data/raw/training_data.jsonl",
    output_file: str = "data/llava_format/train.json"
):
    """Clean and convert to LLaVA format with improved filtering"""
    
    print("\n" + "="*70)
    print("🧹 IMPROVED CLEANING + CONVERSION v1.0")
    print("="*70 + "\n")
    
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
            
            # Convert to LLaVA format
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
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(llava_data, f, indent=2, ensure_ascii=False)
    
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
        print(f"   {reason:20s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\n📁 Output: {output_path}")
    print(f"📊 Final dataset size: {len(llava_data)} samples")
    
    # Show samples
    print(f"\n{'='*70}")
    print(f"📋 SAMPLE CLEANED ROASTS")
    print(f"{'='*70}\n")
    
    import random
    sample_count = min(15, len(llava_data))
    for i, sample in enumerate(random.sample(llava_data, sample_count), 1):
        roast = sample['conversations'][1]['value']
        print(f"{i}. {roast}\n")
    
    print(f"{'='*70}\n")
    
    # Assessment
    if stats['kept'] < 300:
        print("⚠️  WARNING: Less than 300 clean samples!")
        print("   You may need to collect more data.")
        print(f"   Current: {stats['kept']} samples")
        print(f"   Recommended minimum: 500-1000 for decent fine-tuning")
    elif stats['kept'] < 500:
        print("⚠️  CAUTION: {stats['kept']} samples is on the low end.")
        print("   This might work but consider collecting more data.")
    else:
        print(f"✅ {stats['kept']} samples - Good dataset size!")
        print("   Ready for fine-tuning preparation.")
    
    print()

if __name__ == "__main__":
    clean_and_convert()
