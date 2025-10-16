"""
tools/filter_hf_with_visual_criteria.py

Apply the SAME aggressive filtering to HF dataset
that we used on our original data (minus score requirement)
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Optional, Tuple

def ultra_clean_roast(roast_text: str) -> Optional[str]:
    """
    EXACT COPY from clean_and_convert.py
    Ultra-aggressive cleaning - return None if can't be salvaged
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
    roast_text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', roast_text)
    roast_text = re.sub(r'https?://\S+', '', roast_text)
    
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
    roast_text = re.sub(r'([!?]){3,}', r'\1\1', roast_text)
    roast_text = re.sub(r'\.{3,}', '...', roast_text)
    
    # 10. Clean whitespace
    roast_text = re.sub(r'\n+', ' ', roast_text)
    roast_text = re.sub(r'\s+', ' ', roast_text)
    roast_text = roast_text.strip()
    
    # 11. Remove trailing artifacts
    roast_text = re.sub(r'\.\.\.\s*$', '', roast_text)
    roast_text = roast_text.strip()
    
    # 12. Final cleanup - remove if too much was removed
    if not roast_text or len(roast_text) < 20:
        return None
    
    if len(roast_text) < len(original) * 0.3:
        return None
    
    return roast_text

def is_high_quality(roast_text: str) -> Tuple[bool, str]:
    """
    EXACT COPY from clean_and_convert.py (MINUS score check)
    STRICT quality check focused on visual grounding
    Returns: (is_valid, reason)
    """
    text_lower = roast_text.lower()
    
    # 1. Length check - STRICT
    if len(roast_text) < 25:
        return False, "too_short"
    if len(roast_text) > 150:
        return False, "too_long"
    
    # 2. Score check - SKIPPED (HF dataset has no scores)
    # if roast_score < 100:
    #     return False, "low_score"
    
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

def filter_hf_dataset():
    """Apply SAME filtering as original data (minus score requirement)"""
    
    print("\n" + "="*70)
    print("🧹 APPLYING YOUR VISUAL FILTERING TO HF DATASET")
    print("="*70 + "\n")
    print("Using EXACT same criteria from clean_and_convert.py")
    print("(minus upvote score requirement - not available in HF data)\n")
    
    # Load HF dataset
    hf_file = Path("data/huggingface/filtered_roasts.json")
    
    if not hf_file.exists():
        print(f"❌ File not found: {hf_file}")
        print("Run tools/analyze_reddit_roastme_hf.py first!")
        return
    
    with open(hf_file) as f:
        data = json.load(f)
    
    raw_roasts = data['roasts']
    print(f"Raw HF roasts: {len(raw_roasts):,}")
    print(f"Your original raw roasts: 4,002\n")
    
    # Apply SAME filtering
    filtered_roasts = []
    stats = Counter()
    
    for roast in raw_roasts:
        stats['total'] += 1
        
        # Clean (EXACT same function)
        cleaned = ultra_clean_roast(roast)
        
        if not cleaned:
            stats['cleaning_failed'] += 1
            continue
        
        # Quality check (EXACT same function, minus score)
        is_valid, reason = is_high_quality(cleaned)
        
        if not is_valid:
            stats[reason] += 1
            continue
        
        filtered_roasts.append(cleaned)
        stats['kept'] += 1
    
    # Results
    print("="*70)
    print("📊 FILTERING RESULTS")
    print("="*70 + "\n")
    
    survival_rate = (stats['kept']/stats['total']*100) if stats['total'] > 0 else 0
    
    print("HF Dataset:")
    print(f"  ✅ Kept: {stats['kept']:,}/{stats['total']:,} ({survival_rate:.1f}%)\n")
    
    print("Your Original Dataset (for comparison):")
    print(f"  ✅ Kept: 771/4,002 (19.3%)\n")
    
    print("❌ HF Removal breakdown:")
    removal_reasons = [(reason, count) for reason, count in stats.items() 
                       if reason not in ['total', 'kept']]
    removal_reasons.sort(key=lambda x: x[1], reverse=True)
    
    for reason, count in removal_reasons:
        pct = count/stats['total']*100
        print(f"   {reason:25s}: {count:6,} ({pct:5.1f}%)")
    
    # Length analysis
    lengths = [len(r) for r in filtered_roasts]
    
    print(f"\n📏 Length Statistics:")
    print(f"   Min:    {min(lengths)} chars")
    print(f"   Max:    {max(lengths)} chars")
    print(f"   Mean:   {sum(lengths)/len(lengths):.1f} chars")
    print(f"   Median: {sorted(lengths)[len(lengths)//2]} chars")
    
    your_avg = 71  # From your training data
    print(f"\n   Your data mean: ~{your_avg} chars")
    print(f"   HF data mean:   {sum(lengths)/len(lengths):.1f} chars")
    
    # Sample roasts
    print(f"\n{'='*70}")
    print(f"📋 SAMPLE FILTERED ROASTS (Random 30)")
    print(f"{'='*70}\n")
    
    import random
    random.seed(42)
    samples = random.sample(filtered_roasts, min(30, len(filtered_roasts)))
    for i, roast in enumerate(samples, 1):
        print(f"{i}. [{len(roast)} chars] {roast}\n")
    
    # Save
    output_file = Path("data/huggingface/hf_visual_filtered.json")
    output_data = {
        "source": "gus-gustavo/reddit_roastme",
        "filtering": "Same criteria as clean_and_convert.py (minus score requirement)",
        "total_raw": len(raw_roasts),
        "total_kept": len(filtered_roasts),
        "survival_rate": survival_rate,
        "mean_length": sum(lengths)/len(lengths) if lengths else 0,
        "roasts": filtered_roasts
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"{'='*70}")
    print("COMPARISON & VIABILITY")
    print(f"{'='*70}\n")
    
    print(f"Your data:  693 train samples (with images)")
    print(f"HF data:    {len(filtered_roasts):,} text samples (no images)")
    print(f"Ratio:      {len(filtered_roasts)/693:.0f}x more text data\n")
    
    if len(filtered_roasts) > 10000:
        print("✅ EXCELLENT for Option B (Two-Stage Training)")
        print(f"   Stage 1: {len(filtered_roasts):,} roasts for style learning")
        print("   Stage 2: 693 image+roast pairs for visual grounding\n")
    elif len(filtered_roasts) > 5000:
        print("✅ GOOD for Option B")
        print(f"   {len(filtered_roasts):,} samples should be sufficient\n")
    else:
        print("⚠️  MARGINAL for Option B")
        print(f"   {len(filtered_roasts):,} may not be enough for style learning\n")
    
    print(f"📁 Saved to: {output_file}\n")
    
    return filtered_roasts

if __name__ == "__main__":
    filter_hf_dataset()
