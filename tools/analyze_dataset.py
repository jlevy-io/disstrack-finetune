"""
Analyze dataset quality
Shows sample roasts and statistics including outliers
"""
import json
from pathlib import Path
from collections import Counter

def analyze():
    """Analyze processed dataset with detailed statistics"""
    
    data_file = Path("data/llava_format/train.json")
    
    if not data_file.exists():
        print("❌ No training dataset found.")
        print("   Run: python tools/clean_and_convert.py first")
        return
    
    with open(data_file) as f:
        data = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"📊 DATASET ANALYSIS")
    print(f"{'='*70}\n")
    print(f"Total training samples: {len(data)}")
    
    # Extract roasts and metadata
    roasts = []
    for sample in data:
        roast_text = sample['conversations'][1]['value']
        roasts.append({
            'text': roast_text,
            'length': len(roast_text),
            'id': sample['id']
        })
    
    lengths = [r['length'] for r in roasts]
    
    print(f"\n📏 Roast Length Statistics:")
    print(f"   Min:    {min(lengths)} chars")
    print(f"   Max:    {max(lengths)} chars")
    print(f"   Mean:   {sum(lengths) / len(lengths):.1f} chars")
    print(f"   Median: {sorted(lengths)[len(lengths)//2]} chars")
    
    # Show length distribution
    print(f"\n📊 Length Distribution:")
    bins = [(0, 50), (51, 100), (101, 150), (151, 200), (201, 300), (301, 500)]
    for low, high in bins:
        count = sum(1 for l in lengths if low <= l <= high)
        pct = (count / len(lengths)) * 100
        bar = '█' * int(pct / 2)
        print(f"   {low:3d}-{high:3d} chars: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Show outliers (very long roasts)
    print(f"\n⚠️  Long Roasts (>200 chars):")
    long_roasts = sorted([r for r in roasts if r['length'] > 200], 
                         key=lambda x: x['length'], reverse=True)
    
    if long_roasts:
        print(f"   Found {len(long_roasts)} roasts exceeding 200 characters:\n")
        for i, roast in enumerate(long_roasts[:5], 1):  # Show top 5 longest
            print(f"{i}. [{roast['id']}] {roast['length']} chars:")
            print(f"   {roast['text'][:150]}...")
            print()
    else:
        print(f"   ✅ None! All roasts are under 200 characters.")
    
    # Check for artifacts
    print(f"\n🔍 Artifact Detection:")
    artifacts = {
        'Edit/Thanks': 0,
        'URLs': 0,
        'Reddit refs': 0,
        'Emojis': 0
    }
    
    import re
    for roast in roasts:
        text = roast['text']
        if re.search(r'edit:|thank', text, re.IGNORECASE):
            artifacts['Edit/Thanks'] += 1
        if re.search(r'http|www\.|\.com', text, re.IGNORECASE):
            artifacts['URLs'] += 1
        if re.search(r'r/|u/', text):
            artifacts['Reddit refs'] += 1
        if re.search(r'[\U00010000-\U0010ffff]', text):
            artifacts['Emojis'] += 1
    
    total_artifacts = sum(artifacts.values())
    if total_artifacts > 0:
        print(f"   ⚠️  Found {total_artifacts} potential artifacts:")
        for artifact_type, count in artifacts.items():
            if count > 0:
                pct = (count / len(roasts)) * 100
                print(f"      {artifact_type}: {count} ({pct:.1f}%)")
    else:
        print(f"   ✅ No artifacts detected!")
    
    # Sample roasts
    print(f"\n{'='*70}")
    print(f"🔥 RANDOM SAMPLE ROASTS")
    print(f"{'='*70}\n")
    
    import random
    samples = random.sample(roasts, min(10, len(roasts)))
    
    for i, roast in enumerate(samples, 1):
        print(f"{i}. [{roast['id']}] ({roast['length']} chars)")
        print(f"   {roast['text']}\n")
    
    print(f"{'='*70}\n")
    
    # Summary
    clean_pct = ((len(roasts) - total_artifacts) / len(roasts)) * 100
    print(f"✅ Dataset Quality: {clean_pct:.1f}% clean")
    print(f"📁 Dataset location: {data_file}")
    print()

if __name__ == "__main__":
    analyze()
