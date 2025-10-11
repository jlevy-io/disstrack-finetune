"""
Analyze dataset quality
Shows sample roasts and statistics
"""
import json
from pathlib import Path

def analyze():
    """Analyze processed dataset"""
    
    data_file = Path("data/processed/v4_visual/train.json")
    
    if not data_file.exists():
        print("❌ No processed dataset found.")
        print("   Run: python tools/filter_visual.py first")
        return
    
    with open(data_file) as f:
        data = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"📊 DATASET ANALYSIS")
    print(f"{'='*70}\n")
    print(f"Total training samples: {len(data)}")
    
    # Extract roasts
    roasts = [sample['messages'][2]['content'][0]['text'] for sample in data]
    lengths = [len(r) for r in roasts]
    
    print(f"\n📏 Roast lengths:")
    print(f"   Min: {min(lengths)} chars")
    print(f"   Max: {max(lengths)} chars")
    print(f"   Avg: {sum(lengths) / len(lengths):.0f} chars")
    
    # Sample roasts
    print(f"\n🔥 Sample Roasts:\n")
    import random
    samples = random.sample(data, min(10, len(data)))
    
    for i, sample in enumerate(samples, 1):
        roast = sample['messages'][2]['content'][0]['text']
        score = sample['metadata']['roast_score']
        print(f"{i}. [Score: {score}]")
        print(f"   {roast}\n")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    analyze()
