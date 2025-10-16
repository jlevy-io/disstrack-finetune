"""
tools/analyze_reddit_roastme_hf.py

Download and analyze the HuggingFace r/RoastMe dataset
"""

from datasets import load_dataset
import json
from pathlib import Path
from collections import Counter
import re

def clean_roast_text(text: str) -> str:
    """Quick cleaning"""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove Reddit references
    text = re.sub(r'r/\w+', '', text)
    text = re.sub(r'u/\w+', '', text)
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def analyze_dataset():
    print("\n" + "="*70)
    print("📊 DOWNLOADING & ANALYZING r/RoastMe DATASET")
    print("="*70 + "\n")
    
    # Download from HuggingFace
    print("⬇️  Downloading from HuggingFace...")
    print("   (First time will download, subsequent runs use cache)\n")
    
    try:
        dataset = load_dataset("gus-gustavo/reddit_roastme")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nTry installing: pip install datasets")
        return None
    
    print(f"✅ Dataset loaded!\n")
    
    # Determine structure
    print("Dataset structure:")
    print(f"  Keys: {list(dataset.keys())}")
    
    # Get the data
    if 'train' in dataset:
        data = dataset['train']
    else:
        # Sometimes datasets don't have splits
        data = list(dataset.values())[0]
    
    print(f"  Total samples: {len(data)}")
    print(f"  Columns: {data.column_names if hasattr(data, 'column_names') else list(data[0].keys())}")
    print()
    
    # Show first few examples
    print("="*70)
    print("SAMPLE DATA")
    print("="*70 + "\n")
    
    for i in range(min(3, len(data))):
        print(f"Example {i+1}:")
        item = data[i]
        for key, value in item.items():
            if isinstance(value, str):
                if len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        print()
    
    # Identify roast field
    roast_field = None
    for field in ['roast', 'comment', 'text', 'body', 'content']:
        if field in data[0]:
            roast_field = field
            break
    
    if not roast_field:
        print("⚠️  Could not identify roast text field!")
        print(f"Available fields: {list(data[0].keys())}")
        return None
    
    print(f"✅ Using '{roast_field}' as roast text field\n")
    
    # Check for images
    has_images = any(field in data[0] for field in ['image', 'image_url', 'url', 'img'])
    image_field = next((f for f in ['image', 'image_url', 'url', 'img'] if f in data[0]), None)
    
    print(f"📷 Images available: {has_images}")
    if has_images:
        print(f"   Image field: {image_field}")
    print()
    
    # Extract roasts
    roasts = [item[roast_field] for item in data]
    
    # Length analysis
    print("="*70)
    print("LENGTH ANALYSIS")
    print("="*70 + "\n")
    
    lengths = [len(r) for r in roasts]
    print(f"Total roasts: {len(roasts)}")
    print(f"Min length: {min(lengths)} chars")
    print(f"Max length: {max(lengths)} chars")
    print(f"Mean length: {sum(lengths)/len(lengths):.1f} chars")
    print(f"Median length: {sorted(lengths)[len(lengths)//2]} chars")
    
    print("\nLength Distribution:")
    bins = [(0, 50), (51, 100), (101, 150), (151, 200), (201, 300), (301, 500), (501, 10000)]
    for low, high in bins:
        count = sum(1 for l in lengths if low <= l <= high)
        pct = (count / len(lengths)) * 100
        bar = '█' * int(pct / 2)
        print(f"  {low:4d}-{high:4d}: {count:5d} ({pct:5.1f}%) {bar}")
    
    # Quality filtering (similar to your criteria)
    print("\n" + "="*70)
    print("QUALITY FILTERING (YOUR CRITERIA)")
    print("="*70 + "\n")
    
    filtered_roasts = []
    stats = Counter()
    
    for roast in roasts:
        stats['total'] += 1
        
        cleaned = clean_roast_text(roast)
        
        # Length check
        if len(cleaned) < 20:
            stats['too_short'] += 1
            continue
        if len(cleaned) > 150:
            stats['too_long'] += 1
            continue
        
        # Basic quality checks
        roast_lower = cleaned.lower()
        
        # Check for artifacts
        if any(word in roast_lower for word in ['edit:', 'http', 'www.', '.com']):
            stats['has_artifact'] += 1
            continue
        
        filtered_roasts.append(cleaned)
        stats['kept'] += 1
    
    print(f"Kept: {stats['kept']}/{stats['total']} ({stats['kept']/stats['total']*100:.1f}%)\n")
    
    print("Removed:")
    for reason, count in stats.items():
        if reason not in ['total', 'kept'] and count > 0:
            pct = count/stats['total']*100
            print(f"  {reason:20s}: {count:5d} ({pct:5.1f}%)")
    
    # Sample filtered roasts
    print("\n" + "="*70)
    print(f"SAMPLE FILTERED ROASTS (Top 30)")
    print("="*70 + "\n")
    
    import random
    samples = random.sample(filtered_roasts, min(30, len(filtered_roasts)))
    for i, roast in enumerate(samples, 1):
        print(f"{i}. [{len(roast)} chars] {roast}\n")
    
    # Save filtered data
    output_dir = Path("data/huggingface")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "filtered_roasts.json"
    
    output_data = {
        "source": "gus-gustavo/reddit_roastme",
        "total_raw": len(roasts),
        "total_filtered": len(filtered_roasts),
        "has_images": has_images,
        "image_field": image_field,
        "roasts": filtered_roasts
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("="*70)
    print("SUMMARY")
    print("="*70 + "\n")
    
    print(f"✅ Found {len(filtered_roasts)} high-quality roasts (20-150 chars)")
    print(f"📁 Saved to: {output_file}")
    print(f"📷 Images available: {has_images}")
    
    if has_images:
        print("\n💡 Next steps:")
        print("   - Can use directly for training (with images)")
        print("   - Combine with your 305 existing images")
    else:
        print("\n💡 Next steps (no images in dataset):")
        print("   - Use for text-only style learning")
        print("   - Match roasts to your 305 existing images")
        print("   - Consider scraping original Reddit images")
    
    print(f"\n📊 Your data: 693 train samples")
    print(f"📊 New data: {len(filtered_roasts)} roasts")
    print(f"📊 Combined potential: {693 + len(filtered_roasts)} samples")
    print()
    
    return dataset, filtered_roasts, has_images

if __name__ == "__main__":
    analyze_dataset()
