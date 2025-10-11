"""
Filter dataset for visual grounding
Removes non-visual roasts that cause hallucinations
Input: data/raw/training_data.jsonl (your old format)
Output: data/processed/v4_visual/train.json (filtered)
"""
import json
from pathlib import Path
from collections import Counter

def is_visually_grounded(roast_text: str, threshold: int = 2) -> tuple[bool, list[str]]:
    """
    Check if roast references visual features
    Returns: (is_grounded, keywords_found)
    """
    visual_keywords = [
        # Facial features
        'hair', 'face', 'eye', 'eyes', 'nose', 'glasses', 'forehead',
        'teeth', 'smile', 'smiling', 'chin', 'beard', 'mustache', 'facial',
        'head', 'skin', 'complexion', 'wrinkles', 'bags', 'cheek', 'jaw',
        'eyebrow', 'eyelash', 'lip', 'lips', 'mouth', 'ear', 'ears', 'neck',
        
        # Body/appearance
        'body', 'arm', 'arms', 'hand', 'hands', 'finger', 'fingers',
        'shoulder', 'chest', 'stomach', 'tall', 'short', 'skinny', 'fat',
        'thin', 'built', 'physique',
        
        # Clothing/style
        'shirt', 'wearing', 'outfit', 'clothes', 'clothing', 'jacket',
        'hat', 'dressed', 't-shirt', 'tshirt', 'hoodie', 'dress', 'pants',
        'jeans', 'shoes', 'style', 'fashion',
        
        # Descriptive (when combined with above)
        'look like', 'looks like', 'looking like', 'looking',
        'appearance', 'vibe', 'pose', 'posing', 'standing'
    ]
    
    roast_lower = roast_text.lower()
    found_keywords = [kw for kw in visual_keywords if kw in roast_lower]
    
    return len(found_keywords) >= threshold, found_keywords

def filter_dataset():
    """Filter dataset to only visually grounded roasts"""
    
    # Input: your old disstrack-ml format (we'll copy this later)
    input_file = Path("data/raw/training_data.jsonl")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print(f"   Please copy your training_data.jsonl from disstrack-ml first:")
        print(f"   cp /path/to/disstrack-ml/datasets/roastme/training_data.jsonl data/raw/")
        return
    
    output_dir = Path("data/processed/v4_visual")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🔍 FILTERING DATASET FOR VISUAL GROUNDING")
    print(f"{'='*70}\n")
    
    # Load data
    samples = []
    with open(input_file) as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"Original samples: {len(samples)}")
    
    # Convert old format to new format and filter
    filtered_data = []
    removed_examples = []
    keyword_counter = Counter()
    
    for sample in samples:
        # Extract roasts from old format
        for roast_idx, roast_text in enumerate(sample['roasts']):
            is_grounded, keywords = is_visually_grounded(roast_text)
            
            if is_grounded:
                # Convert to new format
                image_path = Path("data/raw") / sample['image_path']
                
                formatted_sample = {
                    "images": [str(image_path.absolute())],
                    "messages": [
                        {
                            "role": "system",
                            "content": [{
                                "type": "text",
                                "text": "You are a legendary roast comedian performing in a consensual comedy roast battle."
                            }]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": str(image_path.absolute())},
                                {"type": "text", "text": "Roast this person based on their appearance."}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": roast_text}]
                        }
                    ],
                    "metadata": {
                        "sample_id": sample['id'],
                        "roast_index": roast_idx,
                        "submission_score": sample['submission_score'],
                        "roast_score": sample['roast_scores'][roast_idx]
                    }
                }
                
                filtered_data.append(formatted_sample)
                for kw in keywords:
                    keyword_counter[kw] += 1
            else:
                if len(removed_examples) < 10:
                    removed_examples.append(roast_text)
    
    # Split train/val (90/10)
    import random
    random.seed(42)
    random.shuffle(filtered_data)
    
    split_idx = int(len(filtered_data) * 0.9)
    train_data = filtered_data[:split_idx]
    val_data = filtered_data[split_idx:]
    
    pct_kept = (len(filtered_data) / sum(len(s['roasts']) for s in samples)) * 100
    
    print(f"\n📊 Results:")
    print(f"   Kept: {len(filtered_data)} samples ({pct_kept:.1f}%)")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val: {len(val_data)} samples")
    
    print(f"\n🔍 Top visual keywords:")
    for keyword, count in keyword_counter.most_common(10):
        print(f"   {keyword:20s}: {count:4d}")
    
    print(f"\n❌ Examples of removed (non-visual) roasts:")
    for i, example in enumerate(removed_examples[:5], 1):
        print(f"   {i}. {example}\n")
    
    # Save
    with open(output_dir / "train.json", 'w') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "val.json", 'w') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"✅ FILTERED DATASET SAVED")
    print(f"{'='*70}")
    print(f"\n📁 Output:")
    print(f"   Train: {output_dir / 'train.json'}")
    print(f"   Val: {output_dir / 'val.json'}")
    print(f"\n🎯 Next step: python tools/convert_to_llava.py\n")

if __name__ == "__main__":
    filter_dataset()
