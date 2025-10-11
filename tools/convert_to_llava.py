"""
Convert filtered dataset to LLaVA format
Input: data/processed/v4_visual/train.json
Output: data/llava_format/train.json
"""
import json
from pathlib import Path

def convert():
    """Convert to LLaVA format for Qwen2-VL-Finetune repo"""
    
    input_file = Path("data/processed/v4_visual/train.json")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print(f"   Run: python tools/filter_visual.py first")
        return
    
    output_file = Path("data/llava_format/train.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Converting to LLaVA format...")
    
    with open(input_file) as f:
        data = json.load(f)
    
    llava_format = []
    
    for idx, sample in enumerate(data):
        # Extract image filename only (not full path)
        image_path = sample['images'][0]
        image_filename = Path(image_path).name
        
        # Convert conversations (skip system message for LLaVA format)
        conversations = []
        for msg in sample['messages'][1:]:  # Skip system message
            role = "human" if msg['role'] == "user" else "gpt"
            
            # Extract text from content
            text = ""
            has_image = False
            
            if isinstance(msg['content'], list):
                for content in msg['content']:
                    if content['type'] == 'text':
                        text = content['text']
                    elif content['type'] == 'image':
                        has_image = True
            else:
                text = msg['content']
            
            # Add <image> token for user message with image
            if role == "human" and has_image:
                text = f"<image>\n{text}"
            elif role == "human" and not has_image:
                # User message should have image token since there's an image field
                text = f"<image>\n{text}"
            
            conversations.append({
                "from": role,
                "value": text
            })
        
        llava_format.append({
            "id": f"roast_{idx}",
            "image": image_filename,
            "conversations": conversations
        })
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(llava_format, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Converted {len(llava_format)} samples")
    print(f"📁 Output: {output_file}")
    
    # Show sample
    print(f"\n📋 Sample converted format:\n")
    sample = llava_format[0]
    print(json.dumps(sample, indent=2)[:600] + "...")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Train: bash scripts/finetune_roastme_h100.sh")
    print()

if __name__ == "__main__":
    convert()
