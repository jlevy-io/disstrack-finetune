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
            if isinstance(msg['content'], list):
                for content in msg['content']:
                    if content['type'] == 'text':
                        text = content['text']
                    elif content['type'] == 'image':
                        # Add <image> token at start for user message
                        text = "<image>\n" + text
            else:
                text = msg['content']
            
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
    print(json.dumps(sample, indent=2)[:500] + "...")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Commit & push to GitHub")
    print(f"   2. Pull on RunPod and train!")
    print()

if __name__ == "__main__":
    convert()
