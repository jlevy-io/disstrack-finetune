"""
tools/prepare_v3_stage1_data.py

Sample 20k SHORT visual roasts from HF dataset for Stage 1
STRICT: 25-100 chars only (under 100 ideal)
"""

import json
import random
from pathlib import Path


def prepare_stage1_data(num_samples: int = 20000, max_length: int = 100, seed: int = 42):
    """
    Prepare text-only training data for Stage 1 style learning
    
    Args:
        num_samples: Target number of samples (will sample from eligible roasts)
        max_length: Maximum roast length (default: 100 to match "under 100 ideal")
        seed: Random seed for reproducibility
    """

    print("\n" + "=" * 70)
    print("📋 PREPARING v3 STAGE 1 DATA (Text-Only, SHORT)")
    print("=" * 70 + "\n")

    hf_file = Path("data/huggingface/hf_visual_filtered.json")

    if not hf_file.exists():
        print(f"❌ Filtered HF dataset not found: {hf_file}")
        print("\nRun this first:")
        print("   python tools/filter_hf_with_visual_criteria.py")
        print()
        return None

    print(f"Loading: {hf_file}")
    with open(hf_file) as f:
        data = json.load(f)

    all_roasts = data["roasts"]
    print(f"Total filtered roasts: {len(all_roasts):,}")
    
    # Filter to SHORT roasts only (25-100 chars)
    short_roasts = [r for r in all_roasts if 25 <= len(r) <= max_length]
    
    print(f"\n📏 Length Filtering:")
    print(f"   Original: {len(all_roasts):,} roasts")
    print(f"   After filtering to 25-{max_length} chars: {len(short_roasts):,} roasts")
    print(f"   Retention: {len(short_roasts)/len(all_roasts)*100:.1f}%")
    print(f"   Target: {num_samples:,} samples\n")

    if len(short_roasts) < num_samples:
        print(f"⚠️  Warning: Only {len(short_roasts):,} short roasts available")
        print(f"   Will use all {len(short_roasts):,} samples instead of {num_samples:,}\n")
        sampled_roasts = short_roasts
    else:
        random.seed(seed)
        sampled_roasts = random.sample(short_roasts, num_samples)

    llava_data = []
    for i, roast in enumerate(sampled_roasts):
        llava_data.append({
            "id": f"hf_stage1_{i}",
            "conversations": [
                {"from": "human", "value": "Roast this person based on their appearance."},
                {"from": "gpt", "value": roast}
            ]
        })

    lengths = [len(r) for r in sampled_roasts]

    print("=" * 70)
    print("📊 FINAL DATASET STATISTICS")
    print("=" * 70 + "\n")

    print(f"Total samples: {len(llava_data):,}")
    print(f"Avg length: {sum(lengths)/len(lengths):.1f} chars")
    print(f"Median length: {sorted(lengths)[len(lengths)//2]} chars")
    print(f"Min length: {min(lengths)} chars")
    print(f"Max length: {max(lengths)} chars")
    print(f"Std dev: {(sum((x - sum(lengths)/len(lengths))**2 for x in lengths) / len(lengths))**0.5:.1f} chars\n")

    print("Length Distribution:")
    bins = [(25, 50), (51, 75), (76, 100)]
    for low, high in bins:
        count = sum(1 for l in lengths if low <= l <= high)
        pct = (count / len(lengths)) * 100
        bar = "█" * int(pct / 2)
        print(f"   {low:3d}-{high:3d} chars: {count:5,} ({pct:5.1f}%) {bar}")

    print(f"\n✅ All roasts are under 100 chars (matches system prompt ideal)")
    print(f"✅ Avg {sum(lengths)/len(lengths):.1f} chars aligns with training goal\n")

    output_dir = Path("data/llava_format")
    output_file = output_dir / "stage1_text_only.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(llava_data, f, indent=2, ensure_ascii=False)

    file_size_mb = output_file.stat().st_size / 1024 / 1024

    print(f"✅ Saved to: {output_file}")
    print(f"📁 File size: {file_size_mb:.1f} MB\n")

    print("=" * 70)
    print("📋 SAMPLE ROASTS (Random 20)")
    print("=" * 70 + "\n")

    samples = random.sample(sampled_roasts, min(20, len(sampled_roasts)))
    for i, roast in enumerate(samples, 1):
        print(f"{i}. [{len(roast)} chars] {roast}\n")

    print("=" * 70)
    print("✅ STAGE 1 DATA READY")
    print("=" * 70 + "\n")

    print("📋 Next Step:")
    print("   bash scripts/finetune_roastme_v3_stage1.sh\n")

    return output_file


if __name__ == "__main__":
    prepare_stage1_data(num_samples=20000, max_length=100)
