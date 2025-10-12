"""
Normalize images for Qwen2.5-VL training
Optimizes size and quality for faster training without quality loss
"""

from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil
import sys

def normalize_images(
    input_dir: str,
    output_dir: str,
    max_size: int = 1024,
    quality: int = 90,
    backup: bool = True,
    verbose: bool = True
):
    """
    Normalize images for vision model training
    
    Args:
        input_dir: Source directory with images
        output_dir: Destination for normalized images
        max_size: Maximum dimension (keeps aspect ratio)
        quality: JPEG quality (1-100)
        backup: Create backup of originals
        verbose: Print detailed stats
    
    Returns:
        dict: Statistics about the normalization
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_path}")
        return None
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Backup if requested
    if backup:
        backup_path = Path(f"{input_dir}_backup")
        if not backup_path.exists():
            if verbose:
                print(f"💾 Creating backup at {backup_path}...")
            shutil.copytree(input_path, backup_path)
            if verbose:
                print(f"✅ Backup created!\n")
    
    if verbose:
        print("="*70)
        print("🖼️  IMAGE NORMALIZATION")
        print("="*70)
        print(f"\nSettings:")
        print(f"  Max dimension: {max_size}px")
        print(f"  JPEG quality: {quality}")
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}\n")
    
    # Find all images
    image_files = list(input_path.glob("*.jpg"))
    image_files += list(input_path.glob("*.jpeg"))
    image_files += list(input_path.glob("*.png"))
    
    if not image_files:
        print("❌ No images found!")
        return None
    
    if verbose:
        print(f"Found {len(image_files)} images\n")
    
    # Stats
    stats = {
        'total': len(image_files),
        'processed': 0,
        'errors': 0,
        'resized': 0,
        'size_before': 0,
        'size_after': 0
    }
    
    iterator = tqdm(image_files, desc="Normalizing") if verbose else image_files
    
    for img_path in iterator:
        try:
            # Get original size
            original_size = img_path.stat().st_size
            stats['size_before'] += original_size
            
            # Open image
            img = Image.open(img_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get current dimensions
            width, height = img.size
            original_dimensions = (width, height)
            
            # Calculate new size (maintain aspect ratio)
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                
                # High-quality resize using LANCZOS
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                stats['resized'] += 1
            
            # Save optimized JPEG
            output_file = output_path / f"{img_path.stem}.jpg"
            img.save(
                output_file,
                'JPEG',
                quality=quality,
                optimize=True,
                progressive=True
            )
            
            # Get new size
            new_size = output_file.stat().st_size
            stats['size_after'] += new_size
            
            stats['processed'] += 1
            
        except Exception as e:
            stats['errors'] += 1
            if verbose:
                print(f"\n⚠️  Error processing {img_path.name}: {e}")
    
    # Report
    if verbose:
        print(f"\n{'='*70}")
        print(f"📊 NORMALIZATION COMPLETE")
        print(f"{'='*70}\n")
        
        print(f"✅ Processed: {stats['processed']}/{stats['total']}")
        print(f"   Resized: {stats['resized']}")
        if stats['errors'] > 0:
            print(f"❌ Errors: {stats['errors']}")
        
        size_before_mb = stats['size_before'] / 1024 / 1024
        size_after_mb = stats['size_after'] / 1024 / 1024
        reduction_pct = (1 - stats['size_after']/stats['size_before'])*100 if stats['size_before'] > 0 else 0
        
        print(f"\n📦 Storage:")
        print(f"   Before: {size_before_mb:.1f} MB")
        print(f"   After:  {size_after_mb:.1f} MB")
        print(f"   Saved:  {size_before_mb - size_after_mb:.1f} MB")
        print(f"   Reduction: {reduction_pct:.1f}%")
        
        print(f"\n📁 Normalized images: {output_path}")
        print(f"\n{'='*70}\n")
    
    return stats

def update_training_json(
    json_file: str,
    new_image_folder: str
):
    """
    Update training JSON to point to normalized images
    """
    
    import json
    
    print("="*70)
    print("🔄 UPDATING TRAINING DATA")
    print("="*70 + "\n")
    
    json_path = Path(json_file)
    
    if not json_path.exists():
        print(f"❌ Training file not found: {json_file}")
        return False
    
    # Backup original
    backup_file = str(json_path).replace('.json', '_backup.json')
    shutil.copy(json_path, backup_file)
    print(f"💾 Backup saved: {backup_file}")
    
    # Load and update
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"📝 Updating {len(data)} entries...")
    
    # Update image paths
    for item in data:
        if 'image' in item:
            filename = Path(item['image']).name
            # Ensure .jpg extension
            if not filename.endswith('.jpg'):
                filename = filename.rsplit('.', 1)[0] + '.jpg'
            item['image'] = filename
    
    # Save
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Updated image paths to: {new_image_folder}/")
    print(f"📁 Saved to: {json_path}\n")
    print("="*70 + "\n")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Normalize images for training")
    parser.add_argument('--input', default='data/raw/images', help='Input directory')
    parser.add_argument('--output', default='data/normalized_images', help='Output directory')
    parser.add_argument('--max-size', type=int, default=1024, help='Max dimension')
    parser.add_argument('--quality', type=int, default=90, help='JPEG quality')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup')
    parser.add_argument('--json', default='data/llava_format/train.json', help='Training JSON to update')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🖼️  IMAGE NORMALIZATION TOOL")
    print("="*70 + "\n")
    
    # Normalize images
    stats = normalize_images(
        input_dir=args.input,
        output_dir=args.output,
        max_size=args.max_size,
        quality=args.quality,
        backup=not args.no_backup,
        verbose=True
    )
    
    if not stats:
        sys.exit(1)
    
    # Ask if should update training data
    print("="*70)
    choice = input("Update training JSON paths? (yes/no): ").strip().lower()
    
    if choice == "yes":
        if update_training_json(args.json, args.output):
            print("✅ All done! Images normalized and training data updated.")
        else:
            print("⚠️  Images normalized but JSON update failed.")
    else:
        print("\n💡 Remember to update your training script to use normalized images!")
        print(f"   Update IMAGE_FOLDER to: {args.output}")
