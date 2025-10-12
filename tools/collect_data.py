"""
Collect r/RoastMe data with automatic image normalization
"""

import praw
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os
from tqdm import tqdm
from PIL import Image
import io

load_dotenv()

class RoastMeCollector:
    def __init__(self, output_dir: str = "data/raw"):
        print("Initializing Reddit connection...")
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "RoastMe Data Collector v1.0")
        )
        
        try:
            print(f"Connected as: {self.reddit.user.me()}")
        except:
            print("Connected (read-only mode)")
        
        self.subreddit = self.reddit.subreddit("RoastMe")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        self.progress_file = self.output_dir / "collection_progress.json"
        self.processed_ids = self._load_progress()
        
        print(f"Previously processed: {len(self.processed_ids)} submissions")

    def _load_progress(self) -> set:
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                data = json.load(f)
                return set(data.get("processed_ids", []))
        return set()

    def _save_progress(self):
        with open(self.progress_file, "w") as f:
            json.dump({
                "processed_ids": list(self.processed_ids),
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)

    def reset_collection(self):
        """Delete all previously collected data"""
        print("\n" + "="*70)
        print("⚠️  RESETTING COLLECTION")
        print("="*70)
        
        self.processed_ids = set()
        if self.progress_file.exists():
            self.progress_file.unlink()
        
        if self.images_dir.exists():
            for img in self.images_dir.glob("*"):
                img.unlink()
        
        training_file = self.output_dir / "training_data.jsonl"
        if training_file.exists():
            training_file.unlink()
        
        metadata_file = self.output_dir / "metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()
        
        print("✅ All previous data cleared!")
        print("="*70 + "\n")

    def is_valid_submission(
        self, 
        submission, 
        min_score: int = 100, 
        min_comments: int = 20
    ) -> bool:
        if submission.id in self.processed_ids:
            return False
        
        if submission.stickied:
            return False
        
        if not submission.url or not any(ext in submission.url.lower() 
                                        for ext in ['.jpg', '.jpeg', '.png', 'i.redd.it', 'imgur']):
            return False
        
        if submission.score < min_score:
            return False
        
        if submission.num_comments < min_comments:
            return False
        
        if submission.over_18:
            return False
        
        return True

    def download_image(self, url: str, max_size: int = 1024, quality: int = 90) -> Optional[bytes]:
        """Download and normalize image inline"""
        try:
            if 'imgur' in url and not url.endswith(('.jpg', '.jpeg', '.png')):
                url = url + '.jpg'
            
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                if 'image' in content_type:
                    # Normalize image inline
                    img = Image.open(io.BytesIO(response.content))
                    
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize if needed (maintain aspect ratio)
                    width, height = img.size
                    if width > max_size or height > max_size:
                        if width > height:
                            new_width = max_size
                            new_height = int(height * (max_size / width))
                        else:
                            new_height = max_size
                            new_width = int(width * (max_size / height))
                        
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Save to bytes as optimized JPEG
                    output = io.BytesIO()
                    img.save(output, 'JPEG', quality=quality, optimize=True, progressive=True)
                    return output.getvalue()
                    
        except Exception as e:
            print(f"  ⚠️  Image download/normalization error: {e}")
        return None

    def extract_top_roasts(
        self, 
        submission, 
        min_score: int = 50,
        max_roasts: int = 15
    ) -> List[Dict]:
        """Extract roasts"""
        try:
            submission.comment_sort = "top"
            submission.comments.replace_more(limit=0)
        except Exception as e:
            print(f"  ⚠️  Comment loading error: {e}")
            return []
        
        roasts = []
        
        for comment in submission.comments:
            if comment.score < min_score:
                continue
            
            if len(comment.body) < 20 or len(comment.body) > 400:
                continue
            
            if comment.distinguished:
                continue
            
            body_lower = comment.body.lower()
            if any(word in body_lower for word in ['removed', 'deleted', '[removed]', '[deleted]']):
                continue
            
            roasts.append({
                "text": comment.body,
                "score": comment.score,
                "author": str(comment.author) if comment.author else "[deleted]",
                "created_utc": comment.created_utc
            })
            
            if len(roasts) >= max_roasts:
                break
        
        return roasts

    def collect_tier(
        self,
        time_filter: str,
        limit: int,
        tier_name: str,
        min_comment_score: int = 50,
        min_submission_score: int = 100,
        min_comments: int = 20,
        min_roasts: int = 2
    ) -> List[Dict]:
        
        print(f"\n{'='*70}")
        print(f"Collecting {tier_name}")
        print(f"Filter: {time_filter} | Limit: {limit}")
        print(f"Min Submission: {min_submission_score} | Min Comments: {min_comments}")
        print(f"Min Comment Score: {min_comment_score} | Min Roasts: {min_roasts}")
        print(f"{'='*70}\n")
        
        collected_data = []
        skipped = 0
        
        try:
            print("Fetching submissions from Reddit...")
            submissions = self.subreddit.top(time_filter=time_filter, limit=limit)
            
            submissions_list = []
            print("Loading submission list...")
            for i, submission in enumerate(submissions):
                submissions_list.append(submission)
                if (i + 1) % 100 == 0:
                    print(f"  Loaded {i + 1} submissions...")
            
            print(f"✓ Got {len(submissions_list)} submissions\n")
            
            with tqdm(total=len(submissions_list), desc=tier_name) as pbar:
                for submission in submissions_list:
                    pbar.update(1)
                    
                    if not self.is_valid_submission(
                        submission, 
                        min_submission_score, 
                        min_comments
                    ):
                        skipped += 1
                        continue
                    
                    image_bytes = self.download_image(submission.url)
                    if not image_bytes:
                        skipped += 1
                        continue
                    
                    roasts = self.extract_top_roasts(
                        submission, 
                        min_score=min_comment_score
                    )
                    
                    if len(roasts) < min_roasts:
                        skipped += 1
                        continue
                    
                    image_filename = f"{submission.id}.jpg"
                    image_path = self.images_dir / image_filename
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    collected_data.append({
                        "id": submission.id,
                        "title": submission.title,
                        "url": submission.url,
                        "submission_score": submission.score,
                        "num_comments": submission.num_comments,
                        "created_utc": submission.created_utc,
                        "image_filename": image_filename,
                        "roasts": [r["text"] for r in roasts],
                        "roast_scores": [r["score"] for r in roasts],
                        "tier": tier_name
                    })
                    
                    self.processed_ids.add(submission.id)
                    
                    if len(collected_data) % 10 == 0:
                        self._save_progress()
                    
                    time.sleep(0.5)
            
            print(f"\n✓ {tier_name}: {len(collected_data)} collected, {skipped} skipped\n")
            
        except Exception as e:
            print(f"\n✗ Error in {tier_name}: {e}")
            import traceback
            traceback.print_exc()
        
        return collected_data

    def collect_all(self) -> List[Dict]:
        """Collection with tiered strategy"""
        
        all_data = []
        
        tiers_config = [
            {
                "time_filter": "all",
                "limit": 1000,
                "tier_name": "Tier 1: All-Time Legends (500+)",
                "min_comment_score": 100,
                "min_submission_score": 500,
                "min_comments": 30,
                "min_roasts": 3
            },
            {
                "time_filter": "all",
                "limit": 1000,
                "tier_name": "Tier 2: All-Time Strong (200-500)",
                "min_comment_score": 75,
                "min_submission_score": 200,
                "min_comments": 25,
                "min_roasts": 3
            },
            {
                "time_filter": "all",
                "limit": 1500,
                "tier_name": "Tier 3: All-Time Solid (100-200)",
                "min_comment_score": 50,
                "min_submission_score": 100,
                "min_comments": 20,
                "min_roasts": 2
            },
            {
                "time_filter": "all",
                "limit": 1500,
                "tier_name": "Tier 4: All-Time Decent (50-100)",
                "min_comment_score": 40,
                "min_submission_score": 50,
                "min_comments": 15,
                "min_roasts": 2
            },
            {
                "time_filter": "year",
                "limit": 1000,
                "tier_name": "Tier 5: Past Year Top",
                "min_comment_score": 50,
                "min_submission_score": 100,
                "min_comments": 15,
                "min_roasts": 2
            },
            {
                "time_filter": "month",
                "limit": 500,
                "tier_name": "Tier 6: Past Month Fresh",
                "min_comment_score": 30,
                "min_submission_score": 50,
                "min_comments": 10,
                "min_roasts": 2
            }
        ]
        
        for config in tiers_config:
            tier_data = self.collect_tier(**config)
            all_data.extend(tier_data)
            self._save_progress()
            
            print(f"📊 Running total: {len(all_data)} samples\n")
            
            if len(all_data) >= 1000:
                print(f"🎯 Target of 1000+ reached!")
                break
        
        return all_data

    def save_dataset(self, data: List[Dict]):
        """Save with metadata"""
        
        training_file = self.output_dir / "training_data.jsonl"
        with open(training_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        total_roasts = sum(len(d['roasts']) for d in data)
        
        metadata = {
            "total_samples": len(data),
            "total_roasts": total_roasts,
            "avg_roasts_per_post": round(total_roasts / len(data), 2) if data else 0,
            "collection_date": datetime.now().isoformat(),
            "source": "r/RoastMe",
            "version": "v1.0",
            "filters": {
                "min_submission_score": "50-500 (tiered)",
                "min_comment_score": "30-100 (tiered)",
                "min_roasts_per_post": 2
            }
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✅ Dataset Saved!")
        print(f"{'='*70}")
        print(f"\nLocation: {self.output_dir}")
        print(f"Total samples: {len(data)}")
        print(f"Total roasts: {total_roasts}")
        print(f"Avg roasts per sample: {total_roasts / len(data):.1f}")
        print(f"\n{'='*70}\n")

def main():
    print("\n" + "="*70)
    print("🔥 r/RoastMe Data Collection v1.0")
    print("="*70 + "\n")
    
    if not os.getenv("REDDIT_CLIENT_ID"):
        print("❌ Reddit API credentials not found!")
        return
    
    collector = RoastMeCollector()
    
    print("\n⚠️  Reset previous collection?")
    print("  1. Reset and start fresh (RECOMMENDED)")
    print("  2. Continue from previous")
    
    choice = input("\nYour choice (1 or 2): ").strip()
    
    if choice == "1":
        confirm = input("Delete all data? (yes/no): ").strip().lower()
        if confirm == "yes":
            collector.reset_collection()
    
    print("\n🎯 Collection Strategy:")
    print("   - Target: 800-1200 posts")
    print("   - Tiered scoring (50-500+ submission)")
    print("   - Images normalized inline (1024px max, 90% JPEG)")
    print("   - Expected time: 2-3 hours")
    print("   - Progress saved automatically\n")
    
    input("Press Enter to start...")
    
    try:
        all_data = collector.collect_all()
        
        if all_data:
            collector.save_dataset(all_data)
            
            print("\n" + "="*70)
            print("✅ Collection Complete!")
            print("="*70 + "\n")
            
            print("📁 Data saved to: data/raw/")
            print(f"   {len(all_data)} samples collected")
            print(f"   Images normalized and optimized")
            print("")
            print("🎯 Next step:")
            print("   python tools/clean_and_convert.py")
            print("")
        else:
            print("\n⚠️  No data collected.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted - progress saved")
        print("Run again to resume!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
