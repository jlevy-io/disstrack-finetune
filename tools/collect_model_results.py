"""
tools/collect_model_results.py

Collect roast generation results from Modal deployment for evaluation
"""

import asyncio
import base64
import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import httpx
from tqdm.asyncio import tqdm


class ModelDataCollector:
    def __init__(
        self,
        upload_url: str,
        generate_batch_url: str,
        num_candidates: int = 5
    ):
        self.upload_url = upload_url
        self.generate_batch_url = generate_batch_url
        self.num_candidates = num_candidates
        self.client = httpx.AsyncClient(timeout=120.0)

    async def test_single_image(self, image_path: Path, ground_truth: str) -> Dict:
        """Test a single image and collect results"""
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        try:
            # Upload image
            upload_response = await self.client.post(
                self.upload_url,
                json={"imageBase64": image_base64}
            )
            upload_response.raise_for_status()
            image_id = upload_response.json()["imageId"]

            # Generate roasts
            generate_response = await self.client.post(
                self.generate_batch_url,
                json={
                    "imageId": image_id,
                    "numCandidates": self.num_candidates
                }
            )
            generate_response.raise_for_status()
            result = generate_response.json()

            return {
                "image_filename": image_path.name,
                "image_id": image_path.stem,
                "candidates": result["candidates"],
                "inference_time_seconds": result["inference_time_seconds"],
                "model": result["model"],
                "ground_truth": ground_truth,
                "success": True,
                "error": None
            }

        except Exception as e:
            return {
                "image_filename": image_path.name,
                "image_id": image_path.stem,
                "candidates": [],
                "inference_time_seconds": 0,
                "model": "unknown",
                "ground_truth": ground_truth,
                "success": False,
                "error": str(e)
            }

    async def collect_results(
        self,
        test_images: List[tuple],
        model_name: str
    ) -> Dict:
        """Collect results for all test images"""
        results = []
        failed = 0

        print(f"\n{'='*70}")
        print(f"🔥 Collecting Results: {model_name}")
        print(f"{'='*70}\n")
        print(f"Images: {len(test_images)}")
        print(f"Candidates per image: {self.num_candidates}")
        print(f"Total roasts to generate: {len(test_images) * self.num_candidates}\n")

        for image_path, ground_truth in tqdm(test_images, desc="Processing"):
            result = await self.test_single_image(image_path, ground_truth)
            results.append(result)

            if not result["success"]:
                failed += 1
                tqdm.write(f"❌ Failed: {image_path.name} - {result['error']}")
            else:
                tqdm.write(f"✅ {image_path.name}: {len(result['candidates'])} roasts in {result['inference_time_seconds']:.2f}s")

            # Brief pause between requests
            await asyncio.sleep(0.5)

        # Calculate stats
        successful_results = [r for r in results if r["success"]]
        total_roasts = sum(len(r["candidates"]) for r in successful_results)
        total_time = sum(r["inference_time_seconds"] for r in successful_results)
        avg_time = total_time / len(successful_results) if successful_results else 0

        stats = {
            "total_images": len(test_images),
            "successful": len(successful_results),
            "failed": failed,
            "total_roasts_generated": total_roasts,
            "total_inference_time": total_time,
            "avg_time_per_image": avg_time
        }

        print(f"\n{'='*70}")
        print(f"📊 Collection Summary")
        print(f"{'='*70}\n")
        print(f"✅ Successful: {stats['successful']}/{stats['total_images']}")
        if failed > 0:
            print(f"❌ Failed: {failed}")
        print(f"🔥 Total roasts: {stats['total_roasts_generated']}")
        print(f"⏱️  Total time: {stats['total_inference_time']:.2f}s")
        print(f"📈 Avg per image: {stats['avg_time_per_image']:.2f}s\n")

        return {
            "model_name": model_name,
            "collection_timestamp": datetime.now().isoformat(),
            "config": {
                "upload_url": self.upload_url,
                "generate_batch_url": self.generate_batch_url,
                "num_candidates": self.num_candidates
            },
            "stats": stats,
            "results": results
        }

    async def close(self):
        await self.client.aclose()


def load_validation_set(num_images: int = 30) -> List[tuple]:
    """Load validation set and sample images"""
    val_file = Path("data/llava_format/val.json")
    image_dir = Path("data/raw/images")

    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    with open(val_file) as f:
        val_data = json.load(f)

    # Sample images
    sampled = random.sample(val_data, min(num_images, len(val_data)))

    # Build list of (image_path, ground_truth) tuples
    test_images = []
    for item in sampled:
        image_path = image_dir / item["image"]
        if image_path.exists():
            ground_truth = item["conversations"][1]["value"]
            test_images.append((image_path, ground_truth))

    print(f"\n✅ Loaded {len(test_images)} validation images")
    return test_images


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect model results from Modal deployment"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model version name (e.g., 'v1' or 'v2')"
    )
    parser.add_argument(
        "--upload-url",
        default="https://jlevy-io--disstrack-roast-upload.modal.run",
        help="Modal upload endpoint URL"
    )
    parser.add_argument(
        "--generate-url",
        default="https://jlevy-io--disstrack-roast-generate-batch.modal.run",
        help="Modal generate-batch endpoint URL"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=30,
        help="Number of validation images to test"
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=5,
        help="Number of roasts per image"
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Output directory"
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"🔥 DissTrack Model Evaluation - Data Collection")
    print(f"{'='*70}\n")
    print(f"Model: {args.model_name}")
    print(f"Images: {args.num_images}")
    print(f"Candidates per image: {args.num_candidates}\n")

    # Load test images (fixed seed for reproducibility)
    random.seed(42)
    test_images = load_validation_set(args.num_images)

    # Collect results
    collector = ModelDataCollector(
        upload_url=args.upload_url,
        generate_batch_url=args.generate_url,
        num_candidates=args.num_candidates
    )

    try:
        data = await collector.collect_results(test_images, args.model_name)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{args.model_name}_results_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"{'='*70}")
        print(f"✅ Results saved to: {output_file}")
        print(f"{'='*70}\n")

        print("📋 Next Steps:")
        if args.model_name == "v2":
            print("   1. Update deployment/modal_inference.py:")
            print('      Change: MODEL_ID = "jasonlevy/roastme-model-v1"')
            print("   2. Redeploy: modal deploy deployment/modal_inference.py")
            print("   3. Run collection for v1:")
            print(f"      python tools/collect_model_results.py --model-name v1")
        elif args.model_name == "v1":
            print("   1. Run comparison:")
            print(f"      python tools/compare_results.py \\")
            print(f"        --v2-results {output_dir}/v2_results_*.json \\")
            print(f"        --v1-results {output_file}")
        print()

    finally:
        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())
