"""
Modal serverless deployment for DissTrack Roast Model
API-compatible with RunPod interface for drop-in replacement

Model: jasonlevy/roastme-model-v1 (Qwen2.5-VL-7B fine-tuned on r/RoastMe)
Cost: ~$0.00376/sec when running (H100 GPU)
Cold start: ~5-10 seconds (H100 is fast!)
Endpoints:
  POST /upload          → {imageId}
  POST /generate-batch  → {candidates: string[], count: number}
"""

import io
import modal
import uuid
import time
from pathlib import Path
from datetime import datetime

# ==========================================
# Configuration
# ==========================================

MODEL_ID = "jasonlevy/roastme-model-v1"
GPU_CONFIG = "H100"
SCALEDOWN_WINDOW = 300
CACHE_DIR = "/cache"

# ==========================================
# Docker Image Setup
# ==========================================

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers==4.56.1",
        "torch==2.8.0",
        "torchvision==0.23.0",
        "qwen-vl-utils",
        "Pillow",
        "accelerate",
        "huggingface_hub",
        "hf-transfer",
        "fastapi[standard]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

cache_volume = modal.Volume.from_name("roastme-model-cache", create_if_missing=True)

app = modal.App(
    "disstrack-roast",
    image=image,
)

# ==========================================
# System Message & Prompt (MATCHES TRAINING DATA)
# ==========================================

SYSTEM_MESSAGE = """You are a sharp-witted roast comedian performing in a consensual roast battle inspired by Reddit's r/RoastMe. Generate one savage, hilarious roast based ONLY on what you see in the photo.

Rules:
- Attack specific visual details: appearance, expression, outfit, setting
- Use clever comparisons and pop culture references  
- Keep it punchy: 1-2 sentences, under 100 characters ideal
- Be brutally funny - no holding back
- No hedging, no apologies, just pure roast

Deliver a devastating one-liner like a standup comedian."""

USER_PROMPT = "Roast this person based on their appearance."

# ==========================================
# Model Class
# ==========================================

@app.cls(
    gpu=GPU_CONFIG,
    timeout=300,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={CACHE_DIR: cache_volume},
)
class RoastModel:
    """
    Serverless roast model inference with image caching
    """
    
    @modal.enter()
    def load_model(self):
        """Load model on container startup (runs once per container)"""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch
        
        self.image_cache = {}
        
        print(f"🔄 Loading model: {MODEL_ID}")
        start_time = time.time()
        
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")

    @modal.method()
    def upload_image(self, image_base64: str) -> str:
        """
        Upload and cache an image, return image_id
        
        Args:
            image_base64: Base64-encoded image string
            
        Returns:
            image_id: UUID for the cached image
        """
        import base64
        from PIL import Image
        
        image_id = str(uuid.uuid4())
        
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        self.image_cache[image_id] = {
            "image": image,
            "created_at": datetime.now().isoformat()
        }
        
        print(f"📤 Uploaded image: {image_id} (cache size: {len(self.image_cache)})")
        
        return image_id
    
    @modal.method()
    def generate_batch(
        self,
        image_id: str,
        num_candidates: int = 3,
        temperature: float = 0.85,      # Tuned for creative roasts
        top_p: float = 0.9,             # Allow diverse vocabulary
        top_k: int = 50,                # Reasonable sampling pool
        max_new_tokens: int = 80        # ~50-100 chars typical
    ) -> dict:
        """
        Generate multiple roasts for a cached image
        
        Args:
            image_id: UUID from upload_image
            num_candidates: Number of roasts to generate (default: 3)
            temperature: Sampling temperature (default: 0.85)
            top_p: Nucleus sampling threshold (default: 0.9)
            top_k: Top-k sampling (default: 50)
            max_new_tokens: Max tokens per roast (default: 80)
            
        Returns:
            dict with 'candidates' list and 'count'
        """
        from qwen_vl_utils import process_vision_info
        import torch
        import random
        
        if image_id not in self.image_cache:
            raise ValueError(f"Image ID {image_id} not found in cache")
        
        image = self.image_cache[image_id]["image"]
        
        print(f"🎯 Generating {num_candidates} roasts for image {image_id}")
        start_time = time.time()
        
        candidates = []
        
        for i in range(num_candidates):
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_MESSAGE}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": USER_PROMPT}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            # Different seed per candidate for variety
            seed = random.randint(0, 1000000)
            torch.manual_seed(seed)
            
            # Generation parameters tuned for ~71 char roasts
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                repetition_penalty=1.2,     # Prevent repetition
                no_repeat_ngram_size=2      # Block 2-gram repetition
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            candidates.append(output_text)
            print(f"  🔥 Candidate {i+1}/{num_candidates}: {output_text[:80]}...")
        
        inference_time = time.time() - start_time
        print(f"✅ Generated {len(candidates)} roasts in {inference_time:.2f}s")
        
        return {
            "candidates": candidates,
            "count": len(candidates),
            "inference_time_seconds": round(inference_time, 2),
            "model": MODEL_ID
        }

# ==========================================
# REST API Endpoints (RunPod-compatible)
# ==========================================

@app.function()
@modal.asgi_app()
def upload():
    """
    Upload image and get image_id
    
    POST /upload
    Body: {"imageBase64": "..."}
    Returns: {"imageId": "uuid"}
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    web_app = FastAPI()
    
    class UploadRequest(BaseModel):
        imageBase64: str
    
    @web_app.post("/")
    async def handle_upload(request: UploadRequest):
        if not request.imageBase64:
            raise HTTPException(status_code=400, detail="Missing imageBase64")
        
        model = RoastModel()
        image_id = model.upload_image.remote(request.imageBase64)
        
        return {"imageId": image_id}
    
    return web_app

@app.function()
@modal.asgi_app()
def generate_batch():
    """
    Generate multiple roasts for a cached image
    
    POST /generate-batch
    Body: {"imageId": "uuid", "numCandidates": 3}
    Returns: {"candidates": [...], "count": 3}
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    web_app = FastAPI()
    
    class GenerateRequest(BaseModel):
        imageId: str
        numCandidates: int = 3
    
    @web_app.post("/")
    async def handle_generate(request: GenerateRequest):
        if not request.imageId:
            raise HTTPException(status_code=400, detail="Missing imageId")
        
        try:
            model = RoastModel()
            result = model.generate_batch.remote(
                image_id=request.imageId,
                num_candidates=request.numCandidates
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    return web_app

@app.function()
@modal.asgi_app()
def generate():
    """
    Generate a single roast (legacy endpoint)
    
    POST /generate
    Body: {"imageId": "uuid"}
    Returns: {"roast": "..."}
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    web_app = FastAPI()
    
    class GenerateRequest(BaseModel):
        imageId: str
    
    @web_app.post("/")
    async def handle_generate(request: GenerateRequest):
        if not request.imageId:
            raise HTTPException(status_code=400, detail="Missing imageId")
        
        try:
            model = RoastModel()
            result = model.generate_batch.remote(
                image_id=request.imageId,
                num_candidates=1
            )
            return {
                "roast": result["candidates"][0],
                "inference_time_seconds": result["inference_time_seconds"],
                "model": result["model"]
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    return web_app

# ==========================================
# Local Testing
# ==========================================

@app.local_entrypoint()
def test():
    """
    Test the model locally with a sample image
    
    Usage: modal run deployment/modal_inference.py
    """
    import base64
    
    test_image_path = input("Enter path to test image (or press Enter to skip): ").strip()
    
    if not test_image_path or not Path(test_image_path).exists():
        print("❌ No valid image provided. Skipping test.")
        return
    
    with open(test_image_path, "rb") as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    print(f"\n🔥 Testing with: {test_image_path}\n")
    
    model = RoastModel()
    print("📤 Uploading image...")
    image_id = model.upload_image.remote(image_base64)
    print(f"✅ Image uploaded: {image_id}\n")
    
    print("🎯 Generating 3 roasts...")
    result = model.generate_batch.remote(image_id=image_id, num_candidates=3)
    
    print(f"\n{'='*70}")
    print("🎤 Generated Roasts:")
    print(f"{'='*70}\n")
    
    for i, roast in enumerate(result["candidates"], 1):
        print(f"{i}. {roast}\n")
    
    print(f"{'='*70}")
    print(f"⏱️  Total time: {result['inference_time_seconds']}s")
    print(f"📊 Model: {result['model']}")
    print(f"{'='*70}\n")
