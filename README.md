# DissTrack Fine-Tuning

Fine-tuning Qwen2.5-VL for savage, visually-grounded comedy roasting.

Built on [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune) by 2U1.

---

## 🎯 Project Overview

**Goal:** Train a vision-language model to generate funny, savage roasts based on people's appearance (consensual, r/RoastMe style).

**Key Challenge Solved:** Base Qwen2-VL is too polite. Fine-tuning on Reddit data creates hallucinations. Our solution: **Filter for visually-grounded roasts only** + **train the multimodal projection layer**.

**Dataset:** 1,270 visually-grounded roasts from r/RoastMe (filtered from 2,607 original samples).

---

## 🏗️ Architecture

```
Local Machine (VS Code)          Cloud (RunPod/Lambda)
────────────────────────         ─────────────────────
1. Edit code                →    4. Pull repo
2. Process data             →    5. Train model  
3. Commit & push            ←    6. Push results
```

**Why This Works:**
- ✅ Edit code comfortably in VS Code locally
- ✅ No GPU needed for data processing
- ✅ Train on powerful cloud GPUs only when needed
- ✅ Version control everything with Git

---

## 📁 Project Structure

```
disstrack-finetune/
├── data/                      # Your dataset (not in git)
│   ├── raw/                   # Original r/RoastMe data
│   │   ├── images/            # 1,270 images (~500MB)
│   │   └── training_data.jsonl
│   ├── processed/             # Filtered for visual grounding
│   │   └── v4_visual/
│   └── llava_format/          # Converted for training
│
├── tools/                     # Data processing (run locally)
│   ├── filter_visual.py       # Filter for visual grounding
│   ├── convert_to_llava.py    # Format converter
│   └── analyze_dataset.py     # Quality checks
│
├── scripts/                   # Training configs (run on cloud)
│   ├── finetune_roastme.sh    # Your custom training script
│   ├── merge_lora_roastme.sh  # Merge LoRA after training
│   └── zero2.json             # DeepSpeed config (from original repo)
│
├── src/                       # Training code (from original repo)
│   ├── train/                 # Training logic
│   ├── model/                 # Model definitions
│   └── dataset/               # Dataset loaders
│
├── deployment/                # Inference (future)
│   └── modal_inference.py     # Serverless deployment
│
└── outputs/                   # Trained models (not in git)
    └── qwen2.5-vl-roastme-v4/
```

---

## 🚀 Quick Start

### 1️⃣ Local Setup (One-Time)

```bash
# Clone YOUR fork
git clone https://github.com/YOUR_USERNAME/disstrack-finetune.git
cd disstrack-finetune

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install local dependencies (data processing only)
pip install -r requirements-local.txt
```

### 2️⃣ Copy Your Existing Data

```bash
# Copy from old disstrack-ml repo
cp -r /path/to/disstrack-ml/datasets/roastme/images/* data/raw/images/
cp /path/to/disstrack-ml/datasets/roastme/training_data.jsonl data/raw/

# Note: These files are .gitignored (too large for git)
```

### 3️⃣ Process Data (Locally)

```bash
# Filter for visual grounding (removes hallucination-prone samples)
python tools/filter_visual.py

# Convert to LLaVA format for training
python tools/convert_to_llava.py

# Optional: Analyze dataset quality
python tools/analyze_dataset.py
```

### 4️⃣ Commit & Push

```bash
git add tools/ scripts/ deployment/ README.md
git commit -m "Add DissTrack training pipeline"
git push origin main
```

### 5️⃣ Train on Cloud

**On RunPod/Lambda Labs:**

```bash
# Clone YOUR repo
git clone https://github.com/YOUR_USERNAME/disstrack-finetune.git
cd disstrack-finetune

# Install training dependencies
pip install -r requirements.txt
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation

# Copy your data (from network storage or upload)
# data/raw/images/ should contain all images
# data/llava_format/train.json should exist

# Train! (~25-30 minutes on A40)
bash scripts/finetune_roastme.sh

# Merge LoRA weights
bash scripts/merge_lora_roastme.sh
```

---

## 📊 Training Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| **Model** | Qwen2.5-VL-7B-Instruct | Latest, best vision-language model |
| **Method** | LoRA (rank 32) | Memory efficient, 4x faster than full fine-tune |
| **Epochs** | 5 | Small dataset needs more passes |
| **LLM LR** | 1e-4 | Standard for language model |
| **Projector LR** | 5e-5 | **CRITICAL: Trains visual grounding!** |
| **Vision LR** | 2e-5 | Small updates to vision encoder |
| **Batch Size** | 2 × 8 accumulation = 16 effective | Fits in A40 48GB |
| **Training Time** | ~25-30 minutes on A40 | ~$0.30 per run |

**Key Innovation:** We train the **merger (projector)** layer that connects vision → text. This fixes visual grounding issues that caused hallucinations in previous attempts.

---

## 🔍 Why This Approach Works

### Problem 1: Base Model Too Polite
**Solution:** Fine-tune on r/RoastMe data

### Problem 2: Model Hallucinates After Fine-Tuning
**Root Cause:** 51% of training data was non-visual (generic insults)
**Solution:** Filter dataset to keep only visually-grounded roasts (48.7% → 100%)

### Problem 3: Model Still Hallucinates
**Root Cause:** Projector layer was frozen (only trained LLM attention)
**Solution:** Train projector with separate learning rate (5e-5)

---

## 📝 Daily Development Workflow

### Making Changes Locally

```bash
# Edit code in VS Code
# ... make changes to tools/scripts ...

# Test locally (if applicable)
python tools/analyze_dataset.py

# Commit & push
git add .
git commit -m "Update filtering logic"
git push origin main
```

### Training on Cloud

```bash
# Pull latest changes
git pull origin main

# Run training
bash scripts/finetune_roastme.sh

# Optional: Commit training logs/metrics
git add outputs/qwen2.5-vl-roastme-v4/training_log.txt
git commit -m "Training run results"
git push origin main
```

---

## 🎓 Dataset Details

**Source:** r/RoastMe subreddit top posts

**Collection Criteria:**
- ✅ Submission score ≥ 100
- ✅ ≥ 10 comments
- ✅ Valid image URL

**Filtering (Visual Grounding):**
- ✅ Contains ≥2 visual keywords (hair, face, glasses, outfit, etc.)
- ❌ Generic insults without visual references
- ❌ Reddit meta-references (post history, awards, etc.)

**Final Stats:**
- Train: ~1,143 samples
- Val: ~127 samples
- Visual grounding: 100%

**Example Good Roast (Visually Grounded):**
> "With that nose you could come out of the shower and your body would still be dry."

**Example Bad Roast (Removed):**
> "I bet your parents were disappointed when you came out." ❌ (No visual reference)

---

## 🔄 Pulling Upstream Updates

To get improvements from the original Qwen2-VL-Finetune repo:

```bash
git fetch upstream
git merge upstream/main
# Resolve any conflicts
git push origin main
```

---

## 🚧 Roadmap

- [x] Data collection from r/RoastMe
- [x] Visual grounding filter
- [x] LLaVA format conversion
- [x] Training script with proper projector training
- [ ] Successful training run
- [ ] Model evaluation
- [ ] Modal serverless deployment
- [ ] Production API integration

---

## 📚 Resources

- [Original Qwen2-VL-Finetune Repo](https://github.com/2U1/Qwen2-VL-Finetune)
- [Qwen2.5-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [r/RoastMe Subreddit](https://reddit.com/r/RoastMe)

---

## 🙏 Credits

- **Training Framework:** [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune) by 2U1
- **Base Model:** Qwen2.5-VL by Alibaba Qwen Team
- **Dataset:** r/RoastMe community
- **Project:** DissTrack by Jason Levy

---

## 📄 License

This project inherits the Apache 2.0 License from Qwen2-VL-Finetune.

---

## 🐛 Troubleshooting

**Issue:** `libcudnn error` during training
**Solution:** `unset LD_LIBRARY_PATH` before training

**Issue:** Out of memory during training
**Solution:** Reduce `per_device_train_batch_size` to 1 in `scripts/finetune_roastme.sh`

**Issue:** Data file not found
**Solution:** Make sure you've run `python tools/filter_visual.py` and `python tools/convert_to_llava.py` first

---

**Ready to train?** Follow the Quick Start guide above! 🚀