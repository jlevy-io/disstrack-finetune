#!/bin/bash
# DissTrack Fine-tuning Script
# Run on RunPod with GPU
# Based on Qwen2-VL-Finetune proven training pipeline

set -e  # Exit on error

echo "=========================================="
echo "🔥 DissTrack Roast Model Training"
echo "=========================================="
echo ""

# ==========================================
# Configuration
# ==========================================

MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH="data/llava_format/train.json"
IMAGE_FOLDER="data/raw/images"
OUTPUT_DIR="outputs/qwen2.5-vl-roastme-v4"

# Image resolution (balance quality vs memory)
MIN_PIXELS=$((256 * 28 * 28))    # 200K tokens
MAX_PIXELS=$((1024 * 28 * 28))   # 802K tokens

echo "📊 Training Configuration:"
echo "   Model: $MODEL_ID"
echo "   Data: $DATA_PATH"
echo "   Images: $IMAGE_FOLDER"
echo "   Output: $OUTPUT_DIR"
echo ""
echo "   Image resolution: ${MIN_PIXELS} - ${MAX_PIXELS} pixels"
echo ""
echo "🚀 Starting training..."
echo ""

# ==========================================
# Training with DeepSpeed
# ==========================================

deepspeed src/train/train_sft.py \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_ID \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --output_dir $OUTPUT_DIR \
    \
    `# Training Schedule` \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    \
    `# Layer Freezing - CRITICAL: Train projector for visual grounding!` \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    \
    `# Learning Rates - Different rates for different components` \
    --learning_rate 1e-4 \
    --merger_lr 5e-5 \
    --vision_lr 2e-5 \
    \
    `# Image Processing` \
    --image_min_pixels $MIN_PIXELS \
    --image_max_pixels $MAX_PIXELS \
    \
    `# Optimization` \
    --bf16 True \
    --use_liger True \
    \
    `# LoRA Configuration` \
    --lora_enable True \
    --vision_lora False \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    \
    `# Training Settings` \
    --max_seq_length 2048 \
    --dataloader_num_workers 4 \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --report_to tensorboard \
    --logging_dir $OUTPUT_DIR/logs

echo ""
echo "=========================================="
echo "✅ Training Complete!"
echo "=========================================="
echo ""
echo "📁 Model saved to: $OUTPUT_DIR"
echo ""
echo "🎯 Next steps:"
echo "   1. Merge LoRA weights: bash scripts/merge_lora_roastme.sh"
echo "   2. Test the model"
echo "   3. Deploy to production"
echo ""
