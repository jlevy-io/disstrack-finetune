#!/bin/bash
set -e

echo "=========================================="
echo "🔥 DissTrack H100 - LoRA-128 (Memory Optimized)"
echo "=========================================="
echo ""

# ==========================================
# Configuration
# ==========================================

MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH="data/llava_format/train.json"
IMAGE_FOLDER="data/raw/images"
OUTPUT_DIR="outputs/qwen2.5-vl-roastme-v4-h100-lora128"

# Image resolution - LOWERED to fit in memory with vision training
MIN_PIXELS=$((256 * 28 * 28))    # 200K tokens (balanced for memory)
MAX_PIXELS=$((768 * 28 * 28))    # 602K tokens (still good quality)

# ==========================================
# Setup Environment
# ==========================================

export PYTHONPATH=src:$PYTHONPATH
unset LD_LIBRARY_PATH

echo "📊 Training Configuration:"
echo "   Hardware: H100 SXM 80GB"
echo "   Strategy: High-Rank LoRA (rank 128)"
echo ""
echo "   Training Strategy:"
echo "     LLM: Frozen + LoRA adapters (rank 128 = 4x capacity)"
echo "     Vision: FULL training (learns facial features)"
echo "     Merger: FULL training (visual grounding!)"
echo ""
echo "   Image Resolution: 256-768 tokens (optimized for memory)"
echo ""
echo "   Performance:"
echo "     Memory: ~30-35GB (comfortable fit)"
echo "     Time: ~12-15 minutes"
echo "     Cost: ~$0.67"
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
    --num_train_epochs 5 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 3 \
    \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    \
    --learning_rate 1e-4 \
    --merger_lr 5e-5 \
    --vision_lr 2e-5 \
    \
    --image_min_pixels $MIN_PIXELS \
    --image_max_pixels $MAX_PIXELS \
    \
    --bf16 True \
    --use_liger True \
    \
    --lora_enable True \
    --vision_lora False \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    \
    --gradient_checkpointing False \
    \
    --max_seq_length 2048 \
    --dataloader_num_workers 8 \
    --logging_steps 5 \
    --save_strategy "steps" \
    --save_steps 25 \
    --save_total_limit 3 \
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
echo "   1. Merge LoRA weights: bash scripts/merge_lora_h100.sh"
echo "   2. Test the model"
echo "   3. Deploy to production"
echo ""
