#!/bin/bash
set -e

echo "=========================================="
echo "🔥 DissTrack H100 - Two-Stage Training"
echo "=========================================="
echo ""

# ==========================================
# Configuration
# ==========================================

BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH="data/llava_format/train.json"
IMAGE_FOLDER="data/raw/images"
OUTPUT_BASE="outputs/qwen2.5-vl-roastme-v4"

STAGE1_DIR="${OUTPUT_BASE}-stage1"
STAGE2_DIR="${OUTPUT_BASE}-stage2"
FINAL_DIR="${OUTPUT_BASE}-merged"

export PYTHONPATH=src:$PYTHONPATH
unset LD_LIBRARY_PATH

# ==========================================
# Stage 1: Train Merger + LoRA (Vision Frozen)
# ==========================================

echo "=========================================="
echo "📍 STAGE 1: Training Merger + LoRA"
echo "=========================================="
echo ""
echo "Configuration:"
echo "   Vision Tower: FROZEN"
echo "   LLM: FROZEN + LoRA (rank 128)"
echo "   Merger: TRAINING"
echo "   Batch Size: 4"
echo "   Image Resolution: 256-768 tokens"
echo "   Expected Memory: ~35-40GB"
echo "   Expected Time: ~8-10 minutes"
echo ""
read -p "Press Enter to start Stage 1..."
echo ""

deepspeed src/train/train_sft.py \
    --deepspeed scripts/zero2.json \
    --model_id $BASE_MODEL \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --output_dir $STAGE1_DIR \
    \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    \
    --learning_rate 1e-4 \
    --merger_lr 5e-5 \
    \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((768 * 28 * 28)) \
    \
    --bf16 True \
    --use_liger True \
    \
    --lora_enable True \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    \
    --gradient_checkpointing True \
    \
    --max_seq_length 2048 \
    --dataloader_num_workers 4 \
    --logging_steps 5 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --report_to tensorboard \
    --logging_dir $STAGE1_DIR/logs

echo ""
echo "=========================================="
echo "✅ STAGE 1 COMPLETE!"
echo "=========================================="
echo ""
echo "📁 Stage 1 checkpoint: $STAGE1_DIR"
echo ""

# Clean GPU memory
pkill -9 python3 || true
sleep 5

# ==========================================
# Stage 2: Fine-tune Vision Tower
# ==========================================

echo "=========================================="
echo "📍 STAGE 2: Fine-tuning Vision Tower"
echo "=========================================="
echo ""
echo "Configuration:"
echo "   Vision Tower: TRAINING"
echo "   LLM: FROZEN + LoRA (rank 64)"
echo "   Merger: FROZEN"
echo "   Batch Size: 1"
echo "   Image Resolution: 128-512 tokens"
echo "   Expected Memory: ~50-60GB"
echo "   Expected Time: ~8-10 minutes"
echo ""
read -p "Press Enter to start Stage 2..."
echo ""

deepspeed src/train/train_sft.py \
    --deepspeed scripts/zero2.json \
    --model_id $STAGE1_DIR \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --output_dir $STAGE2_DIR \
    \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger True \
    \
    --learning_rate 5e-5 \
    --vision_lr 2e-5 \
    \
    --image_min_pixels $((128 * 28 * 28)) \
    --image_max_pixels $((512 * 28 * 28)) \
    \
    --bf16 True \
    --use_liger True \
    \
    --lora_enable True \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    \
    --gradient_checkpointing True \
    \
    --max_seq_length 2048 \
    --dataloader_num_workers 4 \
    --logging_steps 5 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --report_to tensorboard \
    --logging_dir $STAGE2_DIR/logs

echo ""
echo "=========================================="
echo "✅ STAGE 2 COMPLETE!"
echo "=========================================="
echo ""
echo "📁 Stage 2 checkpoint: $STAGE2_DIR"
echo ""

# ==========================================
# Optional: Merge LoRA Weights
# ==========================================

echo "=========================================="
echo "🔧 MERGE LORA WEIGHTS?"
echo "=========================================="
echo ""
echo "Would you like to merge LoRA weights now?"
echo "This will create a standalone model without LoRA adapters."
echo ""
read -p "Merge now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Merging LoRA weights..."
    echo ""
    
    python src/merge_lora_weights.py \
        --model-path $STAGE2_DIR \
        --model-base $BASE_MODEL \
        --save-model-path $FINAL_DIR \
        --safe-serialization
    
    echo ""
    echo "=========================================="
    echo "✅ MERGE COMPLETE!"
    echo "=========================================="
    echo ""
    echo "📁 Final merged model: $FINAL_DIR"
    echo ""
fi

# ==========================================
# Summary
# ==========================================

echo "=========================================="
echo "🎉 TRAINING COMPLETE!"
echo "=========================================="
echo ""
echo "📦 Output Directories:"
echo "   Stage 1 (Merger + LoRA):     $STAGE1_DIR"
echo "   Stage 2 (Vision Fine-tuned): $STAGE2_DIR"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Final Merged Model:          $FINAL_DIR"
fi
echo ""
echo "🎯 Next Steps:"
echo "   1. Test the model:"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "      python -m src.serve.app --model-path $FINAL_DIR"
else
    echo "      python -m src.serve.app --model-path $STAGE2_DIR"
fi
echo ""
echo "   2. Monitor training logs:"
echo "      tensorboard --logdir $OUTPUT_BASE-stage1/logs"
echo "      tensorboard --logdir $OUTPUT_BASE-stage2/logs"
echo ""
echo "   3. Compare checkpoints to find the best one"
echo ""
