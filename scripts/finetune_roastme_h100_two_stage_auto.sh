#!/bin/bash
set -e

echo "=========================================="
echo "🔥 DissTrack H100 - Two-Stage Training (Auto)"
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

# Options
MERGE_WEIGHTS=true
SKIP_STAGE1=false  # Set to true if you want to resume from stage 1
SKIP_STAGE2=false

export PYTHONPATH=src:$PYTHONPATH
unset LD_LIBRARY_PATH

# ==========================================
# Helper Functions
# ==========================================

log_section() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

clear_gpu() {
    echo "🧹 Clearing GPU memory..."
    pkill -9 python3 || true
    sleep 5
}

# ==========================================
# Stage 1: Train Merger + LoRA
# ==========================================

if [ "$SKIP_STAGE1" = false ]; then
    log_section "📍 STAGE 1: Training Merger + LoRA"
    
    echo "Starting in 3 seconds..."
    sleep 3
    
    deepspeed src/train/train_sft.py \
        --deepspeed scripts/zero2.json \
        --model_id $BASE_MODEL \
        --data_path $DATA_PATH \
        --image_folder $IMAGE_FOLDER \
        --output_dir $STAGE1_DIR \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --freeze_vision_tower True \
        --freeze_llm True \
        --freeze_merger False \
        --learning_rate 1e-4 \
        --merger_lr 5e-5 \
        --image_min_pixels $((256 * 28 * 28)) \
        --image_max_pixels $((768 * 28 * 28)) \
        --bf16 True \
        --use_liger True \
        --lora_enable True \
        --lora_rank 128 \
        --lora_alpha 256 \
        --lora_dropout 0.05 \
        --gradient_checkpointing True \
        --max_seq_length 2048 \
        --dataloader_num_workers 4 \
        --logging_steps 5 \
        --save_strategy "steps" \
        --save_steps 50 \
        --save_total_limit 2 \
        --report_to tensorboard \
        --logging_dir $STAGE1_DIR/logs
    
    log_section "✅ STAGE 1 COMPLETE!"
    clear_gpu
else
    log_section "⏭️  SKIPPING STAGE 1 (using existing checkpoint)"
fi

# ==========================================
# Stage 2: Fine-tune Vision
# ==========================================

if [ "$SKIP_STAGE2" = false ]; then
    log_section "📍 STAGE 2: Fine-tuning Vision Tower"
    
    echo "Starting in 3 seconds..."
    sleep 3
    
    deepspeed src/train/train_sft.py \
        --deepspeed scripts/zero2.json \
        --model_id $STAGE1_DIR \
        --data_path $DATA_PATH \
        --image_folder $IMAGE_FOLDER \
        --output_dir $STAGE2_DIR \
        --num_train_epochs 2 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --freeze_vision_tower False \
        --freeze_llm True \
        --freeze_merger True \
        --learning_rate 5e-5 \
        --vision_lr 2e-5 \
        --image_min_pixels $((128 * 28 * 28)) \
        --image_max_pixels $((512 * 28 * 28)) \
        --bf16 True \
        --use_liger True \
        --lora_enable True \
        --lora_rank 64 \
        --lora_alpha 128 \
        --lora_dropout 0.05 \
        --gradient_checkpointing True \
        --max_seq_length 2048 \
        --dataloader_num_workers 4 \
        --logging_steps 5 \
        --save_strategy "steps" \
        --save_steps 50 \
        --save_total_limit 2 \
        --report_to tensorboard \
        --logging_dir $STAGE2_DIR/logs
    
    log_section "✅ STAGE 2 COMPLETE!"
    clear_gpu
else
    log_section "⏭️  SKIPPING STAGE 2 (using existing checkpoint)"
fi

# ==========================================
# Merge LoRA Weights
# ==========================================

if [ "$MERGE_WEIGHTS" = true ]; then
    log_section "🔧 Merging LoRA Weights"
    
    python src/merge_lora_weights.py \
        --model-path $STAGE2_DIR \
        --model-base $BASE_MODEL \
        --save-model-path $FINAL_DIR \
        --safe-serialization
    
    log_section "✅ MERGE COMPLETE!"
fi

# ==========================================
# Final Summary
# ==========================================

log_section "🎉 ALL STAGES COMPLETE!"

echo "📦 Output Directories:"
echo "   Stage 1: $STAGE1_DIR"
echo "   Stage 2: $STAGE2_DIR"
if [ "$MERGE_WEIGHTS" = true ]; then
    echo "   Merged:  $FINAL_DIR"
fi
echo ""
echo "🎯 Test your model:"
if [ "$MERGE_WEIGHTS" = true ]; then
    echo "   python -m src.serve.app --model-path $FINAL_DIR"
else
    echo "   python -m src.serve.app --model-path $STAGE2_DIR"
fi
echo ""
