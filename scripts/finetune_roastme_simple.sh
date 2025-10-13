#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

log_section "🔥 DissTrack - Simple Single-Stage Training"

# Config for this experiment
EXPERIMENT_NAME="roastme-simple-v1"
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"
DATA_PATH="data/llava_format/train.json"
VAL_PATH="data/llava_format/val.json"

log_info "Configuration:"
echo "   Model: $BASE_MODEL" >&2
echo "   Training samples: 693" >&2
echo "   Validation samples: 78" >&2
echo "   Output: $OUTPUT_DIR" >&2
echo "" >&2

log_section "📋 TRAINING STRATEGY"
echo "   Vision Tower: FROZEN (already excellent)" >&2
echo "   LLM: FROZEN + LoRA (rank 64)" >&2
echo "   Merger: TRAINABLE (light adaptation)" >&2
echo "   Epochs: 1 (with early stopping via validation)" >&2
echo "   Batch Strategy: Conservative for fast iteration" >&2
echo "" >&2

# Check disk space
check_disk_space 20 || {
    log_warning "Low disk space detected"
    read -p "Continue anyway? (y/n): " -n 1 -r >&2
    echo "" >&2
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 0
}

echo "" >&2
read -p "Press Enter to start training..." >&2
echo "" >&2

# Batch size calculation - conservative for H100
GLOBAL_BATCH_SIZE=32
PER_DEVICE_BATCH=2
NUM_DEVICES=1
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (PER_DEVICE_BATCH * NUM_DEVICES)))

log_info "Batch configuration:"
echo "   Global batch size: $GLOBAL_BATCH_SIZE" >&2
echo "   Per-device batch: $PER_DEVICE_BATCH" >&2
echo "   Gradient accumulation: $GRAD_ACCUM" >&2
echo "" >&2

# Training
log_section "🚀 STARTING TRAINING"

deepspeed src/train/train_sft.py \
    --deepspeed scripts/zero2.json \
    --model_id $BASE_MODEL \
    --data_path $DATA_PATH \
    --eval_path $VAL_PATH \
    --image_folder $IMAGE_FOLDER \
    --output_dir $OUTPUT_DIR \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --use_liger True \
    --lora_enable True \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PER_DEVICE_BATCH \
    --per_device_eval_batch_size $PER_DEVICE_BATCH \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate 1e-4 \
    --merger_lr 5e-5 \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((768 * 28 * 28)) \
    --max_seq_length 2048 \
    --bf16 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --logging_steps 5 \
    --eval_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    --report_to tensorboard \
    --logging_dir $OUTPUT_DIR/logs \
    --remove_unused_columns False \
    --lazy_preprocess True

exit_code=$?

echo "" >&2

if [ $exit_code -eq 0 ]; then
    log_section "✅ TRAINING COMPLETE!"
    
    clean_corrupted_checkpoints "$OUTPUT_DIR"
    
    echo "" >&2
    log_success "Training finished successfully"
    echo "   Model saved to: $OUTPUT_DIR" >&2
    echo "" >&2
    
    # Find best checkpoint
    best_checkpoint=$(find $OUTPUT_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V -r | head -1)
    
    if [ -n "$best_checkpoint" ]; then
        echo "📁 Best checkpoint: $(basename $best_checkpoint)" >&2
    fi
    
    echo "" >&2
    log_section "🔧 NEXT STEP: MERGE LORA WEIGHTS"
    echo "" >&2
    echo "Run this command:" >&2
    echo "   bash scripts/merge_lora.sh $OUTPUT_DIR ${OUTPUT_DIR}-merged" >&2
    echo "" >&2
    
    # Offer to merge now
    read -p "Merge LoRA weights now? (y/n): " -n 1 -r >&2
    echo "" >&2
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        merge_lora_weights "$OUTPUT_DIR" "${OUTPUT_DIR}-merged"
        
        echo "" >&2
        log_success "✅ Model ready for testing!"
        echo "   Merged model: ${OUTPUT_DIR}-merged" >&2
        echo "" >&2
        echo "📤 Next: Upload to HuggingFace and deploy to Modal" >&2
        echo "" >&2
    fi
    
else
    log_section "❌ TRAINING FAILED"
    echo "Exit code: $exit_code" >&2
    echo "" >&2
    echo "Check logs at: $OUTPUT_DIR/logs" >&2
    echo "" >&2
    exit $exit_code
fi

log_section "🎉 EXPERIMENT COMPLETE"
echo "" >&2
echo "📊 Review training:" >&2
echo "   tensorboard --logdir=$OUTPUT_DIR/logs" >&2
echo "" >&2
