#!/bin/bash
# scripts/finetune_roastme_v3_stage1.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

log_section "🔥 DissTrack v3 - Stage 1: Style Learning (Text-Only)"

BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH="data/llava_format/stage1_text_only.json"
STAGE1_DIR="outputs/roastme-v3-stage1"

BATCH_PER_DEVICE=16
GRAD_ACCUM=4
GLOBAL_BATCH=$((BATCH_PER_DEVICE * GRAD_ACCUM))

export PYTHONPATH=src:$PYTHONPATH
unset LD_LIBRARY_PATH

log_info "Configuration:"
echo "   Model: $BASE_MODEL" >&2
echo "   Training samples: 20,000 (text-only)" >&2
echo "   Output: $STAGE1_DIR" >&2
echo "   Global batch size: $GLOBAL_BATCH" >&2
echo "   Epochs: 1" >&2
echo "" >&2

log_section "📋 STAGE 1 STRATEGY"
echo "   Vision Tower: FROZEN (not used - text only)" >&2
echo "   LLM: FROZEN + LoRA rank 128" >&2
echo "   Merger: FROZEN (not used - text only)" >&2
echo "   Target: Learn r/RoastMe style and ~67 char length" >&2
echo "" >&2

check_disk_space 30 || {
    log_error "Need at least 30GB free for Stage 1"
    exit 1
}

if [ ! -f "$DATA_PATH" ]; then
    log_error "Stage 1 data not found: $DATA_PATH"
    echo "" >&2
    echo "Run this first:" >&2
    echo "   python tools/prepare_v3_stage1_data.py" >&2
    echo "" >&2
    exit 1
fi

TOTAL_STEPS=$((20000 / GLOBAL_BATCH))
echo "Estimated steps: ~$TOTAL_STEPS" >&2
echo "Estimated time: ~30-45 minutes" >&2
echo "Estimated cost: ~\$1-2 (H100)" >&2
echo "Disk usage: ~20GB (1 checkpoint)" >&2
echo "" >&2

read -p "Press Enter to start Stage 1 training..." >&2
echo "" >&2

log_section "🚀 STARTING STAGE 1"

deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero2.json \
    --model_id $BASE_MODEL \
    --data_path $DATA_PATH \
    --output_dir $STAGE1_DIR \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --lora_enable True \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --lora_namespan_exclude "['visual', 'merger', 'lm_head', 'embed_tokens']" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --fp16 False \
    --tf32 True \
    --gradient_checkpointing True \
    --max_seq_length 512 \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --report_to tensorboard \
    --logging_dir $STAGE1_DIR/logs \
    --lazy_preprocess True \
    --dataloader_num_workers 4

exit_code=$?

echo "" >&2

if [ $exit_code -eq 0 ]; then
    log_section "✅ STAGE 1 COMPLETE!"
    
    clean_corrupted_checkpoints "$STAGE1_DIR"
    
    echo "" >&2
    log_success "Style learning finished"
    echo "   Checkpoint: $STAGE1_DIR" >&2
    echo "" >&2
    
    echo "💾 Current disk usage:" >&2
    df -h /workspace 2>/dev/null | grep -v "Filesystem" || df -h / | grep -v "Filesystem"
    echo "" >&2
    
    log_section "🔧 MERGING LORA WEIGHTS"
    echo "Stage 2 requires merged weights from Stage 1..." >&2
    echo "This will take ~2 minutes and use ~15GB disk space" >&2
    echo "" >&2
    
    read -p "Merge now? (recommended: y/n): " -n 1 -r >&2
    echo "" >&2
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if merge_lora_weights "$STAGE1_DIR" "${STAGE1_DIR}-merged"; then
            echo "" >&2
            log_success "Merge complete: ${STAGE1_DIR}-merged"
            
            log_info "Cleaning up Stage 1 checkpoint to save disk space..."
            rm -rf "$STAGE1_DIR"
            log_success "Freed up ~16GB"
            
            echo "" >&2
            echo "💾 Disk after cleanup:" >&2
            df -h /workspace 2>/dev/null | grep -v "Filesystem" || df -h / | grep -v "Filesystem"
            echo "" >&2
            
            log_success "✅ Stage 1 merged model ready!"
            echo "" >&2
            echo "📋 Next Step:" >&2
            echo "   bash scripts/finetune_roastme_v3_stage2.sh" >&2
            echo "" >&2
        else
            log_error "Merge failed!"
            exit 1
        fi
    else
        echo "" >&2
        log_warning "Skipped merge - you'll need to merge before Stage 2"
        echo "   bash scripts/merge_lora.sh $STAGE1_DIR ${STAGE1_DIR}-merged" >&2
        echo "" >&2
    fi
else
    log_section "❌ STAGE 1 FAILED"
    echo "Exit code: $exit_code" >&2
    echo "" >&2
    echo "Check logs at: $STAGE1_DIR/logs" >&2
    echo "" >&2
    exit $exit_code
fi

log_section "🎯 STAGE 1 COMPLETE - READY FOR STAGE 2"
