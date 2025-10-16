#!/bin/bash
# scripts/finetune_roastme_v3_stage2.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

log_section "🔥 DissTrack v3 - Stage 2: Visual Fine-Tuning"

STAGE1_MERGED="outputs/roastme-v3-stage1-merged"
DATA_PATH="data/llava_format/train.json"
VAL_PATH="data/llava_format/val.json"
IMAGE_FOLDER="data/raw/images"
STAGE2_DIR="outputs/roastme-v3-stage2"

BATCH_PER_DEVICE=4
GRAD_ACCUM=4
GLOBAL_BATCH=$((BATCH_PER_DEVICE * GRAD_ACCUM))

export PYTHONPATH=src:$PYTHONPATH
unset LD_LIBRARY_PATH

if [ ! -d "$STAGE1_MERGED" ]; then
    log_error "Stage 1 merged model not found: $STAGE1_MERGED"
    echo "" >&2
    echo "Complete Stage 1 first:" >&2
    echo "   bash scripts/finetune_roastme_v3_stage1.sh" >&2
    echo "" >&2
    exit 1
fi

log_info "Configuration:"
echo "   Base: $(basename $STAGE1_MERGED) (from Stage 1)" >&2
echo "   Training samples: 693" >&2
echo "   Validation samples: 78" >&2
echo "   Output: $STAGE2_DIR" >&2
echo "   Global batch size: $GLOBAL_BATCH" >&2
echo "   Epochs: 2" >&2
echo "" >&2

log_section "📋 STAGE 2 STRATEGY"
echo "   Vision Tower: TRAINING (learn visual features)" >&2
echo "   LLM: FROZEN + small LoRA (preserve Stage 1 style)" >&2
echo "   Merger: TRAINING" >&2
echo "   Learning Rate: LOWER than Stage 1 (prevent forgetting)" >&2
echo "   Early Stopping: YES (prevent overfitting on 693 samples)" >&2
echo "" >&2

check_disk_space 50 || {
    log_error "Need at least 50GB free for Stage 2"
    echo "Current usage:" >&2
    df -h /workspace 2>/dev/null | grep -v "Filesystem" || df -h / | grep -v "Filesystem"
    echo "" >&2
    exit 1
}

TOTAL_STEPS=$(((693 / GLOBAL_BATCH) * 2))
echo "Estimated steps: ~$TOTAL_STEPS" >&2
echo "Estimated time: ~30-45 minutes" >&2
echo "Estimated cost: ~\$1-2 (H100)" >&2
echo "Peak disk usage: ~47GB" >&2
echo "" >&2

read -p "Press Enter to start Stage 2 training..." >&2
echo "" >&2

log_section "🚀 STARTING STAGE 2"

deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero2.json \
    --model_id $STAGE1_MERGED \
    --data_path $DATA_PATH \
    --eval_path $VAL_PATH \
    --image_folder $IMAGE_FOLDER \
    --output_dir $STAGE2_DIR \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm True \
    --freeze_merger False \
    --lora_enable True \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --lora_namespan_exclude "['visual', 'merger', 'lm_head', 'embed_tokens']" \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --per_device_eval_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate 5e-5 \
    --vision_lr 2e-5 \
    --merger_lr 5e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1024 * 28 * 28)) \
    --max_seq_length 2048 \
    --bf16 True \
    --fp16 False \
    --tf32 True \
    --gradient_checkpointing True \
    --logging_steps 5 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    --report_to tensorboard \
    --logging_dir $STAGE2_DIR/logs \
    --lazy_preprocess True \
    --dataloader_num_workers 4

exit_code=$?

echo "" >&2

if [ $exit_code -eq 0 ]; then
    log_section "✅ STAGE 2 COMPLETE!"
    
    clean_corrupted_checkpoints "$STAGE2_DIR"
    
    echo "" >&2
    log_success "Visual fine-tuning finished"
    echo "   Model saved to: $STAGE2_DIR" >&2
    echo "" >&2
    
    best_checkpoint=$(find $STAGE2_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V -r | head -1)
    
    if [ -n "$best_checkpoint" ]; then
        echo "📁 Best checkpoint: $(basename $best_checkpoint)" >&2
    fi
    
    echo "" >&2
    log_section "🔧 MERGE FINAL MODEL"
    echo "" >&2
    read -p "Merge LoRA weights now? (y/n): " -n 1 -r >&2
    echo "" >&2
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        final_ckpt=${best_checkpoint:-$STAGE2_DIR}
        
        if merge_lora_weights "$final_ckpt" "${STAGE2_DIR}-merged"; then
            echo "" >&2
            log_success "✅ Final v3 model ready!"
            echo "   Location: ${STAGE2_DIR}-merged" >&2
            echo "" >&2
            
            log_section "🧹 CLEANUP TO SAVE DISK SPACE"
            echo "" >&2
            
            read -p "Delete Stage 1 merged model? (~15GB) (y/n): " -n 1 -r >&2
            echo "" >&2
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                if [ -d "$STAGE1_MERGED" ]; then
                    log_info "Removing $STAGE1_MERGED..."
                    rm -rf "$STAGE1_MERGED"
                    log_success "Freed ~15GB"
                fi
            fi
            
            echo "" >&2
            read -p "Delete Stage 2 checkpoints? (~32GB) (y/n): " -n 1 -r >&2
            echo "" >&2
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                log_info "Removing Stage 2 checkpoints..."
                rm -rf "$STAGE2_DIR"
                log_success "Freed ~32GB"
            fi
            
            echo "" >&2
            echo "💾 Final disk usage:" >&2
            df -h /workspace 2>/dev/null | grep -v "Filesystem" || df -h / | grep -v "Filesystem"
            echo "" >&2
            
            log_section "🎉 V3 MODEL READY!"
            echo "" >&2
            echo "📁 Final model: ${STAGE2_DIR}-merged" >&2
            echo "" >&2
            echo "📤 Next steps:" >&2
            echo "   1. Upload to HuggingFace:" >&2
            echo "      huggingface-cli upload ${STAGE2_DIR}-merged jasonlevy/roastme-model-v3" >&2
            echo "" >&2
            echo "   2. Deploy to Modal:" >&2
            echo "      Update deployment/modal_inference.py MODEL_ID to v3" >&2
            echo "      modal deploy deployment/modal_inference.py" >&2
            echo "" >&2
            echo "   3. Evaluate:" >&2
            echo "      python tools/collect_model_results.py --model-name v3" >&2
            echo "" >&2
        else
            log_error "Merge failed!"
            exit 1
        fi
    fi
else
    log_section "❌ STAGE 2 FAILED"
    echo "Exit code: $exit_code" >&2
    echo "" >&2
    echo "Check logs at: $STAGE2_DIR/logs" >&2
    echo "" >&2
    exit $exit_code
fi
