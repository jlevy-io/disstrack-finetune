#!/bin/bash
# Stage 2: Fine-tune Vision Tower

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

train_stage2() {
    local stage1_model=$1  # Keep parameter for compatibility
    local status=$2
    local resume_from=$3
    
    log_section "🚀 STAGE 2: Training Vision Tower"
    
    echo "Configuration:" >&2
    echo "   Starting from: Base model (vision tower was frozen in Stage 1)" >&2
    echo "   Vision Tower: TRAINING" >&2
    echo "   LLM: FROZEN + LoRA (rank 64)" >&2
    echo "   Merger: FROZEN" >&2
    echo "   Batch Size: 1" >&2
    echo "   Epochs: 2" >&2
    echo "   Checkpoint Strategy: Keep only latest (saves disk space)" >&2
    echo "" >&2
    
    check_disk_space 20 || {
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo "" >&2
        [[ ! $REPLY =~ ^[Yy]$ ]] && exit 0
    }
    
    echo "" >&2
    read -p "Press Enter to continue..." >&2
    echo "" >&2
    
    # Build command - USE BASE MODEL, not Stage 1 merged
    local cmd="deepspeed src/train/train_sft.py"
    cmd="$cmd --deepspeed scripts/zero2.json"
    cmd="$cmd --model_id $BASE_MODEL"
    cmd="$cmd --data_path $DATA_PATH"
    cmd="$cmd --image_folder $IMAGE_FOLDER"
    cmd="$cmd --output_dir $STAGE2_DIR"
    cmd="$cmd --num_train_epochs 2"
    cmd="$cmd --per_device_train_batch_size 1"
    cmd="$cmd --gradient_accumulation_steps 16"
    cmd="$cmd --freeze_vision_tower False"
    cmd="$cmd --freeze_llm True"
    cmd="$cmd --freeze_merger True"
    cmd="$cmd --learning_rate 5e-5"
    cmd="$cmd --vision_lr 2e-5"
    cmd="$cmd --image_min_pixels $((128 * 28 * 28))"
    cmd="$cmd --image_max_pixels $((512 * 28 * 28))"
    cmd="$cmd --bf16 True"
    cmd="$cmd --use_liger True"
    cmd="$cmd --lora_enable True"
    cmd="$cmd --lora_rank 64"
    cmd="$cmd --lora_alpha 128"
    cmd="$cmd --lora_dropout 0.05"
    cmd="$cmd --gradient_checkpointing True"
    cmd="$cmd --max_seq_length 2048"
    cmd="$cmd --dataloader_num_workers 4"
    cmd="$cmd --logging_steps 5"
    cmd="$cmd --save_strategy steps"
    cmd="$cmd --save_steps 50"
    cmd="$cmd --save_total_limit 1"
    cmd="$cmd --report_to tensorboard"
    cmd="$cmd --logging_dir $STAGE2_DIR/logs"
    
    # Add resume if needed
    if [ "$status" = "incomplete" ] && [ -n "$resume_from" ]; then
        cmd="$cmd --resume_from_checkpoint \"$resume_from\""
    fi
    
    # Run training
    eval $cmd
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        clean_corrupted_checkpoints "$STAGE2_DIR"
        touch "$STAGE2_COMPLETE_MARKER"
        
        # Clean up Stage 1 files to save disk space
        log_section "🧹 CLEANING UP STAGE 1 FILES"
        echo "Stage 2 complete! Cleaning up Stage 1 to save disk space..." >&2
        echo "" >&2
        
        local space_freed=0
        
        # Delete Stage 1 checkpoints
        if [ -d "$STAGE1_DIR" ]; then
            local stage1_size=$(du -sm "$STAGE1_DIR" 2>/dev/null | cut -f1)
            echo "   Deleting Stage 1 checkpoints (~${stage1_size}MB)..." >&2
            rm -rf "$STAGE1_DIR"
            space_freed=$((space_freed + stage1_size))
        fi
        
        # Delete Stage 1 merged model (optional - uncomment if you need more space)
        # WARNING: You won't be able to resume Stage 2 from scratch without this!
        # if [ -d "${STAGE1_DIR}-merged" ]; then
        #     local merged_size=$(du -sm "${STAGE1_DIR}-merged" 2>/dev/null | cut -f1)
        #     echo "   Deleting Stage 1 merged model (~${merged_size}MB)..." >&2
        #     rm -rf "${STAGE1_DIR}-merged"
        #     space_freed=$((space_freed + merged_size))
        # fi
        
        if [ $space_freed -gt 0 ]; then
            echo "" >&2
            log_success "Freed up ~${space_freed}MB of disk space"
        fi
        
        echo "" >&2
        echo "💾 Current disk usage:" >&2
        df -h /workspace 2>/dev/null | grep -v "Filesystem" || df -h / | grep -v "Filesystem"
        echo "" >&2
        
        log_section "✅ STAGE 2 COMPLETE!"
        clear_gpu_memory
        return 0
    else
        log_section "❌ STAGE 2 FAILED"
        echo "Exit code: $exit_code" >&2
        exit $exit_code
    fi
}

run_stage2() {
    local stage1_model=$1
    
    log_section "🔍 Checking Stage 2 Status"
    
    local status="not_started"
    local resume_from=""
    
    # Check if complete
    if [ -f "$STAGE2_COMPLETE_MARKER" ]; then
        log_success "Stage 2: Already COMPLETE"
        echo "" >&2
        return 0
    fi
    
    # Check if incomplete
    if [ -d "$STAGE2_DIR" ]; then
        clean_corrupted_checkpoints "$STAGE2_DIR"
        
        resume_from=$(find $STAGE2_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
        
        if [ -n "$resume_from" ] && validate_checkpoint "$resume_from" 2>/dev/null; then
            status="incomplete"
            log_warning "Stage 2: INCOMPLETE"
            echo "   Last checkpoint: $(basename $resume_from)" >&2
            echo "" >&2
            echo "Options:" >&2
            echo "  [1] Resume Stage 2" >&2
            echo "  [2] Restart Stage 2 from scratch" >&2
            echo "  [3] Cancel" >&2
            echo "" >&2
            read -p "Choose option (1/2/3): " -n 1 -r
            echo "" >&2
            
            case $REPLY in
                1) ;;
                2)
                    rm -rf "$STAGE2_DIR"
                    status="not_started"
                    ;;
                3) exit 0 ;;
                *) exit 1 ;;
            esac
        fi
    else
        log_info "Stage 2: NOT STARTED"
    fi
    
    echo "" >&2
    
    train_stage2 "$stage1_model" "$status" "$resume_from"
}
