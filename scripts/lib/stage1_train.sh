#!/bin/bash
# Stage 1: Train Merger + LoRA

source "$(dirname "$0")/common.sh"

run_stage1() {
    log_section "🔍 Checking Stage 1 Status"
    
    local status="not_started"
    local resume_from=""
    
    # Check if already complete
    if [ -f "$STAGE1_COMPLETE_MARKER" ]; then
        status="complete"
        log_success "Stage 1: COMPLETE"
        
        # Find checkpoint
        if [ -d "$STAGE1_DIR/checkpoint-100" ]; then
            resume_from="$STAGE1_DIR/checkpoint-100"
        else
            resume_from=$(find $STAGE1_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
            [ -z "$resume_from" ] && resume_from="$STAGE1_DIR"
        fi
        
        echo "   Using checkpoint: $(basename $resume_from)"
        
        # Check if merged version exists
        local merged="${STAGE1_DIR}-merged"
        if [ -d "$merged" ]; then
            echo "$merged"
            return 0
        else
            echo "$resume_from"
            return 0
        fi
    fi
    
    # Check if incomplete
    if [ -d "$STAGE1_DIR" ]; then
        status="incomplete"
        log_warning "Stage 1: INCOMPLETE"
        echo ""
        
        clean_corrupted_checkpoints "$STAGE1_DIR"
        
        # Find valid checkpoints
        local valid_checkpoints=()
        for ckpt in $(find $STAGE1_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V -r); do
            if validate_checkpoint "$ckpt" 2>/dev/null; then
                log_success "$(basename $ckpt): Valid"
                valid_checkpoints+=("$ckpt")
            fi
        done
        
        echo ""
        
        if [ ${#valid_checkpoints[@]} -eq 0 ]; then
            log_error "No valid checkpoints found!"
            status="not_started"
        else
            resume_from="${valid_checkpoints[0]}"
            local ckpt_num=$(basename $resume_from | sed 's/checkpoint-//')
            
            echo "Found ${#valid_checkpoints[@]} valid checkpoint(s)"
            echo "Latest valid: $(basename $resume_from)"
            echo "Progress: ~$((ckpt_num * 100 / 132))% complete"
            echo ""
            echo "Options:"
            echo "  [1] Resume from $(basename $resume_from) (recommended)"
            echo "  [2] Start Stage 1 from scratch"
            echo "  [3] Cancel"
            echo ""
            read -p "Choose option (1/2/3): " -n 1 -r
            echo ""
            
            case $REPLY in
                1)
                    log_success "Will resume from $(basename $resume_from)"
                    ;;
                2)
                    log_warning "Starting from scratch"
                    rm -rf "$STAGE1_DIR"
                    status="not_started"
                    resume_from=""
                    ;;
                3)
                    exit 0
                    ;;
                *)
                    log_error "Invalid option"
                    exit 1
                    ;;
            esac
        fi
    else
        log_info "Stage 1: NOT STARTED"
    fi
    
    echo ""
    
    # Run training if needed
    if [ "$status" != "complete" ]; then
        train_stage1 "$status" "$resume_from"
    fi
    
    # Return the path to use for Stage 2
    if [ -d "${STAGE1_DIR}-merged" ]; then
        echo "${STAGE1_DIR}-merged"
    else
        local final=$(find $STAGE1_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
        [ -z "$final" ] && final="$STAGE1_DIR"
        echo "$final"
    fi
}

train_stage1() {
    local status=$1
    local resume_from=$2
    
    log_section "🚀 STAGE 1: Training Merger + LoRA"
    
    if [ "$status" = "incomplete" ]; then
        echo "Resuming from: $resume_from"
        local ckpt_num=$(basename $resume_from | sed 's/checkpoint-//')
        echo "Remaining steps: ~$((132 - ckpt_num))"
    fi
    
    echo ""
    echo "Configuration:"
    echo "   Vision Tower: FROZEN"
    echo "   LLM: FROZEN + LoRA (rank 128)"
    echo "   Merger: TRAINING"
    echo "   Batch Size: 4"
    echo "   Epochs: 3"
    echo ""
    
    read -p "Press Enter to continue..."
    echo ""
    
    # Build command
    local cmd="deepspeed src/train/train_sft.py \
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
        --save_strategy steps \
        --save_steps 50 \
        --save_total_limit 2 \
        --report_to tensorboard \
        --logging_dir $STAGE1_DIR/logs"
    
    # Add resume if needed
    if [ "$status" = "incomplete" ] && [ -n "$resume_from" ]; then
        cmd="$cmd --resume_from_checkpoint $resume_from"
    fi
    
    # Run training
    eval $cmd
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        clean_corrupted_checkpoints "$STAGE1_DIR"
        
        # Find final checkpoint
        local final_ckpt=""
        for ckpt in $(find $STAGE1_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V -r); do
            if validate_checkpoint "$ckpt" 2>/dev/null; then
                final_ckpt="$ckpt"
                break
            fi
        done
        [ -z "$final_ckpt" ] && final_ckpt="$STAGE1_DIR"
        
        log_section "✅ STAGE 1 COMPLETE!"
        echo "📁 Stage 1 checkpoint: $final_ckpt"
        echo ""
        
        # Merge for Stage 2
        local merged="${STAGE1_DIR}-merged"
        if [ ! -d "$merged" ]; then
            echo "🔧 Merging Stage 1 for Stage 2..."
            echo ""
            
            if merge_lora_weights "$final_ckpt" "$merged"; then
                log_success "Stage 1 merged: $merged"
            else
                log_warning "Merge failed, will use checkpoint directly"
            fi
        fi
        
        touch "$STAGE1_COMPLETE_MARKER"
        clear_gpu_memory
        
        return 0
    else
        log_section "❌ STAGE 1 FAILED"
        echo "Exit code: $exit_code"
        exit $exit_code
    fi
}
