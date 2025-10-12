#!/bin/bash
# Stage 1: Train Merger + LoRA

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ==========================================
# Helper Functions (Define First)
# ==========================================

merge_stage1_checkpoint() {
    local checkpoint=$1
    local output=$2
    
    log_section "🔧 Merging Stage 1 LoRA Weights"
    
    echo "Stage 2 requires a merged model (LoRA + base weights combined)." >&2
    echo "" >&2
    echo "Source checkpoint: $(basename $checkpoint)" >&2
    echo "Output merged model: $(basename $output)" >&2
    echo "" >&2
    echo "This will take ~2 minutes and use ~15GB disk space..." >&2
    echo "" >&2
    
    if merge_lora_weights "$checkpoint" "$output"; then
        echo "" >&2
        log_success "Merge complete: $output"
        return 0
    else
        echo "" >&2
        log_error "Merge failed!"
        echo "" >&2
        echo "Stage 2 cannot proceed without a merged model." >&2
        echo "Possible causes:" >&2
        echo "  - Insufficient disk space (~15GB needed)" >&2
        echo "  - Corrupted checkpoint" >&2
        echo "  - Missing dependencies" >&2
        echo "" >&2
        return 1
    fi
}

train_stage1() {
    local status=$1
    local resume_from=$2
    
    log_section "🚀 STAGE 1: Training Merger + LoRA"
    
    echo "Configuration:" >&2
    echo "   Vision Tower: FROZEN" >&2
    echo "   LLM: FROZEN + LoRA (rank 128)" >&2
    echo "   Merger: TRAINING" >&2
    echo "   Batch Size: 4" >&2
    echo "   Epochs: 3" >&2
    echo "   Checkpoint Strategy: Keep only latest (saves disk space)" >&2
    echo "" >&2
    
    if [ "$status" = "incomplete" ]; then
        echo "Resuming from: $resume_from" >&2
        local ckpt_num=$(basename $resume_from | sed 's/checkpoint-//')
        local remaining=$((132 - ckpt_num))
        echo "Progress: $(basename $resume_from)" >&2
        echo "Remaining steps: ~$remaining" >&2
        echo "Estimated time: ~$((remaining * 3 / 60)) minutes" >&2
    else
        echo "Starting fresh training" >&2
        echo "Total steps: ~132" >&2
        echo "Estimated time: ~8-10 minutes" >&2
    fi
    
    echo "" >&2
    read -p "Press Enter to continue..." >&2
    echo "" >&2
    
    # Build training command
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
        --save_total_limit 1 \
        --report_to tensorboard \
        --logging_dir $STAGE1_DIR/logs"
    
    # Add resume checkpoint if resuming
    if [ "$status" = "incomplete" ] && [ -n "$resume_from" ]; then
        cmd="$cmd --resume_from_checkpoint $resume_from"
    fi
    
    # Run training
    eval $cmd
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "" >&2
        log_section "❌ STAGE 1 FAILED"
        echo "Exit code: $exit_code" >&2
        echo "" >&2
        echo "Training was interrupted or failed." >&2
        echo "You can resume by running this script again." >&2
        echo "" >&2
        exit $exit_code
    fi
    
    # Training succeeded - clean up and merge
    echo "" >&2
    log_section "✅ STAGE 1 TRAINING COMPLETE!"
    
    clean_corrupted_checkpoints "$STAGE1_DIR"
    
    # Find the best checkpoint
    local final_ckpt=""
    for ckpt in $(find $STAGE1_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V -r); do
        if validate_checkpoint "$ckpt" 2>/dev/null; then
            final_ckpt="$ckpt"
            break
        fi
    done
    
    if [ -z "$final_ckpt" ]; then
        log_error "No valid checkpoint found after training!"
        echo "This should not happen. Check $STAGE1_DIR" >&2
        exit 1
    fi
    
    echo "📁 Final checkpoint: $final_ckpt" >&2
    echo "" >&2
    
    # Merge LoRA weights (REQUIRED for Stage 2)
    local merged="${STAGE1_DIR}-merged"
    
    if [ ! -d "$merged" ]; then
        if ! merge_stage1_checkpoint "$final_ckpt" "$merged"; then
            echo "" >&2
            log_error "Cannot proceed to Stage 2 without merged model"
            exit 1
        fi
    else
        log_info "Merged model already exists: $merged"
    fi
    
    # Mark Stage 1 as complete
    touch "$STAGE1_COMPLETE_MARKER"
    
    echo "" >&2
    clear_gpu_memory
    
    # Return merged model path for Stage 2
    echo "$merged"
    return 0
}

# ==========================================
# Main Function
# ==========================================

run_stage1() {
    log_section "🔍 Checking Stage 1 Status"
    
    local status="not_started"
    local resume_from=""
    
    # Check if already complete
    if [ -f "$STAGE1_COMPLETE_MARKER" ]; then
        status="complete"
        log_success "Stage 1: COMPLETE"
        
        # Always use merged version for Stage 2
        local merged="${STAGE1_DIR}-merged"
        
        if [ -d "$merged" ]; then
            echo "   Using merged model: $(basename $merged)" >&2
            echo "$merged"
            return 0
        else
            log_warning "Stage 1 complete but merged model not found!"
            echo "   Will attempt to merge existing checkpoint..." >&2
            echo "" >&2
            
            # Find the checkpoint to merge
            local checkpoint=""
            if [ -d "$STAGE1_DIR/checkpoint-100" ]; then
                checkpoint="$STAGE1_DIR/checkpoint-100"
            else
                checkpoint=$(find $STAGE1_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V -r | head -1)
            fi
            
            if [ -n "$checkpoint" ] && [ -d "$checkpoint" ]; then
                if merge_stage1_checkpoint "$checkpoint" "$merged"; then
                    echo "$merged"
                    return 0
                else
                    log_error "Cannot create merged model for Stage 2"
                    exit 1
                fi
            else
                log_error "No valid checkpoint found to merge"
                exit 1
            fi
        fi
    fi
    
    # Check if incomplete
    if [ -d "$STAGE1_DIR" ]; then
        status="incomplete"
        log_warning "Stage 1: INCOMPLETE"
        echo "" >&2
        
        clean_corrupted_checkpoints "$STAGE1_DIR"
        
        echo "Checking for valid checkpoints..." >&2
        echo "" >&2
        
        # Find valid checkpoints
        local valid_checkpoints=()
        for ckpt in $(find $STAGE1_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V -r); do
            ckpt_name=$(basename $ckpt)
            if validate_checkpoint "$ckpt" 2>/dev/null; then
                log_success "$ckpt_name: Valid"
                valid_checkpoints+=("$ckpt")
            fi
        done
        
        echo "" >&2
        
        if [ ${#valid_checkpoints[@]} -eq 0 ]; then
            log_error "No valid checkpoints found!"
            log_info "Stage 1 must be restarted from scratch."
            status="not_started"
        else
            resume_from="${valid_checkpoints[0]}"
            local ckpt_num=$(basename $resume_from | sed 's/checkpoint-//')
            
            echo "Found ${#valid_checkpoints[@]} valid checkpoint(s)" >&2
            echo "Latest valid: $(basename $resume_from)" >&2
            echo "Progress: ~$((ckpt_num * 100 / 132))% complete" >&2
            echo "" >&2
            echo "Options:" >&2
            echo "  [1] Resume from $(basename $resume_from) (recommended)" >&2
            echo "  [2] Start Stage 1 from scratch" >&2
            echo "  [3] Cancel" >&2
            echo "" >&2
            read -p "Choose option (1/2/3): " -n 1 -r
            echo "" >&2
            
            case $REPLY in
                1)
                    log_success "Will resume from $(basename $resume_from)"
                    ;;
                2)
                    log_warning "Starting from scratch"
                    read -p "This will delete existing progress. Confirm? (yes/no): " confirm
                    echo "" >&2
                    if [ "$confirm" = "yes" ]; then
                        rm -rf "$STAGE1_DIR"
                        status="not_started"
                        resume_from=""
                    else
                        echo "Cancelled." >&2
                        exit 0
                    fi
                    ;;
                3)
                    echo "Cancelled." >&2
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
    
    echo "" >&2
    
    # Run training if needed
    if [ "$status" != "complete" ]; then
        local merged_path=$(train_stage1 "$status" "$resume_from")
        echo "$merged_path"
        return 0
    fi
}
