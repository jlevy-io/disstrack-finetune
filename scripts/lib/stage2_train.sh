#!/bin/bash
# Stage 2: Fine-tune Vision Tower

source "$(dirname "$0")/common.sh"

run_stage2() {
    local stage1_model=$1
    
    log_section "🔍 Checking Stage 2 Status"
    
    local status="not_started"
    local resume_from=""
    
    # Check if complete
    if [ -f "$STAGE2_COMPLETE_MARKER" ]; then
        log_success "Stage 2: Already COMPLETE"
        echo ""
        return 0
    fi
    
    # Check if incomplete
    if [ -d "$STAGE2_DIR" ]; then
        clean_corrupted_checkpoints "$STAGE2_DIR"
        
        resume_from=$(find $STAGE2_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
        
        if [ -n "$resume_from" ] && validate_checkpoint "$resume_from" 2>/dev/null; then
            status="incomplete"
            log_warning "Stage 2: INCOMPLETE"
            echo "   Last checkpoint: $(basename $resume_from)"
            echo ""
            echo "Options:"
            echo "  [1] Resume Stage 2"
            echo "  [2] Restart Stage 2 from scratch"
            echo "  [3] Cancel"
            echo ""
            read -p "Choose option (1/2/3): " -n 1 -r
            echo ""
            
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
    
    echo ""
    
    train_stage2 "$stage1_model" "$status" "$resume_from"
}

train_stage2() {
    local stage1_model=$1
    local status=$2
    local resume_from=$3
    
    log_section "🚀 STAGE 2: Training Vision Tower"
    
    echo "Configuration:"
    echo "   Starting from: $(basename $stage1_model)"
    echo "   Vision Tower: TRAINING"
    echo "   LLM: FROZEN + LoRA (rank 64)"
    echo "   Merger: FROZEN"
    echo "   Batch Size: 1"
    echo "   Epochs: 2"
    echo ""
    
    check_disk_space 20 || {
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo ""
        [[ ! $REPLY =~ ^[Yy]$ ]] && exit 0
    }
    
    echo ""
    read -p "Press Enter to continue..."
    echo ""
    
    # Build command
    local cmd="deepspeed src/train/train_sft.py \
        --deepspeed scripts/zero2.json \
        --model_id $stage1_model \
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
        --save_strategy steps \
        --save_steps 50 \
        --save_total_limit 2 \
        --report_to tensorboard \
        --logging_dir $STAGE2_DIR/logs"
    
    # Add resume if needed
    if [ "$status" = "incomplete" ] && [ -n "$resume_from" ]; then
        cmd="$cmd --resume_from_checkpoint $resume_from"
    fi
    
    # Run training
    eval $cmd
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        clean_corrupted_checkpoints "$STAGE2_DIR"
        touch "$STAGE2_COMPLETE_MARKER"
        
        log_section "✅ STAGE 2 COMPLETE!"
        clear_gpu_memory
        return 0
    else
        log_section "❌ STAGE 2 FAILED"
        echo "Exit code: $exit_code"
        exit $exit_code
    fi
}
