#!/bin/bash
set -e

echo "=========================================="
echo "🔥 DissTrack H100 - Resumable Two-Stage Training"
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

# Completion markers
STAGE1_COMPLETE_MARKER="$STAGE1_DIR/.stage1_complete"
STAGE2_COMPLETE_MARKER="$STAGE2_DIR/.stage2_complete"

export PYTHONPATH=src:$PYTHONPATH
unset LD_LIBRARY_PATH

# ==========================================
# Helper: Validate Checkpoint
# ==========================================

validate_checkpoint() {
    local ckpt=$1
    
    # Check required files for resuming training
    if [ ! -f "$ckpt/adapter_model.safetensors" ]; then
        echo "❌ Missing: adapter_model.safetensors"
        return 1
    fi
    
    if [ ! -f "$ckpt/non_lora_state_dict.bin" ]; then
        echo "❌ Missing: non_lora_state_dict.bin"
        return 1
    fi
    
    if [ ! -f "$ckpt/config.json" ]; then
        echo "❌ Missing: config.json (CRITICAL for resuming)"
        return 1
    fi
    
    return 0
}

# ==========================================
# Check Stage 1 Status
# ==========================================

echo "🔍 Checking Stage 1 status..."
echo ""

STAGE1_STATUS="not_started"
RESUME_FROM=""

if [ -f "$STAGE1_COMPLETE_MARKER" ]; then
    # Stage 1 completed successfully
    STAGE1_STATUS="complete"
    echo "✅ Stage 1: COMPLETE"
    
    # Find the final checkpoint
    if [ -d "$STAGE1_DIR/checkpoint-132" ]; then
        RESUME_FROM="$STAGE1_DIR/checkpoint-132"
    elif [ -d "$STAGE1_DIR/checkpoint-100" ]; then
        RESUME_FROM="$STAGE1_DIR/checkpoint-100"
    else
        RESUME_FROM=$(find $STAGE1_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
        [ -z "$RESUME_FROM" ] && RESUME_FROM="$STAGE1_DIR"
    fi
    
    echo "   Using checkpoint: $(basename $RESUME_FROM)"
    
elif [ -d "$STAGE1_DIR" ]; then
    # Stage 1 started but not completed
    STAGE1_STATUS="incomplete"
    
    echo "⚠️  Stage 1: INCOMPLETE"
    echo ""
    echo "Checking for usable checkpoints..."
    echo ""
    
    # Find all checkpoints and validate them
    VALID_CHECKPOINTS=()
    
    for ckpt in $(find $STAGE1_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V -r); do
        ckpt_name=$(basename $ckpt)
        echo "Checking $ckpt_name..."
        
        if validate_checkpoint "$ckpt"; then
            echo "   ✅ Valid and complete"
            VALID_CHECKPOINTS+=("$ckpt")
        else
            echo "   ❌ Corrupted or incomplete"
        fi
        echo ""
    done
    
    if [ ${#VALID_CHECKPOINTS[@]} -eq 0 ]; then
        echo "❌ No valid checkpoints found!"
        echo "   Stage 1 must be restarted from scratch."
        STAGE1_STATUS="not_started"
    else
        # Use the latest valid checkpoint
        RESUME_FROM="${VALID_CHECKPOINTS[0]}"
        CKPT_NUM=$(basename $RESUME_FROM | sed 's/checkpoint-//')
        
        echo "Found ${#VALID_CHECKPOINTS[@]} valid checkpoint(s)"
        echo "Latest valid: $(basename $RESUME_FROM)"
        echo "Progress: ~$((CKPT_NUM * 100 / 132))% complete"
        echo ""
        echo "Options:"
        echo "  [1] Resume from $(basename $RESUME_FROM) (recommended)"
        echo "  [2] Start Stage 1 from scratch"
        echo "  [3] Cancel"
        echo ""
        read -p "Choose option (1/2/3): " -n 1 -r
        echo ""
        
        case $REPLY in
            1)
                echo "✅ Will resume from $(basename $RESUME_FROM)"
                ;;
            2)
                echo "⚠️  Starting from scratch"
                read -p "This will delete existing progress. Confirm? (yes/no): " confirm
                if [ "$confirm" = "yes" ]; then
                    rm -rf "$STAGE1_DIR"
                    STAGE1_STATUS="not_started"
                    RESUME_FROM=""
                else
                    echo "Cancelled."
                    exit 0
                fi
                ;;
            3)
                echo "Cancelled."
                exit 0
                ;;
            *)
                echo "Invalid option."
                exit 1
                ;;
        esac
    fi
else
    echo "📍 Stage 1: NOT STARTED"
fi

echo ""

# ==========================================
# Stage 1: Train Merger + LoRA
# ==========================================

if [ "$STAGE1_STATUS" != "complete" ]; then
    echo "=========================================="
    if [ "$STAGE1_STATUS" = "incomplete" ]; then
        echo "🔄 STAGE 1: RESUMING from $(basename $RESUME_FROM)"
    else
        echo "🚀 STAGE 1: STARTING Training"
    fi
    echo "=========================================="
    echo ""
    echo "Configuration:"
    echo "   Vision Tower: FROZEN"
    echo "   LLM: FROZEN + LoRA (rank 128)"
    echo "   Merger: TRAINING"
    echo "   Batch Size: 4"
    echo "   Epochs: 3"
    echo ""
    
    if [ "$STAGE1_STATUS" = "incomplete" ]; then
        echo "Resuming from: $RESUME_FROM"
        CKPT_NUM=$(basename $RESUME_FROM | sed 's/checkpoint-//')
        REMAINING=$((132 - CKPT_NUM))
        echo "Remaining steps: ~$REMAINING"
        echo ""
    fi
    
    read -p "Press Enter to continue..."
    echo ""
    
    # Build training command
    TRAIN_CMD="deepspeed src/train/train_sft.py \
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

    # Add resume flag if resuming
    if [ "$STAGE1_STATUS" = "incomplete" ] && [ -n "$RESUME_FROM" ]; then
        TRAIN_CMD="$TRAIN_CMD --resume_from_checkpoint $RESUME_FROM"
    fi
    
    # Run training
    eval $TRAIN_CMD
    
    TRAIN_EXIT_CODE=$?
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        # Mark Stage 1 as complete
        touch "$STAGE1_COMPLETE_MARKER"
        echo ""
        echo "=========================================="
        echo "✅ STAGE 1 COMPLETE!"
        echo "=========================================="
        echo ""
        
        # Find final checkpoint - prefer the latest valid one
        FINAL_CKPT=""
        for ckpt in $(find $STAGE1_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V -r); do
            if validate_checkpoint "$ckpt" 2>/dev/null; then
                FINAL_CKPT="$ckpt"
                break
            fi
        done
        
        [ -z "$FINAL_CKPT" ] && FINAL_CKPT="$STAGE1_DIR"
        RESUME_FROM="$FINAL_CKPT"
        
        echo "📁 Final checkpoint: $RESUME_FROM"
        echo ""
    else
        echo ""
        echo "=========================================="
        echo "❌ STAGE 1 FAILED"
        echo "=========================================="
        echo ""
        echo "Exit code: $TRAIN_EXIT_CODE"
        echo ""
        echo "Training was interrupted or failed."
        echo "You can resume by running this script again."
        echo ""
        exit $TRAIN_EXIT_CODE
    fi
    
    # Clean GPU memory
    echo "🧹 Clearing GPU memory..."
    pkill -9 python3 || true
    sleep 5
    echo ""
    
else
    echo "=========================================="
    echo "⏭️  STAGE 1: Already Complete"
    echo "=========================================="
    echo ""
    echo "Using checkpoint: $RESUME_FROM"
    echo ""
    
    # Clean GPU memory
    pkill -9 python3 || true
    sleep 3
fi

# ==========================================
# Verify Stage 1 Checkpoint
# ==========================================

echo "🔍 Verifying Stage 1 checkpoint..."

if [ ! -d "$RESUME_FROM" ]; then
    echo "❌ Error: Checkpoint not found: $RESUME_FROM"
    exit 1
fi

if [ ! -f "$RESUME_FROM/adapter_model.safetensors" ]; then
    echo "❌ Error: Checkpoint is missing adapter weights"
    exit 1
fi

echo "✅ Checkpoint verified"
echo ""

# ==========================================
# Check Stage 2 Status
# ==========================================

echo "🔍 Checking Stage 2 status..."
echo ""

STAGE2_STATUS="not_started"
STAGE2_RESUME_FROM=""

if [ -f "$STAGE2_COMPLETE_MARKER" ]; then
    echo "✅ Stage 2: Already COMPLETE"
    echo ""
    echo "Training is fully complete!"
    echo ""
    read -p "Skip to merge? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
    SKIP_STAGE2=true
    
elif [ -d "$STAGE2_DIR" ]; then
    # Stage 2 started but not completed
    STAGE2_RESUME_FROM=$(find $STAGE2_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
    
    if [ -n "$STAGE2_RESUME_FROM" ] && [ -d "$STAGE2_RESUME_FROM" ]; then
        STAGE2_STATUS="incomplete"
        echo "⚠️  Stage 2: INCOMPLETE"
        echo "   Last checkpoint: $(basename $STAGE2_RESUME_FROM)"
        echo ""
        echo "Options:"
        echo "  [1] Resume Stage 2 (recommended)"
        echo "  [2] Restart Stage 2 from scratch"
        echo "  [3] Cancel"
        echo ""
        read -p "Choose option (1/2/3): " -n 1 -r
        echo ""
        
        case $REPLY in
            1)
                echo "✅ Will resume Stage 2"
                ;;
            2)
                rm -rf "$STAGE2_DIR"
                STAGE2_STATUS="not_started"
                STAGE2_RESUME_FROM=""
                ;;
            3)
                exit 0
                ;;
            *)
                echo "Invalid option."
                exit 1
                ;;
        esac
    else
        STAGE2_STATUS="not_started"
    fi
else
    echo "📍 Stage 2: NOT STARTED"
fi

echo ""

SKIP_STAGE2=${SKIP_STAGE2:-false}

# ==========================================
# Stage 2: Fine-tune Vision Tower
# ==========================================

if [ "$SKIP_STAGE2" = false ]; then
    echo "=========================================="
    if [ "$STAGE2_STATUS" = "incomplete" ]; then
        echo "🔄 STAGE 2: RESUMING"
    else
        echo "🚀 STAGE 2: STARTING"
    fi
    echo "=========================================="
    echo ""
    echo "Configuration:"
    echo "   Starting from: Stage 1 $(basename $RESUME_FROM)"
    echo "   Vision Tower: TRAINING"
    echo "   LLM: FROZEN + LoRA (rank 64)"
    echo "   Merger: FROZEN"
    echo "   Batch Size: 1"
    echo "   Epochs: 2"
    echo ""
    
    # Disk space check
    available_gb=$(df /workspace | tail -1 | awk '{print int($4/1024/1024)}')
    echo "💾 Disk space: ${available_gb}GB available"
    
    if [ $available_gb -lt 20 ]; then
        echo "⚠️  WARNING: Less than 20GB available"
        read -p "Continue? (y/n): " -n 1 -r
        echo ""
        [[ ! $REPLY =~ ^[Yy]$ ]] && exit 0
    fi
    echo ""
    
    read -p "Press Enter to continue..."
    echo ""
    
    # Build Stage 2 command
    TRAIN_CMD="deepspeed src/train/train_sft.py \
        --deepspeed scripts/zero2.json \
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
    
    # Resume or start fresh
    if [ "$STAGE2_STATUS" = "incomplete" ] && [ -n "$STAGE2_RESUME_FROM" ]; then
        TRAIN_CMD="$TRAIN_CMD --model_id $STAGE2_RESUME_FROM --resume_from_checkpoint $STAGE2_RESUME_FROM"
    else
        TRAIN_CMD="$TRAIN_CMD --model_id $RESUME_FROM"
    fi
    
    # Run training
    eval $TRAIN_CMD
    
    TRAIN_EXIT_CODE=$?
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        # Mark Stage 2 as complete
        touch "$STAGE2_COMPLETE_MARKER"
        echo ""
        echo "=========================================="
        echo "✅ STAGE 2 COMPLETE!"
        echo "=========================================="
        echo ""
    else
        echo ""
        echo "=========================================="
        echo "❌ STAGE 2 FAILED"
        echo "=========================================="
        echo ""
        echo "Exit code: $TRAIN_EXIT_CODE"
        echo "You can resume by running this script again."
        echo ""
        exit $TRAIN_EXIT_CODE
    fi
    
    # Clean GPU
    pkill -9 python3 || true
    sleep 3
fi

# ==========================================
# Optional: Merge LoRA Weights
# ==========================================

echo ""
echo "=========================================="
echo "🔧 MERGE LORA WEIGHTS?"
echo "=========================================="
echo ""
read -p "Merge now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    MERGE_SOURCE=$(find $STAGE2_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
    [ -z "$MERGE_SOURCE" ] && MERGE_SOURCE="$STAGE2_DIR"
    
    echo ""
    echo "Merging: $MERGE_SOURCE"
    echo "Output: $FINAL_DIR"
    echo ""
    
    python src/merge_lora_weights.py \
        --model-path "$MERGE_SOURCE" \
        --model-base $BASE_MODEL \
        --save-model-path $FINAL_DIR \
        --safe-serialization
    
    echo ""
    echo "✅ MERGE COMPLETE!"
fi

# ==========================================
# Final Summary
# ==========================================

echo ""
echo "=========================================="
echo "🎉 TRAINING PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "📦 Outputs:"
echo "   Stage 1: $RESUME_FROM"
echo "   Stage 2: $STAGE2_DIR"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Merged:  $FINAL_DIR"
fi
echo ""
echo "💾 Disk usage:"
df -h /workspace | grep workspace
echo ""
echo "🎯 Test your model:"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   python -m src.serve.app --model-path $FINAL_DIR"
else
    echo "   python -m src.serve.app --model-path $STAGE2_DIR"
fi
echo ""
