#!/bin/bash
# Common functions and configuration

# ==========================================
# Configuration
# ==========================================

export BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
export DATA_PATH="data/llava_format/train.json"
export IMAGE_FOLDER="data/raw/images"
export OUTPUT_BASE="outputs/qwen2.5-vl-roastme-v4"

export STAGE1_DIR="${OUTPUT_BASE}-stage1"
export STAGE2_DIR="${OUTPUT_BASE}-stage2"
export FINAL_DIR="${OUTPUT_BASE}-merged"

export STAGE1_COMPLETE_MARKER="$STAGE1_DIR/.stage1_complete"
export STAGE2_COMPLETE_MARKER="$STAGE2_DIR/.stage2_complete"

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

log_info() {
    echo "ℹ️  $1"
}

log_success() {
    echo "✅ $1"
}

log_warning() {
    echo "⚠️  $1"
}

log_error() {
    echo "❌ $1"
}

validate_checkpoint() {
    local ckpt=$1
    
    [ ! -f "$ckpt/adapter_model.safetensors" ] && return 1
    [ ! -f "$ckpt/non_lora_state_dict.bin" ] && return 1
    [ ! -f "$ckpt/config.json" ] && return 1
    [ ! -f "$ckpt/trainer_state.json" ] && return 1
    
    return 0
}

clean_corrupted_checkpoints() {
    local stage_dir=$1
    local quarantine_dir="$stage_dir/.corrupted_checkpoints"
    
    echo "🔍 Scanning for corrupted checkpoints in $(basename $stage_dir)..."
    
    local found_corrupted=false
    
    for ckpt in $(find $stage_dir -maxdepth 1 -type d -name "checkpoint-*" | sort -V); do
        ckpt_name=$(basename $ckpt)
        
        if ! validate_checkpoint "$ckpt" 2>/dev/null; then
            found_corrupted=true
            log_error "Corrupted: $ckpt_name"
            
            mkdir -p "$quarantine_dir"
            echo "      Moving to quarantine: $quarantine_dir/$ckpt_name"
            mv "$ckpt" "$quarantine_dir/$ckpt_name"
        fi
    done
    
    if [ "$found_corrupted" = false ]; then
        log_success "No corrupted checkpoints found"
    else
        echo ""
        log_info "Corrupted checkpoints moved to: $quarantine_dir"
        log_info "(You can safely delete this folder later)"
    fi
    
    echo ""
}

clear_gpu_memory() {
    echo "🧹 Clearing GPU memory..."
    pkill -9 python3 || true
    sleep 3
}

check_disk_space() {
    local required_gb=$1
    local available_gb=$(df /workspace | tail -1 | awk '{print int($4/1024/1024)}')
    
    echo "💾 Disk space: ${available_gb}GB available"
    
    if [ $available_gb -lt $required_gb ]; then
        log_warning "Less than ${required_gb}GB available"
        return 1
    fi
    
    return 0
}

merge_lora_weights() {
    local checkpoint=$1
    local output_path=$2
    
    log_section "🔧 Merging LoRA Weights"
    
    echo "Source: $checkpoint"
    echo "Output: $output_path"
    echo ""
    
    python src/merge_lora_weights.py \
        --model-path "$checkpoint" \
        --model-base $BASE_MODEL \
        --save-model-path "$output_path" \
        --safe-serialization
    
    return $?
}
