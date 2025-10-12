#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/stage1_train.sh"
source "$SCRIPT_DIR/lib/stage2_train.sh"

log_section "🔥 DissTrack H100 - Two-Stage Training"

# Stage 1
stage1_output=$(run_stage1)
stage1_output=$(echo "$stage1_output" | tr -d '[:space:]')  # Strip whitespace

# Verify we got a valid output path
if [ -z "$stage1_output" ]; then
    log_error "Stage 1 returned empty path"
    exit 1
fi

# Give filesystem a moment to sync
sleep 1

if [ ! -d "$stage1_output" ]; then
    log_error "Stage 1 path does not exist: '$stage1_output'"
    echo "Checking outputs directory:" >&2
    ls -la outputs/ >&2
    exit 1
fi

log_success "Stage 1 completed: $stage1_output"
echo "" >&2

# Stage 2
run_stage2 "$stage1_output" || {
    log_error "Stage 2 failed"
    exit 1
}

# Optional: Merge final model
log_section "🔧 MERGE FINAL MODEL?"
read -p "Merge now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    final_ckpt=$(find $STAGE2_DIR -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
    [ -z "$final_ckpt" ] && final_ckpt="$STAGE2_DIR"
    
    merge_lora_weights "$final_ckpt" "$FINAL_DIR"
    log_success "Final model: $FINAL_DIR"
fi

log_section "🎉 TRAINING COMPLETE!"

echo "📦 Outputs:"
echo "   Stage 1: $stage1_output"
echo "   Stage 2: $STAGE2_DIR"
[ -d "$FINAL_DIR" ] && echo "   Final:   $FINAL_DIR"
echo ""
echo "🎯 Test your model:"
[ -d "$FINAL_DIR" ] && echo "   python -m src.serve.app --model-path $FINAL_DIR" || echo "   python -m src.serve.app --model-path $STAGE2_DIR"
echo ""
