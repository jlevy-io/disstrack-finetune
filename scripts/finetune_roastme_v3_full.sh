#!/bin/bash
# scripts/finetune_roastme_v3_full.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

log_section "🔥 DissTrack v3 - Full Two-Stage Pipeline"

echo "This pipeline will:" >&2
echo "   1. Prepare 20k text samples from HF dataset" >&2
echo "   2. Stage 1: Style learning (text-only, 1 epoch)" >&2
echo "   3. Merge Stage 1 LoRA weights" >&2
echo "   4. Stage 2: Visual fine-tuning (693 image pairs, 2 epochs)" >&2
echo "   5. Merge final v3 model" >&2
echo "   6. Cleanup intermediate checkpoints" >&2
echo "" >&2

log_section "📊 ESTIMATES"
echo "   Total time: ~1.5-2 hours" >&2
echo "   Total cost: ~\$3-5 (H100)" >&2
echo "   Peak disk: ~50GB" >&2
echo "   Final disk: ~15GB (after cleanup)" >&2
echo "" >&2

log_section "💾 DISK SPACE CHECK"

available=$(df /workspace 2>/dev/null | tail -1 | awk '{print int($4/1024/1024)}' || df / | tail -1 | awk '{print int($4/1024/1024)}')
echo "Available: ${available}GB" >&2
echo "Required: 50GB minimum" >&2
echo "" >&2

if [ $available -lt 50 ]; then
    log_error "Insufficient disk space!"
    echo "Need: 50GB" >&2
    echo "Have: ${available}GB" >&2
    echo "Short: $((50 - available))GB" >&2
    echo "" >&2
    exit 1
fi

log_success "✅ Disk space OK (${available}GB available)"
echo "" >&2

log_section "📋 DATA PREPARATION"

if [ ! -f "data/llava_format/stage1_text_only.json" ]; then
    log_info "Preparing Stage 1 data (20k samples)..."
    python tools/prepare_v3_stage1_data.py || {
        log_error "Data preparation failed"
        exit 1
    }
else
    log_success "Stage 1 data already prepared"
fi

if [ ! -f "data/llava_format/train.json" ]; then
    log_error "Stage 2 data not found: data/llava_format/train.json"
    echo "Run: python tools/clean_and_convert.py" >&2
    exit 1
fi

echo "" >&2
log_section "🚦 READY TO START"

echo "Review configuration:" >&2
echo "   Stage 1: 20,000 text samples" >&2
echo "   Stage 2: 693 image+roast pairs" >&2
echo "   Validation: 78 samples" >&2
echo "" >&2
echo "Training will start immediately after confirmation." >&2
echo "The process is automated but you can monitor:" >&2
echo "   - Terminal output for progress" >&2
echo "   - TensorBoard for metrics" >&2
echo "" >&2

read -p "Start full v3 training? (yes/no): " confirm >&2
echo "" >&2

if [ "$confirm" != "yes" ]; then
    log_warning "Training cancelled"
    exit 0
fi

START_TIME=$(date +%s)

log_section "PHASE 1: STAGE 1 - STYLE LEARNING"
bash scripts/finetune_roastme_v3_stage1.sh || {
    log_error "Stage 1 failed!"
    exit 1
}

if [ ! -d "outputs/roastme-v3-stage1-merged" ]; then
    log_error "Stage 1 merge incomplete!"
    exit 1
fi

clear_gpu_memory

log_section "PHASE 2: STAGE 2 - VISUAL FINE-TUNING"
bash scripts/finetune_roastme_v3_stage2.sh || {
    log_error "Stage 2 failed!"
    exit 1
}

if [ ! -d "outputs/roastme-v3-stage2-merged" ]; then
    log_error "Stage 2 merge incomplete!"
    exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

log_section "🎉 V3 TRAINING COMPLETE!"

echo "" >&2
log_success "Total time: ${HOURS}h ${MINUTES}m"
echo "" >&2
echo "✅ v3 model ready: outputs/roastme-v3-stage2-merged" >&2
echo "" >&2

echo "💾 Final disk usage:" >&2
df -h /workspace 2>/dev/null | grep -v "Filesystem" || df -h / | grep -v "Filesystem"
echo "" >&2

log_section "📤 NEXT STEPS"
echo "" >&2
echo "1. Upload to HuggingFace:" >&2
echo "   huggingface-cli upload outputs/roastme-v3-stage2-merged jasonlevy/roastme-model-v3" >&2
echo "" >&2
echo "2. Deploy to Modal:" >&2
echo "   # Edit deployment/modal_inference.py" >&2
echo "   MODEL_ID = \"jasonlevy/roastme-model-v3\"" >&2
echo "   modal deploy deployment/modal_inference.py" >&2
echo "" >&2
echo "3. Evaluate v3:" >&2
echo "   python tools/collect_model_results.py --model-name v3 --num-images 30 --num-candidates 5" >&2
echo "" >&2
echo "4. Compare all versions:" >&2
echo "   python tools/compare_results.py \\" >&2
echo "     --v1-results evaluation_results/v1_results_*.json \\" >&2
echo "     --v2-results evaluation_results/v2_results_*.json \\" >&2
echo "     --v3-results evaluation_results/v3_results_*.json" >&2
echo "" >&2

log_section "🎯 TRAINING PIPELINE COMPLETE!"
