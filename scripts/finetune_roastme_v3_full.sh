#!/bin/bash
# scripts/finetune_roastme_v3_full.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

log_section "🔥 DissTrack v3 - Full Two-Stage Pipeline (H200 Optimized)"

echo "This pipeline will:" >&2
echo "   1. Prepare 20k text samples from HF dataset (25-100 chars)" >&2
echo "   2. Stage 1: Style learning (text-only, 1 epoch)" >&2
echo "   3. Merge Stage 1 LoRA weights + cleanup" >&2
echo "   4. Stage 2: Visual fine-tuning (693 image pairs, 2 epochs)" >&2
echo "   5. Merge final v3 model + cleanup" >&2
echo "" >&2

log_section "📊 H200 ESTIMATES"
echo "   GPU: H200 SXM (\$3.05/hr)" >&2
echo "   Stage 1: ~20 min, ~\$1.00" >&2
echo "   Stage 2: ~25 min, ~\$1.25" >&2
echo "   Total time: ~45-60 min" >&2
echo "   Total cost: ~\$2.25-3.00" >&2
echo "" >&2
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

log_section "📋 DATA VERIFICATION"

if [ ! -f "data/huggingface/hf_visual_filtered.json" ]; then
    log_error "HF filtered dataset not found!"
    echo "" >&2
    echo "Run this first:" >&2
    echo "   python tools/filter_hf_with_visual_criteria.py" >&2
    echo "" >&2
    exit 1
fi

if [ ! -f "data/llava_format/stage1_text_only.json" ]; then
    log_info "Preparing Stage 1 data (20k samples, 25-100 chars)..."
    python tools/prepare_v3_stage1_data.py || {
        log_error "Data preparation failed"
        exit 1
    }
else
    log_success "Stage 1 data already prepared"
    
    sample_count=$(jq '. | length' data/llava_format/stage1_text_only.json)
    echo "   Samples: $sample_count" >&2
fi

if [ ! -f "data/llava_format/train.json" ]; then
    log_error "Stage 2 training data not found: data/llava_format/train.json"
    echo "" >&2
    echo "Run: python tools/clean_and_convert.py" >&2
    echo "" >&2
    exit 1
else
    log_success "Stage 2 data verified"
    
    train_count=$(jq '. | length' data/llava_format/train.json)
    val_count=$(jq '. | length' data/llava_format/val.json)
    echo "   Train: $train_count samples" >&2
    echo "   Val: $val_count samples" >&2
fi

echo "" >&2
log_section "🚦 READY TO START"

echo "Final configuration:" >&2
echo "   Stage 1: 20,000 text samples (avg 59 chars)" >&2
echo "   Stage 2: 693 image+roast pairs" >&2
echo "   Validation: 78 samples" >&2
echo "   Hardware: H200 SXM (141GB VRAM)" >&2
echo "   Strategy: Text style learning → Visual grounding" >&2
echo "" >&2
echo "The pipeline is fully automated with cleanup prompts." >&2
echo "You can monitor progress in the terminal output." >&2
echo "" >&2

read -p "Start full v3 training? (yes/no): " confirm >&2
echo "" >&2

if [ "$confirm" != "yes" ]; then
    log_warning "Training cancelled"
    exit 0
fi

START_TIME=$(date +%s)

log_section "🏁 STARTING V3 TRAINING PIPELINE"
echo "" >&2

bash scripts/finetune_roastme_v3_stage1.sh || {
    log_error "Stage 1 failed!"
    exit 1
}

if [ ! -d "outputs/roastme-v3-stage1-merged" ]; then
    log_error "Stage 1 merge incomplete!"
    echo "Cannot proceed to Stage 2 without merged Stage 1 model" >&2
    exit 1
fi

echo "" >&2
log_info "Clearing GPU memory before Stage 2..."
clear_gpu_memory

echo "" >&2
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
COST=$(echo "scale=2; $DURATION / 3600 * 3.05" | bc)

log_section "🎉 V3 TRAINING PIPELINE COMPLETE!"

echo "" >&2
log_success "Total training time: ${HOURS}h ${MINUTES}m"
log_success "Estimated cost: ~\$${COST}"
echo "" >&2

echo "✅ v3 model location: outputs/roastme-v3-stage2-merged" >&2
echo "" >&2

echo "💾 Final disk usage:" >&2
df -h /workspace 2>/dev/null | grep -v "Filesystem" || df -h / | grep -v "Filesystem"
echo "" >&2

log_section "📤 DEPLOYMENT CHECKLIST"
echo "" >&2
echo "□ Upload to HuggingFace:" >&2
echo "  huggingface-cli upload outputs/roastme-v3-stage2-merged jasonlevy/roastme-model-v3" >&2
echo "" >&2
echo "□ Update Modal deployment:" >&2
echo "  1. Edit deployment/modal_inference.py:" >&2
echo "     MODEL_ID = \"jasonlevy/roastme-model-v3\"" >&2
echo "  2. Deploy:" >&2
echo "     modal deploy deployment/modal_inference.py" >&2
echo "" >&2
echo "□ Collect evaluation data:" >&2
echo "  python tools/collect_model_results.py --model-name v3 --num-images 30 --num-candidates 5" >&2
echo "" >&2
echo "□ Compare all versions:" >&2
echo "  python tools/compare_results.py \\" >&2
echo "    --v1-results evaluation_results/v1_results_*.json \\" >&2
echo "    --v2-results evaluation_results/v2_results_*.json \\" >&2
echo "    --v3-results evaluation_results/v3_results_*.json" >&2
echo "" >&2

log_section "🎯 ALL DONE - HAPPY ROASTING! 🔥"
