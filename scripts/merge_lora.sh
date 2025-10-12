#!/bin/bash
# Merge LoRA weights with base model
# Run on RunPod after training completes

set -e

MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH="${1:-outputs/qwen2.5-vl-roastme-v4}"
OUTPUT_PATH="${2:-${ADAPTER_PATH}-merged}"

echo "=========================================="
echo "🔧 Merging LoRA Weights"
echo "=========================================="
echo ""
echo "Base model: $MODEL_ID"
echo "Adapter: $ADAPTER_PATH"
echo "Output: $OUTPUT_PATH"
echo ""

python src/merge_lora_weights.py \
    --model-path $ADAPTER_PATH \
    --model-base $MODEL_ID \
    --save-model-path $OUTPUT_PATH \
    --safe-serialization

echo ""
echo "✅ Merge complete!"
echo "📁 Merged model: $OUTPUT_PATH"
echo ""
