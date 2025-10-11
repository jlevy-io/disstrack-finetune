#!/bin/bash
# Merge LoRA weights with base model
# Run on RunPod after training completes

set -e

MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH="outputs/qwen2.5-vl-roastme-v4"
OUTPUT_PATH="outputs/qwen2.5-vl-roastme-v4-merged"

echo "=========================================="
echo "🔧 Merging LoRA Weights"
echo "=========================================="
echo ""
echo "Base model: $MODEL_ID"
echo "Adapter: $ADAPTER_PATH"
echo "Output: $OUTPUT_PATH"
echo ""

python src/train/merge_lora.py \
    --model_id $MODEL_ID \
    --adapter_path $ADAPTER_PATH \
    --output_path $OUTPUT_PATH

echo ""
echo "✅ Merge complete!"
echo "📁 Merged model: $OUTPUT_PATH"
echo ""
