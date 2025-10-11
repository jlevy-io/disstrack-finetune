#!/bin/bash
set -e

echo "=========================================="
echo "🔧 Merging LoRA Weights (H100 Training)"
echo "=========================================="
echo ""

# Configuration
MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH="outputs/qwen2.5-vl-roastme-v4-h100-lora128"
OUTPUT_PATH="outputs/qwen2.5-vl-roastme-v4-h100-lora128-merged"

export PYTHONPATH=src:$PYTHONPATH

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
echo "=========================================="
echo "✅ Merge Complete!"
echo "=========================================="
echo ""
echo "📁 Merged model: $OUTPUT_PATH"
echo ""
echo "🎯 Next steps:"
echo "   1. Test: python src/serve/app.py --model-path $OUTPUT_PATH"
echo "   2. Deploy to Modal/RunPod"
echo ""
