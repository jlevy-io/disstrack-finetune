#!/bin/bash
set -e

echo "=========================================="
echo "🔥 DissTrack H100 SXM Training (Fixed)"
echo "=========================================="
echo ""

MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH="data/llava_format/train.json"
IMAGE_FOLDER="data/raw/images"
OUTPUT_DIR="outputs/qwen2.5-vl-roastme-v4-h100"

MIN_PIXELS=$((512 * 28 * 28))
MAX_PIXELS=$((1536 * 28 * 28))

export PYTHONPATH=src:$PYTHONPATH
unset LD_LIBRARY_PATH

# Kill any lingering processes
pkill -9 python || true
sleep 2

echo "📊 Training Configuration:"
echo "   Hardware: H100 SXM 80GB"
echo "   Strategy: FULL FINE-TUNING"
echo "   Batch Size: 4 (reduced from 6 to fit in memory)"
echo "   Resolution: High (512-1536 tokens)"
echo ""

deepspeed src/train/train_sft.py \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_ID \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --output_dir $OUTPUT_DIR \
    \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    \
    --learning_rate 2e-5 \
    --merger_lr 1e-5 \
    --vision_lr 1e-6 \
    \
    --image_min_pixels $MIN_PIXELS \
    --image_max_pixels $MAX_PIXELS \
    \
    --bf16 True \
    --use_liger True \
    \
    --lora_enable False \
    \
    --gradient_checkpointing False \
    \
    --max_seq_length 2048 \
    --dataloader_num_workers 8 \
    --logging_steps 5 \
    --save_strategy "steps" \
    --save_steps 25 \
    --save_total_limit 3 \
    --report_to tensorboard \
    --logging_dir $OUTPUT_DIR/logs

echo ""
echo "✅ Training Complete!"
