#!/bin/bash
set -e

echo "=========================================="
echo "🚀 DissTrack Training Setup (RunPod)"
echo "=========================================="
echo ""

# Fix libcudnn error (from original README)
unset LD_LIBRARY_PATH

echo "🔧 Upgrading pip..."
pip install --upgrade pip wheel

echo "📦 Installing PyTorch (CUDA 12.8)..."
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

echo "⚡ Installing Flash Attention..."
MAX_JOBS=4 pip install flash-attn --no-build-isolation

echo "📚 Installing Qwen utilities..."
pip install qwen-vl-utils

echo "📦 Installing remaining dependencies (ignoring system packages)..."
pip install -r requirements.txt --ignore-installed pyparsing --ignore-installed setuptools

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "🧪 Testing installation..."
python3 -c "
import torch
import transformers
import deepspeed
import flash_attn
from qwen_vl_utils import process_vision_info

print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ CUDA devices:', torch.cuda.device_count())
print('✓ Transformers:', transformers.__version__)
print('✓ DeepSpeed:', deepspeed.__version__)
print('✓ Flash Attention: OK')
print('✓ Qwen VL Utils: OK')
print('')
print('🎉 All systems ready!')
"

echo ""
echo "🎯 Next steps:"
echo "   1. Upload data: see upload instructions"
echo "   2. Run: bash scripts/finetune_roastme.sh"
echo ""
