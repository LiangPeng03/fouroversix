#!/bin/bash
# 修复 RTX 5090 与 PyTorch 2.3.0 不兼容的问题

conda activate 4o6

# 1. 升级 PyTorch 到 2.7.1 + CUDA 12.8（支持 sm_120 / Blackwell）
pip install --force-reinstall torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir

# 2. 重新安装 fouroversix（它本身要求 torch>=2.7.0）
pip install -e /home/pengliang/Desktop/fouroversix --no-build-isolation

# 3. 可选：重新编译 awq 的 C++ kernel（如果后续 AWQ 报错）
# cd /home/pengliang/Desktop/fouroversix/third_party/llm-awq
# pip install -e . --no-build-isolation

# 4. 验证
echo "===== PyTorch Version ====="
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda)"
echo "===== CUDA Available ====="
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0))"
