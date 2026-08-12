#!/bin/bash
# Qwen3-32B AWQ + Four-over-Six W4A4 evaluation.
# Run from the Four-over-Six repository root.

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore
export HF_ALLOW_CODE_EVAL=1
export PYTHONUNBUFFERED=1

# MODEL="/home/pengliang/models/Meta-Llama-3-70B"
MODEL="/home/pengliang/models/Qwen3-32B"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/fouroversix_llama_awq_w4a4.log"
exec > >(tee -a "$LOG_FILE") 2>&1

python -u -m scripts.ptq_hf \
    --model-name "$MODEL" \
    --ptq-method awq \
    --seed 42 \
    --dtype nvfp4 \
    --a-dtype nvfp4 \
    --quantize-backend cuda \
    --matmul-backend cutlass \
    --task wikitext2 \
    --task c4 
    # --task piqa \
    # --task winogrande \
    # --task boolq \
    # --task hellaswag \
    # --task arc_challenge
