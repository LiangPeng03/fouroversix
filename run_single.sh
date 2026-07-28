#!/bin/bash
# Single-model evaluation script for FourOver6 (mirroring MR-GPTQ nv.sh)
# Usage: cd /path/to/fouroversix && bash run_single.sh

gpu_id=1
export CUDA_VISIBLE_DEVICES=$gpu_id
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"

# --- Model list (modify MODEL=$MODELx to switch) ---
MODEL1="HuggingFaceTB/SmolLM2-135M"
MODEL2="meta-llama/Llama-2-7b-hf"
MODEL3="meta-llama/Meta-Llama-3-8B"
MODEL4="Qwen/Qwen3-0.6B"
MODEL5="Qwen/Qwen3-8B"
MODEL6="meta-llama/Llama-3.2-1B"
MODEL7="Qwen/Qwen3-1.7B"

MODEL=$MODEL7
SEED=0

# --- Inline Python helper to parse stdout results ---
parse_results() {
python3 -c "
import sys, re
text = sys.stdin.read()
lines = text.split('\n')
results = {}
for i, line in enumerate(lines):
    line = line.strip()
    if not line:
        continue
    parts = line.split()
    if len(parts) == 3 and i + 1 < len(lines):
        model, ptq, task = parts
        next_line = lines[i + 1].strip()
        try:
            d = eval(next_line)
            if isinstance(d, dict):
                results[task] = d
        except:
            pass

def get_metric(task, keys):
    d = results.get(task, {})
    for k in keys:
        if k in d:
            return d[k]
    return None

def fmt(v, p):
    if v is None:
        return 'N/A'
    return f'{v:.{p}f}'

wiki = get_metric('wikitext2', ['perplexity,none'])
c4   = get_metric('c4', ['perplexity,none'])
piqa = get_metric('piqa', ['acc_norm,none', 'acc,none'])
arc  = get_metric('arc_challenge', ['acc_norm,none', 'acc,none'])
hella = get_metric('hellaswag', ['acc_norm,none', 'acc,none'])
wino = get_metric('winogrande', ['acc_norm,none', 'acc,none'])
boolq = get_metric('boolq', ['acc_norm,none', 'acc,none'])

print(f'Wiki={fmt(wiki,3)} | C4={fmt(c4,3)} | PIQA={fmt(piqa,4)} | ARC={fmt(arc,4)} | Hella={fmt(hella,4)} | Wino={fmt(wino,4)} | BoolQ={fmt(boolq,4)}')
"
}

parse_times() {
python3 -c "
import sys
text = sys.stdin.read()
lines = text.split('\n')
quant_time = None
eval_time = None
for line in lines:
    line = line.strip()
    if line.startswith('QUANT_TIME:'):
        quant_time = float(line.split(':', 1)[1].strip())
    elif line.startswith('EVAL_TIME:'):
        eval_time = float(line.split(':', 1)[1].strip())
print(f'Quant={(quant_time or 0)/60:.2f}min | Eval={(eval_time or 0)/60:.2f}min')
"
}

eval_model() {
    local PTQ=$1
    local DTYPE_ARG=$2
    local LABEL=$3

    echo "========================================="
    echo "Running $LABEL for $MODEL"
    echo "========================================="

    tmp="tmp_eval_${LABEL}_$$.out"
    python -m scripts.ptq_hf \
        --model-name "$MODEL" \
        --ptq-method "$PTQ" \
        --seed $SEED \
        $DTYPE_ARG \
        --quantize-backend cuda \
        --matmul-backend cutlass \
        --task wikitext2 \
        --task c4 \
        --task piqa --task winogrande --task boolq \
        --task hellaswag --task arc_challenge \
        > "$tmp" 2>&1
    

    if [ $? -ne 0 ]; then
        echo "$MODEL | $LABEL | FAILED"
        cat "$tmp"
        rm -f "$tmp"
        return
    fi

    # Parse and print summary line
    summary=$(cat "$tmp" | parse_results)
    times=$(cat "$tmp" | parse_times)
    echo "Result: $summary"
    echo "Time: $times"

    rm -f "$tmp"
}

# --- FP16 (high_precision) baseline ---
# eval_model "high_precision" "--dtype bfloat16" "FP16"

# --- awq + FourOver6 (W4A4) ---
eval_model "awq" "--dtype nvfp4 --a-dtype nvfp4" "awq+W4A4"

echo "========================================="
echo "Evaluation finished!"
echo "========================================="
