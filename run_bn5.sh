#!/bin/bash
# Batch evaluation script for FourOver6 - New 5 downstream tasks (no wikitext2/c4)
# Usage: cd /path/to/fouroversix && bash run_batch_new5.sh [GPU_IDS]
# Examples:
#   bash run_batch_new5.sh 0,1      # dual GPU (default)
#   bash run_batch_new5.sh 1,2      # dual GPU (other cards)

# --- 配置区 ---
GPU_IDS=${1:-0,1}
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

LOG_FILE="4o6_eval_new5_summary.log"
PYTHON_BIN="python"

MODELS=(
    "HuggingFaceTB/SmolLM2-135M"
    # "meta-llama/Llama-2-7b-hf"
    "meta-llama/Meta-Llama-3-8B"
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-8B"
    "meta-llama/Llama-3.2-1B"
    "Qwen/Qwen3-1.7B"
)

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"
export HF_ALLOW_CODE_EVAL="1" 

# --- Inline Python helper to parse stdout results ---
parse_results() {
python3 -c "
import sys, re
import numpy as np
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

gsm8k  = get_metric('gsm8k', ['exact_match,strict-match', 'exact_match,none'])
mmlu   = get_metric('mmlu', ['acc_norm,none', 'acc,none'])
arc_e  = get_metric('arc_easy', ['acc_norm,none', 'acc,none'])
mbpp   = get_metric('mbpp', ['pass_at_1,none', 'pass@1,none'])
lambada = get_metric('lambada_openai', ['acc,none', 'acc_norm,none'])

print(f'{fmt(gsm8k,4)} | {fmt(mmlu,4)} | {fmt(arc_e,4)} | {fmt(mbpp,4)} | {fmt(lambada,3)}')
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
print(f'{quant_time/60:.2f} | {eval_time/60:.2f}')
"
}

# --- Initialize log file ---
echo "" >> $LOG_FILE
echo "==========================================================" >> $LOG_FILE
echo "Batch Run Started at: $(date) on GPUs: $GPU_IDS" >> $LOG_FILE
printf "%-30s | %-12s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n" \
    "Model" "PTQ" "GSM8K" "MMLU" "ARC-E" "MBPP" "LAMBADA" "Quant(min)" "Eval(min)" >> $LOG_FILE
echo "----------------------------------------------------------" >> $LOG_FILE

# --- Build task list ---
TASKS=()
for MODEL in "${MODELS[@]}"; do
    for PTQ in  "awq"; do
        TASKS+=("$MODEL|$PTQ")
    done
done

# --- Concurrent dispatch ---
for (( i=0; i<${#TASKS[@]}; i++ )); do
    TASK="${TASKS[$i]}"
    IFS='|' read -r MODEL PTQ <<< "$TASK"

    GPU_IDX=$((i % NUM_GPUS))
    GPU_ID=${GPUS[$GPU_IDX]}

    echo "----------------------------------------------------------"
    echo "Dispatching: $MODEL | PTQ: $PTQ -> GPU $GPU_ID"

    (
        export CUDA_VISIBLE_DEVICES=$GPU_ID

        DTYPE_ARG=""
        if [ "$PTQ" = "high_precision" ]; then
            DTYPE_ARG="--dtype bfloat16"
        elif [ "$PTQ" = "awq" ]; then
            DTYPE_ARG="--dtype nvfp4 --a-dtype nvfp4"
        fi

        tmp="tmp_eval_${PTQ}_${GPU_ID}_$$.out"
        $PYTHON_BIN -m scripts.ptq_hf \
            --model-name "$MODEL" \
            --ptq-method "$PTQ" \
            --seed 0 \
            $DTYPE_ARG \
            --quantize-backend cuda \
            --matmul-backend cutlass \
            --task gsm8k --task mmlu --task arc_easy \
            --task mbpp --task lambada_openai \
            > "$tmp" 2>&1

        if [ $? -ne 0 ]; then
            echo "$MODEL | $PTQ | FAILED (GPU $GPU_ID)"
            cat "$tmp"
            rm -f "$tmp"
            exit 1
        fi

        # Parse results
        summary=$(cat "$tmp" | parse_results)
        times=$(cat "$tmp" | parse_times)
        echo "Result ($PTQ on GPU $GPU_ID): $summary"
        echo "Time: Quant=$times min"

        # Write to log with flock to avoid concurrent conflicts
        (
            flock -x 200
            printf "%-30s | %-12s | %s | %s\n" "$MODEL" "$PTQ" "$summary" "$times" >> $LOG_FILE
        ) 200> "${LOG_FILE}.lock"

        rm -f "$tmp"
    ) &

    # Full-load control: wait when dispatched tasks reach NUM_GPUS
    if (( (i + 1) % NUM_GPUS == 0 )); then
        wait
    fi
done

# Wait for remaining background jobs
wait

echo "==========================================================" >> $LOG_FILE
echo "" >> $LOG_FILE
echo "All Tasks Finished."
