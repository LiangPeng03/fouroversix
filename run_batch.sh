#!/bin/bash
# Batch evaluation script for FourOver6 (mirroring MR-GPTQ test.sh, dual-GPU)
# Usage: cd /path/to/fouroversix && bash run_batch.sh [GPU_IDS]
# Examples:
#   bash run_batch.sh 0,1      # dual GPU (default)
#   bash run_batch.sh 1,2      # dual GPU (other cards)

# --- 配置区 ---
GPU_IDS=${1:-0,1}
IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

LOG_FILE="4o6_eval_summary.log"
PYTHON_BIN="python"

MODELS=(
    "HuggingFaceTB/SmolLM2-135M"
    # "meta-llama/Llama-2-7b-hf"
    "Qwen/Qwen3-0.6B"
    "meta-llama/Meta-Llama-3-8B"
    "Qwen/Qwen3-8B"
    "meta-llama/Llama-3.2-1B"
    "Qwen/Qwen3-1.7B"
)

export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore"

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

print(f'{fmt(wiki,3)} | {fmt(c4,3)} | {fmt(piqa,4)} | {fmt(arc,4)} | {fmt(hella,4)} | {fmt(wino,4)} | {fmt(boolq,4)}')
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
printf "%-30s | %-12s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n" \
    "Model" "PTQ" "W-PPL" "C4-PPL" "PIQA" "ARC-C" "Hella" "WINO" "BoolQ" "Quant(min)" "Eval(min)" >> $LOG_FILE
echo "----------------------------------------------------------" >> $LOG_FILE

# --- Build task list ---
TASKS=()
for MODEL in "${MODELS[@]}"; do
    for PTQ in "awq"; do
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
            --task hellaswag --task arc_challenge \
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