#!/bin/bash
# Zero-shot commonsense tasks (BoolQ, ARC-Challenge, ARC-Easy, HellaSwag) with lm-eval + vLLM.
#
# Usage: bash run_zero_shot.sh <model_path> [tasks]
#   tasks: comma-separated lm-eval task names (default: boolq,arc_challenge,arc_easy,hellaswag)
# Env: TP (default 8)
# Results: results/<model_name>/
set -e

MODEL="$(realpath "${1:?Usage: bash run_zero_shot.sh <model_path> [tasks]}")"
TASKS="${2:-boolq,arc_challenge,arc_easy,hellaswag}"
TP="${TP:-8}"

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_SHORT=$(basename "${MODEL}")

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lm-eval

echo "Running zero-shot (${TASKS}): ${MODEL_SHORT}"
lm-eval \
    --model vllm \
    --model_args "pretrained=${MODEL},tensor_parallel_size=${TP},dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096,trust_remote_code=True,seed=42" \
    --tasks "${TASKS}" \
    --num_fewshot 0 \
    --batch_size auto \
    --gen_kwargs "temperature=0.7,top_p=0.8,top_k=20,min_p=0.0,max_gen_toks=16384" \
    --output_path "${EVAL_DIR}/results/${MODEL_SHORT}" \
    --log_samples \
    --seed 42
