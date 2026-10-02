#!/bin/bash
# AIME 2024 / 2025 (N=16 samples per problem) over multiple seeds, then grade + aggregate.
# Generation runs in the `vllm` env; grading (deepscaler math grader) in the `laquant` env.
#
# Usage: bash run_aime.sh <model_path> [dataset=aime2025] [tp_size=4] [seeds="0 1 2 3"] [extra eval_aime.py args...]
# Example:
#   bash run_aime.sh ../../outputs/Qwen3-1.7B-w3-laquant aime2025
#   bash run_aime.sh ../../outputs/Qwen3-1.7B-w3-laquant aime2024 4 "0"     # single seed
#   bash run_aime.sh /path/to/R1-Llama-8B-quant aime2024 4 "0 1 2 3" --DR1   # DeepSeek-R1-Distill models
set -e

MODEL="${1:?Usage: bash run_aime.sh <model_path> [dataset] [tp_size] [seeds] [extra args...]}"
DATASET="${2:-aime2025}"
TP_SIZE="${3:-4}"
SEEDS="${4:-0 1 2 3}"
shift $(( $# < 4 ? $# : 4 ))
EXTRA_ARGS=("$@")
MODEL="$(realpath "${MODEL}")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm

cd "$(dirname "$0")"

for SEED in ${SEEDS}; do
    echo "===== ${DATASET} seed ${SEED}: ${MODEL} ====="
    python eval_aime.py \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --tensor_parallel_size "${TP_SIZE}" \
        --N 16 \
        --q_first \
        --seed "${SEED}" \
        --postfix "seed${SEED}" \
        "${EXTRA_ARGS[@]}"
done

conda activate laquant
python aggregate.py --model "${MODEL}" --dataset "${DATASET}" --seeds ${SEEDS} --N 16
