#!/bin/bash
# MMLU-Pro / GPQA-Diamond / LSAT-AR / LiveCodeBench (v4_v5) with lighteval + vLLM.
#
# Usage: bash run_lighteval.sh <benchmark> <model_path> [seeds]
#   benchmark: mmlu_pro | gpqa | lsat | lcb
#   seeds:     default "42" (lcb: "42 0 1")
# Env:
#   TP (default 1), DP (default 8): vLLM tensor / data parallel sizes.
#   TP=1 with DP>1 needs the DP patch on the lighteval install (patch_lighteval_dp.py,
#   already applied to the `lighteval` conda env). For large models use e.g. TP=8 DP=1.
#
# Generation: temperature 0.6, top_p 0.95, top_k 20, max_new_tokens 32768.
# GPQA / LSAT / LCB report pass@1 over n=16 samples; MMLU-Pro uses n=1.
# Results: results/<benchmark>/<model_name>[_seed<S>]/
set -e

BENCH="${1:?Usage: bash run_lighteval.sh <mmlu_pro|gpqa|lsat|lcb> <model_path> [seeds]}"
MODEL="$(realpath "${2:?model_path required}")"
TP="${TP:-1}"
DP="${DP:-8}"

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
CUSTOM_TASKS=()
case "${BENCH}" in
    mmlu_pro) TASK="lighteval|mmlu_pro|0|0" ;;
    gpqa)     TASK="custom|gpqa:diamond_n16|0"; CUSTOM_TASKS=(--custom-tasks "$EVAL_DIR/tasks/gpqa_custom.py") ;;
    lsat)     TASK="custom|lsat_ar:n16|0";      CUSTOM_TASKS=(--custom-tasks "$EVAL_DIR/tasks/lsat_ar_custom.py") ;;
    lcb)      TASK="lighteval|lcb:codegeneration|0|0" ;;
    *) echo "Unknown benchmark: ${BENCH}"; exit 1 ;;
esac
if [ "${BENCH}" = "lcb" ]; then SEEDS="${3:-42 0 1}"; else SEEDS="${3:-42}"; fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lighteval

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((TP * DP - 1)))}
MODEL_SHORT=$(basename "${MODEL}")
OUTPUT_BASE="$EVAL_DIR/results/${BENCH}"

# lighteval caches samples under <MODEL>/<16-hex model hash>/ (the cache_dir is ignored for
# absolute model paths). Clear it so every run regenerates from scratch.
clear_cache() {
    find "${MODEL}" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended \
        -regex '.+/[0-9a-f]{16}$' -exec rm -rf {} + 2>/dev/null || true
}
# Also clear it (and the temp config) if lighteval fails or the run is interrupted.
YAML_FILE=""
trap 'clear_cache; [ -n "${YAML_FILE}" ] && rm -f "${YAML_FILE}"' EXIT

for SEED in ${SEEDS}; do
    RUN_TAG="${MODEL_SHORT}"
    [ "$(echo ${SEEDS} | wc -w)" -gt 1 ] && RUN_TAG="${MODEL_SHORT}_seed${SEED}"
    echo "========================================="
    echo "Running ${BENCH}: ${RUN_TAG} (TP=${TP}, DP=${DP})"
    echo "========================================="

    YAML_FILE=$(mktemp /tmp/lighteval_XXXXXX.yaml)
    cat > "$YAML_FILE" <<EOF
model_parameters:
    model_name: "${MODEL}"
    revision: "main"
    dtype: "bfloat16"
    tensor_parallel_size: ${TP}
    data_parallel_size: ${DP}
    gpu_memory_utilization: 0.9
    max_model_length: 40960
    seed: ${SEED}
    trust_remote_code: True
    add_special_tokens: True
    generation_parameters:
      temperature: 0.6
      top_k: 20
      top_p: 0.95
      seed: ${SEED}
      max_new_tokens: 32768
EOF

    clear_cache
    lighteval vllm \
        "$YAML_FILE" \
        "$TASK" \
        "${CUSTOM_TASKS[@]}" \
        --output-dir "$OUTPUT_BASE" \
        --results-path-template "{output_dir}/${RUN_TAG}" \
        --save-details
    clear_cache

    rm -f "$YAML_FILE"
    echo "Done: ${RUN_TAG}"
done
