#!/bin/bash
# WikiText-2 / C4 perplexity (seq_len 8192 by default, single GPU).
#
# Usage: bash run_ppl.sh <model_path> [seq_len=8192]
set -e

MODEL="$(realpath "${1:?Usage: bash run_ppl.sh <model_path> [seq_len]}")"
SEQ_LEN="${2:-8192}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate laquant

cd "$(dirname "$0")"
python eval_ppl.py --model "${MODEL}" --seq-len "${SEQ_LEN}"
