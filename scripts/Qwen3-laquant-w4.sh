#!/bin/bash
# Main config: Qwen3-1.7B, W4G128, window size 2, smooth-L1 loss, LWC + per-channel scaling.
# Run inside the `laquant` conda env (see README).
set -eo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3-1.7B}
DATASET=${DATASET:-./data/Openr1-dataset-sampled-laquant}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/$(basename "${MODEL}")-w4-laquant}
mkdir -p "${OUTPUT_DIR}"

torchrun --nproc_per_node=8 --master_port=${MASTER_PORT:-29500} -m laquant.train \
    --student_model "${MODEL}" \
    --teacher_model "${MODEL}" \
    --dataset_path "${DATASET}" \
    --w_bit 4 \
    --w_group_size 128 \
    --init_scale_mode minmax \
    --enable_lwc \
    --enable_channel_scale \
    --no_freeze_weight \
    --quant_lr 1e-4 \
    --weight_lr 5e-5 \
    --loss_mode smooth_l1 \
    --num_epochs 20 \
    --window_size 2 \
    --max_seq_length 2048 \
    --max_samples 4096 \
    --batch_size 64 \
    --disk_offload \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42 \
    2>&1 | tee "${OUTPUT_DIR}/train_log.txt"
