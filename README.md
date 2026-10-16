# LAQuant: A Simple Online Transform-free Large Reasoning Model Quantization by Layer-wise Lookahead Loss

**Paper:** [arXiv](https://arxiv.org/abs/2605.08755)

## News

- **2026.09.26**: Accepted to NeurIPS 2026 (Main Track).

## Repository layout

```
laquant/
├── train.py      entry point: args, per-layer training loop, main
├── data.py       calibration data loading/tokenization, batching, hidden-state cache
├── checkpoint.py final checkpoint saving, resumable per-layer checkpoints
├── utils.py      distributed setup, memory management, logging, losses
├── qlinear.py    QLinear (fake-quant linear with LWC + channel scaling)
└── quantizer.py  STE integer asymmetric weight quantizer
scripts/          training launch scripts
evaluation/       AIME, MMLU-Pro, GPQA, LSAT, LiveCodeBench, perplexity, zero-shot
```

## Setup

All experiments were run in the `nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu24.04`

Training runs in the `laquant` conda env:

```bash
conda create -n laquant python=3.12
conda activate laquant
pip install -r requirements.txt
```

Evaluation uses separate envs (`vllm`, `lighteval`, `lm-eval`); see
[evaluation/README.md](evaluation/README.md).

## Data

The calibration set is the sampled OpenR1-Math data used in the paper, on the HF Hub:
[`sumin99/Openr1-dataset-sampled-laquant`](https://huggingface.co/datasets/sumin99/Openr1-dataset-sampled-laquant).
Download it once before training (~530MB, 32,768 samples):

```bash
conda activate laquant
hf download sumin99/Openr1-dataset-sampled-laquant --repo-type dataset \
    --local-dir ./data/Openr1-dataset-sampled-laquant \
    --include "dataset_dict.json" "train/data-*.arrow" "train/state.json" "train/dataset_info.json"
```

The dataset is stored in `save_to_disk` format and is loaded with `load_from_disk`, so
`--dataset_path` must be this local directory (the training scripts use
`./data/Openr1-dataset-sampled-laquant` by default). The `--include` filter downloads
only the dataset files.
By default the first `--max_samples` samples are used, truncated to `--max_seq_length`
tokens from the start of each sample.

## Training

Main configuration (Qwen3-1.7B, W3G128 / W4G128, W=2, smooth-L1, 8 GPUs):

```bash
conda activate laquant
bash scripts/Qwen3-laquant-w3.sh    # -> ./outputs/Qwen3-1.7B-w3-laquant
bash scripts/Qwen3-laquant-w4.sh    # -> ./outputs/Qwen3-1.7B-w4-laquant
# optional overrides (OUTPUT_DIR defaults to ./outputs/<model name>-w{3,4}-laquant):
MODEL=/path/to/local/Qwen3-1.7B DATASET=/path/to/Openr1-dataset-sampled-laquant \
    OUTPUT_DIR=./outputs/my_run MASTER_PORT=29501 bash scripts/Qwen3-laquant-w3.sh
```

The scripts can be launched from any directory (paths are relative to the repo root)
and also write the console output to `<output_dir>/train_log.txt`. The W3 script runs
(the W4 script differs only in `--w_bit 4` and the output directory); run it from the
repo root so that the `laquant` package is importable:

```bash
torchrun --nproc_per_node=8 -m laquant.train \
    --student_model Qwen/Qwen3-1.7B --teacher_model Qwen/Qwen3-1.7B \
    --dataset_path ./data/Openr1-dataset-sampled-laquant \
    --w_bit 3 --w_group_size 128 --init_scale_mode minmax \
    --enable_lwc --enable_channel_scale --no_freeze_weight \
    --quant_lr 1e-4 --weight_lr 5e-5 --loss_mode smooth_l1 \
    --num_epochs 20 --window_size 2 \
    --max_seq_length 2048 --max_samples 4096 --batch_size 64 \
    --disk_offload \
    --output_dir ./outputs/Qwen3-1.7B-w3-laquant --seed 42
```

`--batch_size` is the global batch size, split across GPUs. `--disk_offload` keeps
the cached hidden states on local disk (`/tmp`) instead of RAM.

### Outputs

```
<output_dir>/
├── model.safetensors, config.json, tokenizer files   fake-quantized model (use this for evaluation)
├── quant_params.pt       per-module scales, zeros, LWC and channel-scale params
├── loss_log.txt          per-layer / per-epoch training log
└── train_log.txt         console output (written by the launch scripts)
```

With `--save_pre_baked`, the continuous (pre-quantization) weights are also saved to
`<output_dir>/pre-baked/`. They are only needed to continue QAT from a finished run,
e.g. `--student_model <run>/pre-baked --init_scale_mode checkpoint --checkpoint_qparams_file <run>/quant_params.pt`.

The fake-quantized model is a standard HF checkpoint (weights already rounded to
the quantization grid, with channel scales folded into the norms), so it loads
directly in vLLM / transformers.

## Evaluation

See [evaluation/README.md](evaluation/README.md). For example:

```bash
bash evaluation/aime/run_aime.sh ./outputs/Qwen3-1.7B-w3-laquant aime2025
bash evaluation/lighteval/run_lighteval.sh gpqa ./outputs/Qwen3-1.7B-w3-laquant
```
