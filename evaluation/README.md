# Evaluation

Every script takes the fake-quantized model directory written by `train.py`
(`--output_dir`), activates its own conda env, and can be called from any directory.

| Benchmark | Script | Conda env |
|---|---|---|
| AIME 2024 / 2025 | `aime/run_aime.sh` | `vllm` (generation), `laquant` (grading) |
| MMLU-Pro, GPQA-Diamond, LSAT-AR, LiveCodeBench | `lighteval/run_lighteval.sh` | `lighteval` |
| Perplexity (WikiText-2, C4) | `ppl/run_ppl.sh` | `laquant` |
| Zero-shot (BoolQ, ARC-C/E, HellaSwag) | `zero_shot/run_zero_shot.sh` | `lm-eval` |

Reasoning benchmarks (AIME, MMLU-Pro, GPQA, LSAT, LiveCodeBench) sample with
temperature 0.6, top_p 0.95, top_k 20 and up to 32768 new tokens.

## Setup

Same Docker container as training (`nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu24.04`,
see the top-level README), one conda env per tool:

| Conda env | Python | Requirements |
|---|---|---|
| `laquant` | 3.12 | `../requirements.txt` |
| `vllm` | 3.10 | `aime/requirements.txt` |
| `lighteval` | 3.12 | `lighteval/requirements.txt` |
| `lm-eval` | 3.12 | `zero_shot/requirements.txt` |

```bash
conda create -n vllm python=3.10 && conda activate vllm && pip install -r aime/requirements.txt
conda create -n lighteval python=3.12 && conda activate lighteval && pip install -r lighteval/requirements.txt
python lighteval/patch_lighteval_dp.py   # data-parallel fix for lighteval 0.12.0 (see below)
conda create -n lm-eval python=3.12 && conda activate lm-eval && pip install -r zero_shot/requirements.txt
```

## AIME 2024 / 2025

```bash
bash aime/run_aime.sh <model> aime2025                     # seeds 0 1 2 3, TP=4, N=16
bash aime/run_aime.sh <model> aime2024 4 "0 1 2 3"         # dataset, TP, seeds
bash aime/run_aime.sh <model> aime2024 4 "0 1 2 3" --DR1   # DeepSeek-R1-Distill models
```

- `eval_aime.py` generates 16 samples per problem with vLLM and saves them to
  `aime/generations/<model>_seed<S>/<dataset>/<dataset>_16.parquet`.
- `aggregate.py` (run at the end) grades them with the deepscaler math grader
  (`deepscaler_math.py`): a response is correct if it contains `</think>` and the
  last `\boxed{}` after it matches the answer under mathd normalization or sympy
  equivalence. It reports pass@1, pass@16, First10 (pass@1 over the first 10
  problems) and mean response length as mean ± std over seeds, and writes
  `aime/generations/<model>_<dataset>_summary.txt`.

## MMLU-Pro / GPQA-Diamond / LSAT-AR / LiveCodeBench

```bash
bash lighteval/run_lighteval.sh mmlu_pro <model>   # n=1
bash lighteval/run_lighteval.sh gpqa     <model>   # pass@1 over n=16
bash lighteval/run_lighteval.sh lsat     <model>   # pass@1 over n=16
bash lighteval/run_lighteval.sh lcb      <model>   # v4_v5, pass@1 over n=16, seeds 42 0 1
TP=8 DP=1 bash lighteval/run_lighteval.sh gpqa <large-model>
```

- GPQA-Diamond and LSAT-AR use the custom task definitions in `lighteval/tasks/`.
- GPQA-Diamond ([`Idavidrein/gpqa`](https://huggingface.co/datasets/Idavidrein/gpqa)) is a
  gated dataset: request access on its HF page, then log in once with `hf auth login`
  (in the `lighteval` env) before running.
- The default TP=1, DP=8 needs `patch_lighteval_dp.py`: stock lighteval 0.12.0
  forces the Ray executor whenever DP > 1, which hangs with TP=1.
- lighteval caches samples inside the model directory; the script clears that
  cache before and after each run so every run regenerates from scratch.
- Results go to `lighteval/results/<benchmark>/<model>[_seed<S>]/`.

## Perplexity

```bash
bash ppl/run_ppl.sh <model>          # WikiText-2 and C4, seq_len 8192
```

## Zero-shot

```bash
bash zero_shot/run_zero_shot.sh <model>                     # boolq,arc_challenge,arc_easy,hellaswag
bash zero_shot/run_zero_shot.sh <model> arc_challenge       # a subset
CUDA_VISIBLE_DEVICES=0,1,2,3 TP=4 bash zero_shot/run_zero_shot.sh <model>   # 4 GPUs (default TP=8)
```

Results go to `zero_shot/results/<model>/`.
