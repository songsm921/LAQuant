"""Grade AIME generations with the deepscaler math grader and gather scores across seeds.

A response is correct iff it contains </think>, the text after </think> has a
\\boxed{} answer (last one), and that answer matches the ground truth under
mathd normalization or sympy equivalence.

Per seed: pass@1 (mean over all N samples) ± std over the N sample columns,
first10, pass@N and mean response length (tokens). Then mean ± std over seeds.

Usage: python aggregate.py --model <model_path> --dataset aime2025 --seeds 0 1 2 3
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import transformers

from deepscaler_math import extract_answer, grade_answer_mathd, grade_answer_sympy

THOUGHT_DELIMITER_END = "</think>"


def deepscaler_is_correct(model_response, ground_truth):
    if THOUGHT_DELIMITER_END not in model_response:
        return False
    model_answer = extract_answer(model_response.split(THOUGHT_DELIMITER_END)[1])
    if model_answer is None:
        return False

    truth = str(ground_truth)
    if "\\boxed" in truth:
        truth = extract_answer(truth)
        if truth is None:
            return False
    return grade_answer_mathd(model_answer, truth) or grade_answer_sympy(model_answer, truth)


_tokenizer = None


def _init_worker(tokenizer_path):
    global _tokenizer
    _tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)


def _grade_row(row):
    responses, ground_truth, n = row
    correct, lengths = [], []
    for r in responses[:n]:
        correct.append(int(deepscaler_is_correct(r, ground_truth)))
        lengths.append(len(_tokenizer(r).input_ids))
    correct += [0] * (n - len(correct))
    lengths += [0] * (n - len(lengths))
    return correct, lengths


def grade_parquet(path, n, tokenizer_path, workers):
    df = pd.read_parquet(path)
    rows = [(df['responses'][i], df['ground_truth'][i], n) for i in range(len(df))]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(tokenizer_path,)) as ex:
        results = list(ex.map(_grade_row, rows))
    correct = np.array([c for c, _ in results])
    lengths = np.array([l for _, l in results])
    return {
        'pass1': correct.mean() * 100,
        'pass1_std': correct.mean(axis=0).std() * 100,
        'first10': correct[:10].mean() * 100,
        'passN': (correct.sum(axis=-1) > 0).mean() * 100,
        'avg_len': lengths.mean(),
        'per_q': correct.sum(axis=-1).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="aime2025")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--output_folder", type=str, default="./generations")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer for response lengths (default: --model)")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    model_name = os.path.basename(args.model.rstrip('/'))
    results = []
    for seed in args.seeds:
        path = os.path.join(args.output_folder, f"{model_name}_seed{seed}",
                            args.dataset, f"{args.dataset}_{args.N}.parquet")
        if not os.path.exists(path):
            print(f"Seed {seed}: parquet not found at {path}")
            continue
        r = grade_parquet(path, args.N, args.tokenizer or args.model, args.workers)
        r['seed'] = seed
        results.append(r)
        print(f"Seed {seed}: pass@1={r['pass1']:.2f}% ± {r['pass1_std']:.2f}%, "
              f"first10={r['first10']:.2f}%, pass@{args.N}={r['passN']:.2f}%, "
              f"avg_tokens={r['avg_len']:.0f}")

    if not results:
        raise SystemExit("No results found!")

    lines = [f"Model: {args.model}",
             f"Seeds: {[r['seed'] for r in results]}, Dataset: {args.dataset}, N: {args.N}, "
             f"grader: deepscaler"]
    for key, label in [('pass1', 'Pass@1'), ('first10', 'First10'), ('passN', f'Pass@{args.N}')]:
        v = [r[key] for r in results]
        lines.append(f"{label:8s} mean={np.mean(v):.2f}% ± {np.std(v):.2f}%  "
                     f"[{min(v):.2f}%, {max(v):.2f}%]")
    lines.append(f"Avg tokens: {np.mean([r['avg_len'] for r in results]):.0f}")
    print("\n" + "=" * 60 + "\n" + "\n".join(lines) + "\n" + "=" * 60)

    print("\nPer-question average (across seeds):")
    for qi in range(len(results[0]['per_q'])):
        c = [r['per_q'][qi] for r in results]
        print(f"  Q{qi+1:2d}: {np.mean(c):.1f}/{args.N} ± {np.std(c):.1f}")

    summary_path = os.path.join(args.output_folder, f"{model_name}_{args.dataset}_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
