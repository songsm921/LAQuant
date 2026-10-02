"""AIME 2024 / AIME 2025 generation with vLLM (N samples per problem).

Saves the raw responses and ground truths to
{output_folder}/{model_name}{postfix}/{dataset}/{dataset}_{N}.parquet.
Grading is done afterwards by aggregate.py (deepscaler math grader).
"""
import argparse
import os

import datasets
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def load_dataset(name):
    if name == 'aime2025':
        return datasets.load_dataset("math-ai/aime25")['test']
    return datasets.load_dataset("Maxwell-Jia/AIME_2024")['train']


def generate(args, eval_dataset, output_dir):
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        top_k=-1 if args.DR1 else 20,
        max_tokens=args.max_new_tokens,
        n=args.N,
    )

    output_dict = {'responses': [], 'ground_truth': []}

    for example in tqdm(eval_dataset, desc=f"Generation for {args.dataset}...", dynamic_ncols=True):
        if args.dataset == 'aime2024':
            question = example['Problem']
            golden_answer = int(example['Answer'])
        else:
            question = example['problem']
            golden_answer = int(example['answer'])

        prompt = question + PROMPT if args.q_first else PROMPT + question
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if args.DR1:
            text = text + "<think>\n"

        outputs = llm.generate([text], sampling_params)
        output_dict['responses'].append([o.text for o in outputs[0].outputs])
        output_dict['ground_truth'].append(golden_answer)

    out_path = os.path.join(output_dir, f"{args.dataset}_{args.N}.parquet")
    pd.DataFrame(output_dict).to_parquet(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="aime2025", choices=["aime2024", "aime2025"])
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--N", type=int, default=16, help="Samples per problem")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--q_first", action="store_true",
                        help="Put the question before the instruction (used for all paper results)")
    parser.add_argument("--DR1", action="store_true",
                        help="DeepSeek-R1-Distill models: top_k=-1 and prefill '<think>\\n'")
    parser.add_argument("--output_folder", type=str, default="./generations")
    parser.add_argument("--postfix", type=str, default="")
    args = parser.parse_args()

    postfix = f"_{args.postfix}" if args.postfix else ""
    output_dir = os.path.join(
        args.output_folder, os.path.basename(args.model.rstrip('/')) + postfix, args.dataset,
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to {output_dir}")
    print("Temperature: ", args.temperature)

    generate(args, load_dataset(args.dataset), output_dir)
