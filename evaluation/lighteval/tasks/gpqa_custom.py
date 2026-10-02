import random
from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig


def record_to_sample(record):
    gold_index = random.randint(0, 3)
    choices = [record["Incorrect Answer 1"], record["Incorrect Answer 2"], record["Incorrect Answer 3"]]
    choices.insert(gold_index, record["Correct Answer"])
    return Sample(
        input=record["Question"].strip(),
        choices=choices,
        target=ascii_uppercase[gold_index],
    )


def sample_to_fewshot(sample):
    return f"{sample.input}\n\n" + f"ANSWER: {sample.target}"


gpqa_diamond_n16 = LightevalTaskConfig(
    name="gpqa:diamond_n16",
    suite=["custom"],
    prompt_function=prompt.gpqa_instruct,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[multiple_choice(cache=True)],
    scorer=choice(),
    hf_repo="Idavidrein/gpqa",
    hf_subset="gpqa_diamond",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metrics=[
        Metrics.gpqa_instruct_pass_at_k(sample_params={"k": 1, "n": 16}),
        Metrics.gpqa_instruct_pass_at_k(sample_params={"k": 16, "n": 16}),
    ],
    stop_sequence=[],
    version=1,
)

TASKS_TABLE = [gpqa_diamond_n16]
