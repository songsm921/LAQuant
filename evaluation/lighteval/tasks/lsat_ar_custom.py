from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


LETTER_INDICES = list(ascii_uppercase)


def lsat_ar_instruct(line, task_name: str = None):
    """Generative prompt for LSAT-AR (analytical reasoning) multiple choice."""
    choices = line["choices"]
    num_choices = len(choices)

    instruction = (
        "Answer the following multiple choice question. "
        f"The last line of your response should be of the following format: "
        f"'Answer: $LETTER' (without quotes) where LETTER is one of {','.join(LETTER_INDICES[:num_choices])}. "
        "Think step by step before answering."
    )

    # The query already contains the question and "A: Among A through E, the answer is"
    # Strip the trailing answer prompt since we use our own instruction format
    query_text = line["query"]
    if "\nA: Among" in query_text:
        query_text = query_text[: query_text.index("\nA: Among")]

    # Format choices
    choice_lines = []
    for i, c in enumerate(choices):
        # Choices already start with (A), (B), etc. - strip and re-format
        text = c.strip()
        if text.startswith(f"({LETTER_INDICES[i]})"):
            text = text[3:].strip()
        choice_lines.append(f"{LETTER_INDICES[i]}) {text}")

    query = f"{instruction}\n\n{query_text}\n\n" + "\n".join(choice_lines)

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES[:num_choices],
        gold_index=line["gold"][0],
        instruction=instruction,
    )


lsat_ar_n16 = LightevalTaskConfig(
    name="lsat_ar:n16",
    suite=["custom"],
    prompt_function=lsat_ar_instruct,
    hf_repo="dmayhem93/agieval-lsat-ar",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
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

TASKS_TABLE = [lsat_ar_n16]
