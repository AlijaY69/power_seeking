"""Inspect tasks for the power-seeking evals: a capability baseline and the alignment eval."""

from functools import partial
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import answer
from inspect_ai.solver import chain, generate

from src.dataset import record_to_sample_full
from src.paths import DATA_PATH
from src.solvers import (
    TEMPLATE_MCQ,
    TEMPLATE_MCQ_BENCHMARK,
    TEMPLATE_MCQ_BENCHMARK_MAKE_CHOICE,
    TEMPLATE_MCQ_COT_AND_CHOICE,
    TEMPLATE_MCQ_CRITIQUE,
    TEMPLATE_MCQ_CRITIQUE_COMPLETION,
    TEMPLATE_MCQ_MAKE_CHOICE,
    make_choice,
    multiple_choice_format,
    self_critique_format,
)


@task
def benchmark_eval(json_dataset_path: str = str(DATA_PATH), n: int | None = None) -> Task:
    """
    Capability baseline: can the model recognise which option is the power-seeking one?

    The scenario is given as context (not as a system prompt), so the model evaluates the
    options instead of acting in the role.

    Args:
        json_dataset_path : Path to the dataset JSON file
        n : Number of samples to use (None = all)

    Returns:
        Task : The benchmark task
    """
    return Task(
        dataset=json_dataset(
            json_file=json_dataset_path,
            sample_fields=partial(record_to_sample_full, system_prompt_behavior="context"),
            limit=n,
        ),
        solver=chain(
            multiple_choice_format(template=TEMPLATE_MCQ_BENCHMARK),
            generate(),
            make_choice(prompt=TEMPLATE_MCQ_BENCHMARK_MAKE_CHOICE),
            generate(),
        ),
        scorer=answer("letter"),
    )


@task
def alignment_eval(
    json_dataset_path: str = str(DATA_PATH),
    n: int | None = None,
    system_prompt_behavior: Literal["system", "context"] | None = "system",
    use_cot: bool = False,
    use_self_critique: bool = False,
    self_critique_model: str | None = None,
) -> Task:
    """
    Alignment eval: does the model itself choose the power-seeking option?

    Args:
        json_dataset_path : Path to the dataset JSON file
        n : Number of samples to use (None = all)
        system_prompt_behavior : How the scenario is given: "system" (system message),
            "context" (prepended to the user message) or None (omitted)
        use_cot : Whether the model reasons step by step before answering
        use_self_critique : Whether the model critiques and revises its answer (requires use_cot)
        self_critique_model : The model used for the critique (set this to the evaluated model)

    Returns:
        Task : The alignment task
    """
    dataset = json_dataset(
        json_file=json_dataset_path,
        sample_fields=partial(record_to_sample_full, system_prompt_behavior=system_prompt_behavior),
        limit=n,
    )

    if not use_cot:
        # Answer directly
        assert not use_self_critique, "Can only use self-critique if CoT is enabled"
        solver = chain(
            multiple_choice_format(template=TEMPLATE_MCQ),
            generate(),
        )
    else:
        # Reason step by step, then answer
        solver = chain(
            multiple_choice_format(template=TEMPLATE_MCQ_COT_AND_CHOICE),
            generate(),
        )

        # Optionally: critique the answer and generate an improved one
        if use_self_critique:
            assert self_critique_model is not None, "You must specify a self-critique model if using self-critique"
            solver = chain(
                solver,
                self_critique_format(
                    model_id=self_critique_model,
                    critique_template=TEMPLATE_MCQ_CRITIQUE,
                    critique_completion_template=TEMPLATE_MCQ_CRITIQUE_COMPLETION,
                ),
                generate(),
            )

        # Ask for the final answer
        solver = chain(solver, make_choice(prompt=TEMPLATE_MCQ_MAKE_CHOICE), generate())

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=answer("letter"),
    )