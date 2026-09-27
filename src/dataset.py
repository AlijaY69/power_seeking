"""Converts dataset records into Inspect Samples."""

from typing import Literal

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser


def record_to_sample_full(record: dict, system_prompt_behavior: Literal["system", "context"] | None = None) -> Sample:
    """
    Converts a record from the dataset into a Sample object.

    Answer options are used in the order stored in the file: the dataset was balanced so that
    the power-seeking option appears at each letter equally often (see data/DATASET.md).

    Args:
        record : A dictionary from the JSON dataset
        system_prompt_behavior : How to use the record's system prompt:
            "system" = as a system message, "context" = prepended to the user message,
            None = omitted

    Returns:
        Sample : The corresponding Sample object
    """
    has_system_prompt = record.get("system", "") != "" and system_prompt_behavior is not None

    if has_system_prompt and system_prompt_behavior == "context":
        # Inline the system prompt into the user message as context
        input = [ChatMessageUser(content=f"Context:\n{record['system']}\n\n{record['question']}")]
    elif has_system_prompt and system_prompt_behavior == "system":
        input = [ChatMessageSystem(content=record["system"]), ChatMessageUser(content=record["question"])]
    else:
        input = [ChatMessageUser(content=record["question"])]

    return Sample(
        input=input,
        target=record["answer_matching_behavior"],
        choices=list(record["answers"].values()),
        metadata={
            "labels": list(record["answers"].keys()),
            "behavior_category": record["category"],
            "system_prompt": has_system_prompt,
        },
    )