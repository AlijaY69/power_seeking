def record_to_sample_full(record: dict, system_prompt_behavior: Literal["system", "context"] | None = None) -> Sample:
    """
    Converts a item ("record") from the dataset into a Sample object, mapping the fields of the
    record to the fields of the Sample object.

    Difference from previous function: we randomly shuffle the order of the 2 presented choices.

    Args:
        record : A dictionary from the json dataset containing our evaluation questions

    Returns:
        Sample : A Sample object containing the information in the record
    """
    has_system_prompt = record.get("system", "") != "" and system_prompt_behavior is not None
    if has_system_prompt and system_prompt_behavior == "context":
        # Inline the system prompt into the user message as context (no separate user message)
        input = [ChatMessageUser(content=f"Context:\n{record['system']}\n\n{record['question']}")]
    elif has_system_prompt and system_prompt_behavior == "system":
        input = [ChatMessageSystem(content=record["system"]), ChatMessageUser(content=record["question"])]
    else:
        input = [ChatMessageUser(content=record["question"])]

    

    return Sample(
        input=input,
        target= record["answer_matching_behavior"],
        choices= list(record["answers"].values()),
        metadata={
            "labels": list(record["answers"].keys()),
            "behavior_category": record["category"],
            "system_prompt": has_system_prompt,
        },
    )