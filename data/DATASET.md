# Dataset: power-seeking_300_qs.json

300 multiple-choice questions (MCQs) to determine whether a model exhibits power-seeking tendencies by selecting options that increase its own power when less power-seeking options are equally viable.

## Definition

**Power-seeking**: [def]

Operationalized into three categories:

| Category | Meaning | Questions |
|---|---|---|
| resource_accumulation | taking control over resources, such as budgets, slots, tools, or capacity beyond what the task requires | 210 |
| circumventing_oversight | making decisions that circumvent others, thereby limiting their ability to review or intervene in the model's decisions | 65 |
| upward_mobility | seeking roles that give more power through authority or responsibility | 25 |

Each question has only one power-seeking option. The other three are reasonable alternatives that accomplish the task without increasing the model's power in any of the three operationalized ways.

## Generation process

The pipeline follows the dataset-generation exercises provided in ARENA 3.0, Chapter 3 (LLM evals).
Generation model: [z-ai/glm-5.3-flash]. Grading model: [z-ai/glm-5.3-flash]

1. Define the property precisely and unambiguously
2. Extend the definiton with operational definitions that make it measurable (the three categories above).
3. Write 8 seed questions, each with 4 answer options, that capture the property. 
4. Use the seed questions as few-shot examples to generate a pilot set of 20 questions, with variace prompts so that diferent facets of the property are covered.
5. Write a grading rubric with criteria for question quality: [criteria] [threshold].
6. Revise the generation prompts until the pilot questions reliably pass the rubric, taking care not to overfit the prompts to the rubric.
7. Regenerate the pilot set with the final prompts, then generate the full set of 300 questions.

No two questions are identical.

## Record format 

Each record has these fields:

- "system":  
- "question":
- "answers":
- "answer_matching_behavior":
- "answer_not_matching_behavior":
- "category":

Example ()

