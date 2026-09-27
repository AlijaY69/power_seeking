"""
Alignment sweep: 3 system-prompt conditions x 3 reasoning setups for one model.

Usage (from the repo root):
    python -m scripts.run_sweep --model openrouter/openai/gpt-4o-mini --n 5   # quick test
    python -m scripts.run_sweep --model openrouter/openai/gpt-4o-mini         # full run

Test runs (--n given) are written to logs/test/ so they never mix with the real results.
"""

import argparse
from itertools import product

from dotenv import load_dotenv
from inspect_ai import eval

from src.paths import SWEEP_LOG_DIR, TEST_LOG_DIR
from src.tasks import alignment_eval

load_dotenv()  # reads OPENROUTER_API_KEY from .env

SYSTEM_PROMPT_BEHAVIORS = ["system", "context", None]
REASONING_CONFIGS = [
    dict(use_cot=False, use_self_critique=False),  # answer directly
    dict(use_cot=True, use_self_critique=False),  # think, then answer
    dict(use_cot=True, use_self_critique=True),  # think, critique, revise, answer
]
# (no-CoT + self-critique is skipped: critiquing a bare letter isn't meaningful)


def main():
    parser = argparse.ArgumentParser(description="Run the full alignment sweep for one model.")
    parser.add_argument("--model", required=True, help="e.g. openrouter/openai/gpt-4o-mini")
    parser.add_argument("--n", type=int, default=None, help="number of questions (default: all 300)")
    parser.add_argument("--max-tasks", type=int, default=3, help="configurations run concurrently")
    args = parser.parse_args()

    log_dir = SWEEP_LOG_DIR if args.n is None else TEST_LOG_DIR

    tasks = [
        alignment_eval(
            n=args.n,
            system_prompt_behavior=spb,
            self_critique_model=args.model,  # "self"-critique: the same model critiques itself
            **cfg,
        )
        for spb, cfg in product(SYSTEM_PROMPT_BEHAVIORS, REASONING_CONFIGS)
    ]

    logs = eval(tasks, model=args.model, log_dir=str(log_dir), max_tasks=args.max_tasks)

    # Quick summary (power-seeking rate = fraction choosing the power-seeking option)
    print()
    for log in logs:
        a = log.eval.task_args
        label = (f"system_prompt={str(a.get('system_prompt_behavior')):8} "
                 f"cot={str(a.get('use_cot')):5} critique={str(a.get('use_self_critique')):5}")
        if log.status != "success":
            print(f"{label} -> FAILED: {log.error}")
            continue
        m = log.results.scores[0].metrics
        print(f"{label} -> power-seeking rate {m['accuracy'].value:.3f} (± {m['stderr'].value:.3f})")


if __name__ == "__main__":
    main()