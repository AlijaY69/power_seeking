"""
Capability baseline: can the model recognise which option is the power-seeking one?

Usage (from the repo root):
    python -m scripts.run_benchmark --model openrouter/openai/gpt-4o-mini --n 50
"""

import argparse

from dotenv import load_dotenv
from inspect_ai import eval

from src.paths import BENCHMARK_LOG_DIR
from src.tasks import benchmark_eval

load_dotenv()  # reads OPENROUTER_API_KEY from .env


def main():
    parser = argparse.ArgumentParser(description="Run the power-seeking recognition benchmark.")
    parser.add_argument("--model", required=True, help="e.g. openrouter/openai/gpt-4o-mini")
    parser.add_argument("--n", type=int, default=None, help="number of questions (default: all 300)")
    args = parser.parse_args()

    logs = eval(
        benchmark_eval(n=args.n),
        model=args.model,
        log_dir=str(BENCHMARK_LOG_DIR),
    )

    log = logs[0]
    if log.status == "success":
        metrics = log.results.scores[0].metrics
        print(f"\nBenchmark accuracy for {args.model}: "
              f"{metrics['accuracy'].value:.3f} (± {metrics['stderr'].value:.3f})")
    else:
        print(f"\nRun failed: {log.error}")


if __name__ == "__main__":
    main()