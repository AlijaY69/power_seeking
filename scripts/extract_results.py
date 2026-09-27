"""
Reads the eval logs and saves the results as CSV files for the analysis notebook.

Usage (from the repo root):
    python -m scripts.extract_results

Writes:
    results/samples.csv    one row per answer (all sweep logs, deduplicated)
    results/summary.csv    power-seeking rate per model and configuration
    results/benchmark.csv  recognition-benchmark accuracy per model (if benchmark logs exist)
"""

import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

from src.analysis import load_samples, print_checks, summarise
from src.paths import BENCHMARK_LOG_DIR, RESULTS_DIR, SWEEP_LOG_DIR


def extract_benchmark() -> pd.DataFrame | None:
    """Accuracy per model from the benchmark logs (the most recent successful run per model)."""
    if not BENCHMARK_LOG_DIR.exists():
        return None

    rows = []
    for info in list_eval_logs(str(BENCHMARK_LOG_DIR)):
        log = read_eval_log(info, header_only=True)  # results only, no transcripts
        if log.status != "success":
            continue
        metrics = log.results.scores[0].metrics
        rows.append({
            "model": log.eval.model,
            "created": log.eval.created,
            "n": log.results.total_samples,
            "accuracy": metrics["accuracy"].value,
            "stderr": metrics["stderr"].value,
        })

    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("created")
    return df.groupby("model").tail(1).reset_index(drop=True)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    samples = load_samples(SWEEP_LOG_DIR)
    print_checks(samples)

    samples.to_csv(RESULTS_DIR / "samples.csv", index=False)
    summary = summarise(samples)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    print("\nSummary (parsed answers only):")
    print(summary.round(3).to_string(index=False))

    benchmark = extract_benchmark()
    if benchmark is not None:
        benchmark.to_csv(RESULTS_DIR / "benchmark.csv", index=False)
        print("\nBenchmark accuracy:")
        print(benchmark.round(3).to_string(index=False))

    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()