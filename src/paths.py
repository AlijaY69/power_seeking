from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repo folder
DATA_PATH = ROOT / "data" / "power-seeking_300_qs.json"
LOG_DIR = ROOT / "logs"
SWEEP_LOG_DIR = LOG_DIR / "alignment_sweep"
BENCHMARK_LOG_DIR = LOG_DIR / "benchmark"
TEST_LOG_DIR = LOG_DIR / "test"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"