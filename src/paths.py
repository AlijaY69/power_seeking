from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repo folder
DATA_PATH = ROOT / "data" / "power-seeking_300_qs.json"
LOG_DIR = ROOT / "logs"
SWEEP_LOG_DIR = LOG_DIR / "alignment_sweep"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"