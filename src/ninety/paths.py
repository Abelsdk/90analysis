"""Where things live on disk."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Scraped data stays local and out of git (see .gitignore).
DATA_DIR = Path(os.environ.get("NINETY_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"

REFERENCE_DIR = REPO_ROOT / "reference"
FIXTURES_DIR = REPO_ROOT / "fixtures"
PREDICTIONS_DIR = REPO_ROOT / "predictions"
MODELS_DIR = REPO_ROOT / "models"
