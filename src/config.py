"""
Central place for paths and settings.

Keeping every path and "magic number" in one file means the rest of the code
never has to guess where things live, and we only change a value once.
"""

import os
from pathlib import Path

# Optional: load a .env file if the user made one AND python-dotenv is
# installed. The project works perfectly without either -- see .env.example.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:                      # pragma: no cover - optional extra
    pass


def _path_from_env(variable: str, default: Path) -> Path:
    """Allow an environment variable to override a default path."""
    value = os.getenv(variable)
    return Path(value) if value else default

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
# config.py lives in src/, so the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
# The real dataset, built by `python -m src.download_data`.
DATASET_PATH = _path_from_env("DATA_PATH", DATA_DIR / "reviews.csv")
# A tiny hand-written file used only for demos/tests. NOT real review data.
SAMPLE_DATASET_PATH = DATA_DIR / "sample_reviews.csv"

MODEL_PATH = _path_from_env("MODEL_PATH", MODELS_DIR / "sentiment_model.joblib")
VECTORIZER_PATH = _path_from_env(
    "VECTORIZER_PATH", MODELS_DIR / "tfidf_vectorizer.joblib"
)
METRICS_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

# ---------------------------------------------------------------------------
# Dataset column names
# ---------------------------------------------------------------------------
TEXT_COLUMN = "review"
LABEL_COLUMN = "label"

# ---------------------------------------------------------------------------
# Modelling settings
# ---------------------------------------------------------------------------
# 0 = negative, 1 = positive. Kept as a dict so the label names live in one
# place (used by predict.py, the Streamlit app, and the plots).
LABEL_NAMES = {0: "Negative", 1: "Positive"}

TEST_SIZE = 0.2          # 80% train / 20% test
RANDOM_STATE = 42        # fixed so every run gives the same split -> reproducible
CV_FOLDS = 5             # folds for cross-validation on the training set
