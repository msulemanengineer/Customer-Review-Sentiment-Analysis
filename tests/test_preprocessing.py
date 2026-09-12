"""Tests for text cleaning and dataset loading.

Run all tests from the project root with:
    pytest -v
"""

import pandas as pd
import pytest

from src import config
from src.preprocessing import (
    SAFE_STOPWORDS,
    clean_series,
    clean_text,
    load_dataset,
    normalize_label,
    remove_stopwords,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

def test_clean_text_lowercases():
    assert clean_text("GREAT Phone") == "great phone"


def test_clean_text_removes_punctuation():
    assert clean_text("Great!!! Really, great...") == "great really great"


def test_clean_text_collapses_whitespace():
    assert clean_text("  too    many \n spaces \t here ") == "too many spaces here"


def test_clean_text_removes_urls():
    assert clean_text("Awful. See http://example.com/x?y=1 for proof") == (
        "awful see for proof"
    )
    assert clean_text("check www.spam.co now") == "check now"


def test_clean_text_removes_html_tags():
    assert clean_text("Loved it.<br /><br />Would buy again") == (
        "loved it would buy again"
    )


def test_clean_text_keeps_digits():
    # "2 stars" or "10/10" carry real information, so digits must survive.
    assert clean_text("Only 2 stars, 10/10 would not repeat") == (
        "only 2 stars 10 10 would not repeat"
    )


def test_clean_text_keeps_negation_words():
    # This is the whole reason we do not remove stopwords by default.
    assert "not" in clean_text("This is not good").split()


def test_clean_text_handles_non_strings():
    # A CSV column can hand us NaN (a float) or None. Must not crash.
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text(12345) == ""


def test_clean_text_on_empty_and_symbol_only_input():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text("!!!???") == ""


def test_clean_text_is_idempotent():
    # Cleaning already-clean text must change nothing. This matters because
    # train.py cleans the dataset and predict.py cleans again at inference.
    once = clean_text("Great PHONE!! Buy it.")
    assert clean_text(once) == once


def test_clean_series_applies_to_whole_column():
    series = pd.Series(["GOOD!", "Bad...", None])
    assert list(clean_series(series)) == ["good", "bad", ""]


# ---------------------------------------------------------------------------
# stopwords (the opt-in helper)
# ---------------------------------------------------------------------------

def test_remove_stopwords_drops_common_words():
    assert remove_stopwords("this is the best of them") == "best"


def test_stopword_list_protects_negations():
    # If any of these ever sneak into the list, sentiment breaks.
    for word in ("not", "no", "never", "but", "very", "too"):
        assert word not in SAFE_STOPWORDS


def test_remove_stopwords_keeps_negation():
    assert remove_stopwords("it is not good") == "not good"


# ---------------------------------------------------------------------------
# normalize_label
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        (1, 1), (0, 0),
        ("1", 1), ("0", 0),
        ("positive", 1), ("negative", 0),
        ("Positive", 1), ("NEGATIVE", 0),
        ("pos", 1), ("neg", 0),
        (" positive ", 1),
    ],
)
def test_normalize_label_accepts_known_forms(raw, expected):
    assert normalize_label(raw) == expected


@pytest.mark.parametrize("raw", ["neutral", "maybe", "", None, float("nan"), 7])
def test_normalize_label_rejects_unknown_forms(raw):
    # Returning None (rather than guessing) lets load_dataset drop the row
    # instead of silently training on a wrong label.
    assert normalize_label(raw) is None


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

def _write_csv(tmp_path, rows):
    path = tmp_path / "tiny.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_dataset_adds_clean_column(tmp_path):
    path = _write_csv(tmp_path, [
        {"review": "GREAT phone!", "label": 1},
        {"review": "Terrible.", "label": 0},
    ])
    df = load_dataset(path, verbose=False)

    assert list(df.columns) == ["review", "label", "clean_review"]
    assert df["clean_review"].tolist() == ["great phone", "terrible"]
    assert df["label"].tolist() == [1, 0]


def test_load_dataset_drops_duplicates(tmp_path):
    path = _write_csv(tmp_path, [
        {"review": "Same text", "label": 1},
        {"review": "Same text", "label": 1},
        {"review": "Different text", "label": 0},
    ])
    df = load_dataset(path, verbose=False)
    assert len(df) == 2


def test_load_dataset_drops_missing_and_unlabelled_rows(tmp_path):
    path = _write_csv(tmp_path, [
        {"review": "Good one", "label": 1},
        {"review": None, "label": 1},          # missing text
        {"review": "   ", "label": 0},         # blank text
        {"review": "Fine", "label": "neutral"},  # unusable label
        {"review": "!!!", "label": 0},         # nothing left after cleaning
    ])
    df = load_dataset(path, verbose=False)
    assert df["review"].tolist() == ["Good one"]


def test_load_dataset_normalizes_string_labels(tmp_path):
    path = _write_csv(tmp_path, [
        {"review": "Loved it", "label": "positive"},
        {"review": "Hated it", "label": "negative"},
    ])
    df = load_dataset(path, verbose=False)
    assert df["label"].tolist() == [1, 0]
    assert df["label"].dtype.kind == "i"        # integers, not strings


def test_load_dataset_missing_file_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_dataset(tmp_path / "does_not_exist.csv", verbose=False)


def test_load_dataset_wrong_columns_raises(tmp_path):
    path = _write_csv(tmp_path, [{"text": "hello", "sentiment": 1}])
    with pytest.raises(ValueError, match="missing required column"):
        load_dataset(path, verbose=False)


def test_sample_dataset_in_repo_loads(tmp_path):
    """The committed demo file must stay loadable -- the README points at it."""
    df = load_dataset(config.SAMPLE_DATASET_PATH, verbose=False)
    assert len(df) > 0
    assert set(df[config.LABEL_COLUMN].unique()) == {0, 1}
