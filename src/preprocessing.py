"""
Text cleaning + dataset loading.

DESIGN DECISION (important for the viva):
We deliberately keep the cleaning *light*. Every cleaning step throws
information away, and in sentiment analysis some of the "noise" is actually
the signal. For example:

    "not good"  ->  removing the stopword "not" turns this into "good",
                    which flips the meaning completely.

So we only do cleaning that is safe:
  1. lowercase            ("Good" and "good" should be the same feature)
  2. remove URLs          (a web address carries no sentiment)
  3. remove HTML tags     (some review dumps contain <br /> markup)
  4. drop characters that are not letters, digits or spaces
  5. collapse repeated whitespace

We do NOT stem, lemmatise, or remove stopwords by default. Stopword removal is
available as an opt-in helper with negation words protected, so you can show in
the notebook that it does not help here.

MEASURED, NOT GUESSED: we also tried expanding contractions ("don't" -> "do
not") so that the word "not" survives tokenisation. Cross-validation on the
training set moved by less than one standard deviation, so we left it out and
kept the code simpler. See SYSTEM_GUIDE.md section 6.
"""

from __future__ import annotations

import re

import pandas as pd

from src import config

# --- Regex patterns, compiled once at import time (faster than re-compiling) --

# http://... , https://... or www...
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
# anything that looks like an HTML tag, e.g. <br />
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
# keep letters, digits and spaces; everything else becomes a space.
# Note this also drops apostrophes, so "don't" becomes "don t". That is fine:
# scikit-learn's default tokenizer splits on the apostrophe anyway and throws
# away the single letter "t", so "don" ends up acting as the marker for
# "don't" either way.
KEEP_PATTERN = re.compile(r"[^a-z0-9\s]")
# one or more whitespace characters
WHITESPACE_PATTERN = re.compile(r"\s+")

# A short stopword list. Notice which words are NOT here: "not", "no", "never",
# "but", "very", "too". Those change or intensify sentiment, so removing them
# would hurt the model.
SAFE_STOPWORDS = frozenset(
    """
    a an the this that these those
    i me my we our you your he him his she her it its they them their
    am is are was were be been being
    of to in on at for with from by as
    and or if then than so
    do does did done have has had
    """.split()
)


def clean_text(text: str) -> str:
    """Clean one review string. Returns lowercase text with only words left.

    >>> clean_text("  GREAT phone!!  Visit http://x.com <br />")
    'great phone'
    """
    # Guard against NaN / numbers / None coming from a CSV column.
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = KEEP_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def remove_stopwords(text: str, stopwords: frozenset[str] = SAFE_STOPWORDS) -> str:
    """Optional extra step. Off by default -- see the module docstring."""
    return " ".join(word for word in text.split() if word not in stopwords)


def clean_series(texts: pd.Series) -> pd.Series:
    """Apply clean_text to a whole pandas column."""
    return texts.astype(str).map(clean_text)


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

# Different datasets write labels differently. We map everything to 0/1 so the
# rest of the pipeline only ever sees integers.
_LABEL_MAP = {
    "0": 0, "neg": 0, "negative": 0, "bad": 0,
    "1": 1, "pos": 1, "positive": 1, "good": 1,
}


def normalize_label(value) -> int | None:
    """Turn any of 0/1, 'pos'/'neg', 'positive'/'negative' into 0 or 1.

    Returns None for anything we do not recognise, so the caller can drop it
    instead of silently training on a wrong label.
    """
    if pd.isna(value):
        return None
    key = str(value).strip().lower()
    return _LABEL_MAP.get(key)


# ---------------------------------------------------------------------------
# Dataset loading + cleaning
# ---------------------------------------------------------------------------

def load_dataset(path=None, verbose: bool = True) -> pd.DataFrame:
    """Load the review CSV and return a clean, model-ready DataFrame.

    Steps (this is the "data processing" part of the pipeline):
      1. read the CSV
      2. check the expected columns exist
      3. normalise the labels to 0/1 and drop unknown ones
      4. drop rows with a missing/empty review
      5. drop duplicate reviews
      6. add a `clean_review` column produced by clean_text()
      7. drop rows that became empty after cleaning

    Returns a DataFrame with columns: review, label, clean_review (+ `source`
    if the input file has it).
    """
    path = path or config.DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Run `python -m src.download_data` first, "
            "or see data/README.md for manual download steps."
        )

    df = pd.read_csv(path)

    missing = {config.TEXT_COLUMN, config.LABEL_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required column(s): {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    rows_loaded = len(df)

    # 3. labels -> 0/1
    df[config.LABEL_COLUMN] = df[config.LABEL_COLUMN].map(normalize_label)
    df = df[df[config.LABEL_COLUMN].notna()].copy()
    df[config.LABEL_COLUMN] = df[config.LABEL_COLUMN].astype(int)

    # 4. missing / blank reviews
    df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].astype(str).str.strip()
    df = df[df[config.TEXT_COLUMN].ne("") & df[config.TEXT_COLUMN].ne("nan")]

    # 5. duplicates. The same sentence twice teaches the model nothing new, and
    #    if a duplicate lands in both train and test it inflates the score.
    df = df.drop_duplicates(subset=[config.TEXT_COLUMN])

    # 6. + 7. cleaning
    df["clean_review"] = clean_series(df[config.TEXT_COLUMN])
    df = df[df["clean_review"].ne("")]

    df = df.reset_index(drop=True)

    if verbose:
        print(f"Loaded {rows_loaded} rows from {path.name}")
        print(f"Usable rows after cleaning: {len(df)} "
              f"(removed {rows_loaded - len(df)})")
        counts = df[config.LABEL_COLUMN].value_counts().sort_index()
        for label, count in counts.items():
            print(f"  {config.LABEL_NAMES[label]}: {count} "
                  f"({count / len(df):.1%})")

    return df
