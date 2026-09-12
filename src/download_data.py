"""
Download the real dataset and save it as data/reviews.csv.

Dataset: "Sentiment Labelled Sentences" (UCI Machine Learning Repository)
  3,000 review sentences taken from amazon.com, yelp.com and imdb.com,
  each labelled 1 = positive or 0 = negative.
  Created for: Kotzias et al., "From Group to Individual Labels using Deep
  Features", KDD 2015.

Run with:
    python -m src.download_data

If your machine has no internet access, follow data/README.md to download the
zip by hand -- this script can also read an already-downloaded zip:
    python -m src.download_data --zip path/to/sentiment+labelled+sentences.zip
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile

import pandas as pd

from src import config

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/331/"
    "sentiment+labelled+sentences.zip"
)

# file name inside the zip -> short name we store in the `source` column
SOURCE_FILES = {
    "amazon_cells_labelled.txt": "amazon",
    "yelp_labelled.txt": "yelp",
    "imdb_labelled.txt": "imdb",
}


def _read_zip(raw_bytes: bytes) -> pd.DataFrame:
    """Turn the downloaded zip into one tidy DataFrame."""
    frames = []
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
        for member in archive.namelist():
            # skip macOS junk folders that are inside this particular zip
            if member.startswith("__MACOSX") or not member.endswith(".txt"):
                continue
            filename = member.rsplit("/", 1)[-1]
            if filename not in SOURCE_FILES:
                continue

            with archive.open(member) as handle:
                # The files are "sentence <TAB> label" with no header row.
                frame = pd.read_csv(
                    handle,
                    sep="\t",
                    header=None,
                    names=[config.TEXT_COLUMN, config.LABEL_COLUMN],
                    quoting=3,          # csv.QUOTE_NONE: quotes are part of the text
                    encoding="utf-8",
                    on_bad_lines="skip",
                )
            frame["source"] = SOURCE_FILES[filename]
            frames.append(frame)
            print(f"  {filename}: {len(frame)} rows")

    if not frames:
        raise RuntimeError("No expected .txt files found inside the archive.")
    return pd.concat(frames, ignore_index=True)


def main(zip_path: str | None = None) -> int:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if zip_path:
        print(f"Reading local archive: {zip_path}")
        raw = open(zip_path, "rb").read()
    else:
        print(f"Downloading {DATA_URL}")
        try:
            with urllib.request.urlopen(DATA_URL, timeout=60) as response:
                raw = response.read()
        except Exception as error:                      # noqa: BLE001
            print(f"\nDownload failed: {error}")
            print("See data/README.md for manual download instructions.")
            return 1
        print(f"Downloaded {len(raw) / 1024:.0f} KB")

    df = _read_zip(raw)

    # Keep only rows with a usable label, then write a plain CSV.
    df = df[df[config.LABEL_COLUMN].isin([0, 1])]
    df[config.LABEL_COLUMN] = df[config.LABEL_COLUMN].astype(int)
    df.to_csv(config.DATASET_PATH, index=False)

    print(f"\nSaved {len(df)} reviews -> {config.DATASET_PATH}")
    print(df[config.LABEL_COLUMN].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", default=None,
                        help="Path to an already-downloaded dataset zip.")
    args = parser.parse_args()
    sys.exit(main(args.zip_path))
