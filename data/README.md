# Data folder

This folder holds two very different things. Please do not mix them up.

| File | What it is | In git? |
| --- | --- | --- |
| `reviews.csv` | **The real dataset.** 3,000 human-labelled review sentences. Downloaded by a script — not stored in this repo. | No |
| `sample_reviews.csv` | **Hand-written demo data.** 20 sentences I wrote myself to test the batch-upload feature. Not real customer reviews. | Yes |

---

## 1. The real dataset: `reviews.csv`

### What it is

**Sentiment Labelled Sentences**, from the UCI Machine Learning Repository.

- **3,000 sentences** taken from real reviews on three websites:
  - `amazon.com` — product reviews (mobile phone accessories)
  - `yelp.com` — restaurant reviews
  - `imdb.com` — movie reviews
- **1,000 sentences from each site**, exactly 500 positive and 500 negative — so the dataset is perfectly balanced.
- Each row is one sentence plus a label: `1` = positive, `0` = negative.
- The authors deliberately picked sentences with a **clearly** positive or negative tone, so there is **no neutral class**.

### Where it comes from

- **Page:** <https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences>
- **Direct zip:** <https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip> (~82 KB)
- **Paper to cite:** Kotzias, D., Denil, M., de Freitas, N., Smyth, P. — *"From Group to Individual Labels using Deep Features"*, KDD 2015.
- **Licence:** the UCI repository distributes it for research and educational use. The dataset authors ask that you cite the paper above, which is why it is cited here and in the main README.

This dataset was chosen because it is small (82 KB), genuinely public, genuinely human-labelled, and made of real customer review text — so nothing in this project depends on invented data.

### How to get it (automatic — do this)

From the **project root**:

```bash
python -m src.download_data
```

This downloads the zip, reads the three `*_labelled.txt` files out of it, and writes `data/reviews.csv` with these columns:

| Column | Meaning |
| --- | --- |
| `review` | the review sentence (raw text, uncleaned) |
| `label` | `1` = positive, `0` = negative |
| `source` | `amazon`, `yelp` or `imdb` |

Expected output:

```
Downloading https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip
Downloaded 82 KB
  amazon_cells_labelled.txt: 1000 rows
  imdb_labelled.txt: 1000 rows
  yelp_labelled.txt: 1000 rows

Saved 3000 reviews -> data/reviews.csv
label
0    1500
1    1500
```

### How to get it (manual — if you have no internet on this machine)

1. Download the zip from either link above on any machine.
2. Copy it onto this machine — do **not** unzip it.
3. Point the script at it:

```bash
python -m src.download_data --zip "C:/Downloads/sentiment+labelled+sentences.zip"
```

If you would rather do it entirely by hand: unzip it, and you will find three tab-separated files with no header row, in the format `sentence <TAB> label`. Concatenate them into a CSV with the columns `review,label` and save it as `data/reviews.csv`. The training script only needs those two columns; `source` is optional.

### Why `reviews.csv` is not committed to git

Three reasons:

1. It is someone else's dataset — better to point at the original source than to redistribute a copy.
2. Anyone can rebuild it byte-for-byte with one command, so committing it adds nothing.
3. It keeps the repository small.

It is listed in `.gitignore` for exactly this reason.

### What the cleaning step does to it

`src/preprocessing.load_dataset()` drops unusable rows before training. On this dataset it removes **18 of the 3,000 rows** (duplicated sentences, and rows with nothing left after cleaning), leaving **2,982 usable reviews**: 1,492 negative and 1,490 positive. Those are the numbers reported throughout the README.

---

## 2. The demo data: `sample_reviews.csv`

**This is not real data.** I wrote these 20 sentences by hand so that:

- the batch-CSV upload in the Streamlit app has something to try, and
- the test suite has a small file it can load without downloading anything.

They are short, obvious, clearly-worded sentences — much easier than real reviews. **Any accuracy you see on this file is meaningless as a measure of model quality.** The honest performance number is the one measured on the held-out test set of real reviews, reported in the main `README.md` and in `reports/metrics.json`.

Format:

```csv
review,label
"The battery lasted two days and then stopped charging completely.",0
"Absolutely love this product, it works exactly as described.",1
```

---

## Using your own data instead

Any CSV with a `review` column and a `label` column will work:

```bash
python -m src.train --data path/to/my_reviews.csv
```

The loader is forgiving about how labels are written — `1`/`0`, `positive`/`negative`, `pos`/`neg` are all accepted and normalised to `1`/`0`. Anything it does not recognise (including `neutral`) is dropped, with the count reported, rather than being silently guessed at.
