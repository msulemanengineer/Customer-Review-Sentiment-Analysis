# SYSTEM GUIDE — Study and Viva Preparation

This is the study companion for the Customer Review Sentiment Analysis project. It explains **every concept used in the project in simple English**, in the order you would meet it.

Read it once slowly. Then read the headings only and try to explain each one out loud. If you can do that, you can handle the viva.

For questions and short answers, see **[VIVA_QUESTIONS.md](VIVA_QUESTIONS.md)**. For the code and the results, see **[README.md](README.md)**.

---

## Contents

| # | Topic |
| --- | --- |
| 1 | [What is sentiment analysis?](#1-what-is-sentiment-analysis) |
| 2 | [What is NLP?](#2-what-is-nlp) |
| 3 | [What problem does this project solve?](#3-what-problem-does-this-project-solve) |
| 4 | [The complete ML pipeline](#4-the-complete-ml-pipeline) |
| 5 | [The dataset](#5-the-dataset) |
| 6 | [Data cleaning](#6-data-cleaning) |
| 7 | [Train/test split](#7-traintest-split) |
| 8 | [What is data leakage?](#8-what-is-data-leakage) |
| 9 | [What is TF-IDF?](#9-what-is-tf-idf) |
| 10 | [The TF-IDF formula in simple terms](#10-the-tf-idf-formula-in-simple-terms) |
| 11 | [What is term frequency?](#11-what-is-term-frequency) |
| 12 | [What is inverse document frequency?](#12-what-is-inverse-document-frequency) |
| 13 | [Why TF-IDF instead of simple word counts?](#13-why-tf-idf-instead-of-simple-word-counts) |
| 14 | [What are n-grams?](#14-what-are-n-grams) |
| 15 | [What is Logistic Regression?](#15-what-is-logistic-regression) |
| 16 | [Why Logistic Regression for text classification?](#16-why-logistic-regression-for-text-classification) |
| 17 | [What is Naive Bayes?](#17-what-is-naive-bayes) |
| 18 | [Why compare different models?](#18-why-compare-different-models) |
| 19 | [Accuracy](#19-accuracy) |
| 20 | [Precision](#20-precision) |
| 21 | [Recall](#21-recall) |
| 22 | [F1-score](#22-f1-score) |
| 23 | [Confusion matrix](#23-confusion-matrix) |
| 24 | [Overfitting](#24-overfitting) |
| 25 | [Underfitting](#25-underfitting) |
| 26 | [Training vs testing data](#26-training-vs-testing-data) |
| 27 | [Feature engineering](#27-feature-engineering) |
| 28 | [Model serialization](#28-model-serialization) |
| 29 | [How Streamlit works](#29-how-streamlit-works) |
| 30 | [Limitations](#30-limitations) |
| 31 | [Future improvements](#31-future-improvements) |
| + | [Extra concepts worth knowing](#extra-concepts-worth-knowing) |
| + | [Cheat sheet: the numbers to remember](#cheat-sheet-the-numbers-to-remember) |

---

## 1. What is sentiment analysis?

**Sentiment analysis is teaching a computer to decide whether a piece of writing is positive or negative.**

A human reads *"the battery died after two days"* and instantly knows the customer is unhappy. Sentiment analysis is getting software to reach the same conclusion.

It is also called **opinion mining**, because we are mining opinions out of text.

There are different versions of the task:

| Version | What it predicts | Ours? |
| --- | --- | --- |
| Binary | Positive or Negative | ✅ **this project** |
| Three-class | Positive, Negative, or Neutral | ❌ |
| Fine-grained | A star rating, 1 to 5 | ❌ |
| Aspect-based | Sentiment per topic: *"screen: good, battery: bad"* | ❌ |
| Emotion detection | Angry, sad, happy, surprised… | ❌ |

We do **binary** sentiment analysis because our dataset has exactly two labels and because it is the cleanest way to demonstrate the full workflow.

**Where it is used in real life:** a company tracking whether reviews are getting worse after a product change; support teams routing angry messages to a human first; brands monitoring social media; airlines and hotels summarising thousands of guest comments.

---

## 2. What is NLP?

**NLP (Natural Language Processing) is the field of computer science that deals with human language.**

"Natural language" just means the language people actually speak and write — English, Urdu, Arabic — as opposed to an artificial language like Python or SQL.

The core difficulty: **computers do arithmetic, and language is not arithmetic.** A computer can compare `5 > 3` instantly, but it has no idea whether `"decent"` is closer to `"good"` or to `"bad"`. NLP is the collection of techniques for bridging that gap.

Language is hard for specific reasons:

- **Ambiguity** — *"the food was sick"* might be praise or a complaint.
- **Context** — *"it's fine"* can mean genuinely fine, or deeply annoyed.
- **Order matters** — "not really good" ≠ "really not good".
- **Infinite variety** — people invent words, misspell, use slang and emoji.

Common NLP tasks besides sentiment: translation, spam detection, summarisation, question answering, named-entity recognition (pulling out names and places), speech recognition, and chatbots.

**In this project** the NLP-specific parts are: cleaning text (§6), tokenisation (splitting text into words), n-grams (§14), and turning words into numbers with TF-IDF (§9). Everything after that is ordinary machine learning on a table of numbers.

---

## 3. What problem does this project solve?

**The problem:** a business receives far more written reviews than any person can read, and the important ones — the complaints — are buried in the pile.

**What this project does:** given the text of a review, it predicts **Positive** or **Negative** and reports how confident it is.

**How that helps:**

1. **Scale.** Thousands of reviews get tagged in seconds.
2. **Prioritisation.** Negative reviews can be routed to support first.
3. **Trend tracking.** "We were 80% positive last month and 60% this month" is an early warning something broke.
4. **Insight.** Because the model keeps one weight per word, you can read off which words unhappy customers actually use.

**What it does not solve:** it cannot tell you *why* someone is unhappy (that is aspect-based sentiment analysis), it has no neutral option, and it cannot detect sarcasm. Being clear about these boundaries is part of the answer, not a weakness in it.

---

## 4. The complete ML pipeline

This is the spine of the whole project. Learn this sequence — most viva questions are really "explain one box in this diagram".

```
  ┌─────────────────────────────────────────────────────────────┐
  │  1. GET DATA          data/reviews.csv (3,000 reviews)      │
  │     src/download_data.py                                    │
  └──────────────────────────┬──────────────────────────────────┘
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  2. EXPLORE           how many rows? balanced? how long?    │
  └──────────────────────────┬──────────────────────────────────┘
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  3. CLEAN             fix labels, drop blanks + duplicates  │
  │     src/preprocessing.py        3,000 -> 2,982 usable rows  │
  └──────────────────────────┬──────────────────────────────────┘
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  4. PREPROCESS TEXT   lowercase, strip URLs/HTML/symbols    │
  │     clean_text()                                            │
  └──────────────────────────┬──────────────────────────────────┘
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  5. SPLIT             80% train (2,385) / 20% test (597)    │
  │     stratified, random_state=42     <-- BEFORE the next box │
  └──────────────────────────┬──────────────────────────────────┘
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  6. FEATURES          TF-IDF: text -> numbers               │
  │     src/features.py   fit on TRAIN only, transform test     │
  │                       2,385 x 20,000 sparse matrix          │
  └──────────────────────────┬──────────────────────────────────┘
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  7. TRAIN             Logistic Regression  (+ Naive Bayes   │
  │     src/train.py      as a baseline), 5-fold CV on train    │
  └──────────────────────────┬──────────────────────────────────┘
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  8. EVALUATE          on the 597 unseen test reviews        │
  │     src/evaluate.py   accuracy, precision, recall, F1, CM   │
  │                       -> 82.08% accuracy                    │
  └──────────────────────────┬──────────────────────────────────┘
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  9. SAVE              joblib -> models/*.joblib             │
  │                       the model AND the vectorizer          │
  └──────────────────────────┬──────────────────────────────────┘
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 10. PREDICT / SERVE   src/predict.py  +  app.py (Streamlit) │
  │     loads the saved files — never retrains                  │
  └─────────────────────────────────────────────────────────────┘
```

**The one-sentence version:** get data → clean it → split it → turn text into numbers → train → measure on unseen data → save → serve.

**The two rules that matter most:**
1. **Split before you fit anything.** (§8)
2. **Measure on data the model has never seen.** (§26)

---

## 5. The dataset

We use **Sentiment Labelled Sentences** from the UCI Machine Learning Repository.

| | |
| --- | --- |
| What | 3,000 sentences taken from real online reviews |
| Where from | amazon.com (products), yelp.com (restaurants), imdb.com (movies) — 1,000 each |
| Labels | `1` = positive, `0` = negative |
| Balance | 500 positive + 500 negative per site, so **perfectly balanced** |
| Neutral? | **No.** The authors deliberately picked clearly-positive or clearly-negative sentences |
| Size | ~82 KB |
| Citation | Kotzias et al., *"From Group to Individual Labels using Deep Features"*, KDD 2015 |

**Why this dataset?** It is small, genuinely public, genuinely human-labelled, and made of real customer review text. Nothing in this project rests on invented data.

**After cleaning:** 2,982 usable rows (1,492 negative, 1,490 positive). Average length 12.2 words — these are single sentences, not full reviews, which matters when explaining the model's limits.

**Be ready for this question: "why is the balance important?"**
Because it decides whether accuracy is a fair headline number. On a balanced dataset, 50% is the "coin flip" baseline, so 82% is clearly real learning. On a 95%-positive dataset, 95% accuracy could mean the model learned nothing at all (§19).

> ⚠️ **Know the difference between the two data files.** `data/reviews.csv` is the real dataset. `data/sample_reviews.csv` is 20 sentences written by hand to test the CSV-upload button — demo data, never used to measure performance. If you are asked about data honesty, say this.

---

## 6. Data cleaning

Cleaning happens in two layers: **row-level** (which rows do we keep?) and **text-level** (what do we do to the words?).

### Row-level cleaning — `load_dataset()`

| Step | What it does | Why |
| --- | --- | --- |
| Check columns | error if `review` / `label` missing | fail loudly and early, not halfway through training |
| Normalise labels | `1`, `"positive"`, `"pos"` → `1` | different datasets write labels differently |
| Drop unknown labels | `"neutral"` → dropped | **never guess a label.** A wrong label actively teaches the model something false |
| Drop missing text | `NaN`, `""`, `"   "` → dropped | nothing to learn from an empty review |
| **Drop duplicates** | identical review text kept once | see the box below |
| Drop post-clean blanks | `"!!!"` cleans to `""` → dropped | no words left to classify |

On our data this removes **18 of 3,000 rows**.

> **Why dropping duplicates really matters.** It is not about tidiness. If the same sentence appears twice and the split puts one copy in train and the other in test, the model has *already seen the test answer*. Your test score goes up without the model getting any better. That is a mild form of data leakage (§8).

### Text-level cleaning — `clean_text()`

We deliberately keep this **light**. Every cleaning step throws information away, and in sentiment analysis the "noise" is sometimes the signal.

**What we do:**

| Step | Example | Why |
| --- | --- | --- |
| Lowercase | `"GREAT"` → `"great"` | otherwise `"Good"` and `"good"` are two separate features, splitting the evidence |
| Remove URLs | `"see http://x.com"` → `"see"` | a web address carries no sentiment |
| Remove HTML tags | `"good<br />"` → `"good"` | scraped reviews contain markup |
| Remove symbols | `"great!!!"` → `"great"` | the tokenizer discards them anyway |
| Collapse whitespace | `"too    many"` → `"too many"` | `"good"` and `"good "` must be one feature |

**What we deliberately DON'T do — this is the part interviewers probe:**

- **No stopword removal.** Stopwords are very common words (`the`, `is`, `of`). Removing them is standard advice in NLP — and **wrong for sentiment analysis**, because the most dangerous "stopword" is `"not"`. Remove it from *"not good"* and you get *"good"*: the exact opposite meaning. We provide an opt-in `remove_stopwords()` helper whose word list deliberately excludes `not`, `no`, `never`, `but`, `very` and `too`.
- **No stemming or lemmatisation.** These chop words to a root form (`"loved"` → `"love"`). It shrinks the vocabulary, but costs nuance and needs an extra library. On 12-word sentences it was not worth it.
- **No contraction expansion.** We *tried* converting `"don't"` → `"do not"` so the word `"not"` would survive. Cross-validation moved by less than one standard deviation, so we dropped it to keep the code simple. **This is the answer to give when asked how you made preprocessing decisions: we measured them.**
- **Digits are kept.** `"2 stars"` and `"10/10"` carry real information.

---

## 7. Train/test split

We split the 2,982 reviews into:

- **Training set — 2,385 reviews (80%).** The model learns from these.
- **Test set — 597 reviews (20%).** Locked away until the very end, used once to measure.

```python
train_test_split(
    df["clean_review"], df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)
```

### Why split at all?

Because **a model can memorise.** If you train on all the data and then test on that same data, you are asking "can you remember what I just told you?" — not "have you learned anything useful?" Real users will send reviews the model has never seen, so the only honest test is on reviews the model has never seen.

*Classroom analogy:* revising past papers is training. If the exam is literally the same past paper, a high score proves nothing. The test set is the *unseen* exam.

### Why 80/20?

A trade-off. More training data → a better model. More test data → a more trustworthy score. 80/20 is the usual compromise; with 2,982 rows it gives 597 test reviews, enough for a meaningful number. (A rough rule: with 597 samples the accuracy estimate is good to roughly ±3 percentage points.)

### Why `stratify=y`?

**Stratification keeps the same class balance in both halves.** Our data is ~50/50, so stratifying gives 299 negative and 298 positive in the test set.

Without it, random chance could hand you a test set that is 60% positive, and your score would be measuring luck as much as skill. Stratification matters even more on imbalanced data: on a 95/5 split, a random 20% test set might contain almost no minority-class examples at all — you would have no way to measure the thing you most care about.

### Why `random_state=42`?

The split is random. `random_state` fixes the random number generator so **the same split happens every time**. That makes results reproducible: anyone who runs `python -m src.train` gets 82.08%, and `python -m src.evaluate` can rebuild the identical test set later. (42 is just a convention — a joke from *The Hitchhiker's Guide to the Galaxy*. Any fixed number works.)

---

## 8. What is data leakage?

**Data leakage is when information from your test set sneaks into training. The result is a score that looks great and is a lie.**

It is the most important mistake to be able to explain, because it is invisible — nothing crashes, your numbers just quietly become wrong (too good).

### The leak that this project specifically avoids

Look at the *wrong* way to do it:

```python
# ❌ WRONG — leaks
X = vectorizer.fit_transform(all_reviews)      # fitted on EVERYTHING
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

Why is that a leak? Because `fit` does two things: it **builds the vocabulary**, and it **computes the IDF weights** (§12). If it sees all the data, then:

- words that only ever appear in the test set still get a vocabulary column, and
- the IDF value for every word is calculated using document counts from the test set.

So the training features were shaped by test data. The model gained knowledge about data it is supposed to have never seen, and the test score comes out optimistically high.

```python
# ✅ RIGHT — what this project does
X_train_text, X_test_text, y_train, y_test = train_test_split(...)   # split FIRST
X_train = vectorizer.fit_transform(X_train_text)   # fit: learn vocab + IDF from train
X_test  = vectorizer.transform(X_test_text)        # transform only: reuse what it learned
```

**`fit_transform` on train. `transform` on test. Never `fit` on test.** If you remember one line from this guide, remember that one.

A side effect of doing it correctly: test-set words the vectorizer never saw are silently ignored. That feels like losing information, but it is exactly right — it is what will happen in production when a real user types a brand-new word.

### Other kinds of leakage to be able to name

| Kind | Example |
| --- | --- |
| Preprocessing leakage | fitting a scaler, vectorizer, or imputer on all the data before splitting |
| Duplicate leakage | the same review in both halves — which is why we drop duplicates (§6) |
| Target leakage | a feature that secretly contains the answer, e.g. keeping a `star_rating` column while predicting sentiment |
| Temporal leakage | training on future data to predict the past (for time-series data, split by date, not randomly) |
| Test-set tuning | choosing hyperparameters by whichever scores best on the *test* set — the test set stops being unseen. **This is why we tuned `C` and `min_df` with cross-validation on the training set only.** |

That last row is worth emphasising in a viva: we ran a 7-value grid for `C` and a 3-value grid for `min_df`, and every one of those scores came from cross-validation *inside the training set*. The test set was used exactly once, at the end.

---

## 9. What is TF-IDF?

**TF-IDF turns a piece of text into a list of numbers, where each number says how important one word is to that text.**

TF-IDF stands for **Term Frequency – Inverse Document Frequency**. It is two ideas multiplied together:

- **TF (Term Frequency)** — how often the word appears **in this review**. *"This word is used a lot here, so it is probably about this."*
- **IDF (Inverse Document Frequency)** — how rare the word is **across all reviews**. *"This word is rare overall, so when it does appear it is informative."*

**score = TF × IDF**

The multiplication is the clever part: a word scores high **only if it is frequent here AND rare overall.** That is a good working definition of "informative".

| Word | Frequent in this review? | Rare overall? | TF-IDF |
| --- | --- | --- | --- |
| `"the"` | yes | no (it is everywhere) | **low** |
| `"refund"` | yes | yes | **high** ✅ |
| `"quantum"` | no (absent) | yes | **zero** |

### Why we need it at all

Logistic Regression is arithmetic: multiply each input number by a weight and add them up. **You cannot multiply the word `"terrible"` by 0.7.** So text must become numbers first. TF-IDF is that conversion.

### What comes out

One row per review, one column per vocabulary term. In our project: **2,385 rows × 20,000 columns.**

Only **48,425 of those 47 million cells are non-zero (0.10%)** — because a 12-word review touches about 20 columns out of 20,000. This is called a **sparse matrix**, and SciPy stores only the non-zero values plus their positions. Storing it densely would mean 47 million numbers, nearly all zeros. Sparsity is what makes text classification practical.

### Related jargon

- **Vocabulary** — the list of every term the vectorizer knows, each mapped to a column number. Ours has 20,000 entries. It is learned during `fit` and then frozen.
- **Bag of words** — the family of approaches TF-IDF belongs to. Called that because it treats a review as an unordered *bag* of words: *"not good"* and *"good not"* produce identical unigram features. Bigrams (§14) are a partial patch for this.
- **Vectorizer** — the object that does the conversion (`TfidfVectorizer`).
- **Feature** — one column, i.e. one term. `"great"` is a feature; `"not good"` is a feature.

---

## 10. The TF-IDF formula in simple terms

### The idea in words

> **TF-IDF(word, review) = (how often the word appears in this review) × (how rare the word is across all reviews)**

### The formula scikit-learn actually uses

**Step 1 — Term Frequency**

```
TF(word, review) = number of times the word appears in that review
```

**Step 2 — Inverse Document Frequency**

```
                        1 + total number of reviews
IDF(word) = ln( ─────────────────────────────────────── ) + 1
                   1 + number of reviews containing word
```

The `ln` is a natural logarithm; the `+1`s are "smoothing" that prevents dividing by zero and stops any word getting a weight of exactly zero.

**Step 3 — Multiply**

```
TF-IDF = TF × IDF
```

**Step 4 — Normalise the row (L2 normalisation)**

Each review's row is divided by its own length, so every row has a total "length" of 1.

*Why?* So a long review does not beat a short one just for having more words. Without it, a 200-word review would have much bigger numbers than a 10-word one, and the model would treat length as sentiment. (`tests/test_features.py` asserts every row length is exactly 1.)

### A worked example

Suppose 100 reviews, and we are looking at one review containing the word `"delicious"` twice.

- `"delicious"` appears in 5 of the 100 reviews.
- TF = 2
- IDF = ln((1 + 100) / (1 + 5)) + 1 = ln(101/6) + 1 = ln(16.8) + 1 = 2.82 + 1 = **3.82**
- TF-IDF = 2 × 3.82 = **7.64** → then divided by the row length.

Now compare `"the"`, which appears in all 100 reviews:

- IDF = ln(101/101) + 1 = ln(1) + 1 = 0 + 1 = **1.0**

So `"delicious"` carries almost **4× the weight** of `"the"` per occurrence. That is IDF doing its job: common words get pushed towards irrelevance, distinctive words get amplified.

### Why the logarithm?

Without it, a word appearing in 1 review out of a million would get a weight of 1,000,000 — it would swamp everything. The log compresses that range so rare words are rewarded strongly but not absurdly.

---

## 11. What is term frequency?

**Term frequency (TF) = how many times a word appears in one document.**

"Document" here means one review.

For the review *"the food was good, really good"*:

| term | TF |
| --- | --- |
| the | 1 |
| food | 1 |
| was | 1 |
| good | **2** |
| really | 1 |

**The intuition:** a word repeated in a review is probably central to what that review is about.

**Its weakness on its own:** the highest TF in almost any English text belongs to `"the"`, `"a"`, `"is"` — words that tell you nothing. TF alone would make every document look the same. That is precisely the gap IDF fills (§12).

**A variant worth naming:** `sublinear_tf` replaces the count with `1 + log(count)`, so saying `"bad"` five times is not treated as five times more negative than saying it once. We **tested it and did not use it**, because our reviews are single sentences where almost no word repeats — there was nothing to dampen, and it cost 0.2 accuracy points. Being able to say "I tried it, measured it, and rejected it" is a strong answer.

---

## 12. What is inverse document frequency?

**Inverse document frequency (IDF) = a score that is HIGH for rare words and LOW for common words.**

"Document frequency" is the number of documents a word appears in. *Inverse* means we flip it: high document frequency → low score.

| Word | Appears in… | Document frequency | IDF |
| --- | --- | --- | --- |
| `"the"` | nearly every review | very high | **≈1.0 (lowest)** |
| `"good"` | many reviews | high | low-ish |
| `"refund"` | a few reviews | low | **high** |
| `"flawless"` | one review | very low | **highest** |

**The intuition:** a word that appears everywhere cannot distinguish between documents. A word that appears in only a few documents is highly distinctive — so when you *do* see it, it tells you a lot.

**Two important properties:**

1. **IDF is computed once, during `fit`, from the training documents only** — then frozen and reused. This is exactly where data leakage would creep in if you fitted on the whole dataset (§8).
2. **IDF is a property of the corpus, not of a single review.** Every review shares the same IDF values; only the TF part changes per review.

**The neat consequence:** IDF gives you *automatic, data-driven stopword removal*. You do not need to hand-write a list of boring words — `"the"` gets a near-minimum weight automatically, because the maths noticed it is everywhere. This is one more reason we do not remove stopwords manually (§6).

`tests/test_features.py` checks this directly: it asserts that `"quality"` (in 1 of 6 test documents) gets a higher IDF than `"phone"` (in 2 of 6).

---

## 13. Why TF-IDF instead of simple word counts?

Simple word counts are called **Bag of Words** or **CountVectorizer**: just count each word.

Consider the review *"the food was not good"* :

These are the **real** values our trained vectorizer produces for that sentence (unigrams only shown, and its IDF for reference):

| Term | Count (BoW) | TF-IDF | IDF |
| --- | --- | --- | --- |
| the | 1 | **0.1336** (tiny) | 1.805 |
| was | 1 | 0.2063 | 2.786 |
| not | 1 | 0.2472 | 3.339 |
| good | 1 | 0.2625 | 3.546 |
| food | 1 | **0.3127** | 4.223 |

**With plain counts, every one of these words is equally important — all are 1.** That is obviously wrong: `"the"` tells you nothing and `"food"` tells you what the review is about. TF-IDF corrects it, giving `"food"` **2.3× the weight of `"the"`**, purely because it noticed `"the"` appears in nearly every review and `"food"` does not. Nobody told it that — it is in the IDF column.

Four concrete advantages:

1. **Common words are automatically suppressed** — no hand-written stopword list needed.
2. **Distinctive words are amplified** — `"refund"`, `"flawless"`, `"disgusting"` get the weight they deserve.
3. **Length is normalised** — L2 normalisation means a long review does not out-shout a short one.
4. **It empirically works better** for text classification, and it is the standard baseline everyone compares against.

**The honest caveat:** TF-IDF is *not always* better. Multinomial Naive Bayes was literally designed for raw counts, and sometimes performs better with them. TF-IDF also does not fix the real weaknesses of bag-of-words — it still ignores word order, still cannot handle sarcasm, and still has no idea that `"great"` and `"excellent"` mean nearly the same thing (word embeddings solve that; TF-IDF does not).

---

## 14. What are n-grams?

**An n-gram is a sequence of n consecutive words.**

Take *"the food was not good"*:

| Type | n | What you get |
| --- | --- | --- |
| **Unigrams** | 1 | `the`, `food`, `was`, `not`, `good` |
| **Bigrams** | 2 | `the food`, `food was`, `was not`, `not good` |
| **Trigrams** | 3 | `the food was`, `food was not`, `was not good` |

We use `ngram_range=(1, 2)` — **unigrams and bigrams together.**

### Why bigrams matter so much here

Bag-of-words throws away word order. With unigrams only, *"not good"* becomes the two independent features `not` and `good` — and `good` is the second-strongest *positive* word in our model. The negation is invisible.

Bigrams fix this partially: **`"not good"` becomes its own single feature**, and the model can learn a negative weight for it, completely separately from `good`.

**And it demonstrably worked.** In our trained model:

| Feature | Weight |
| --- | --- |
| `good` | **+8.14** (positive) |
| `not good` | **−3.45** (negative) |

The model learned that the two-word phrase means something different from the word alone. This is the single best concrete example to give when asked why you used bigrams. Measured impact: `(1,2)` beat `(1,1)` by 0.2–0.4 accuracy points.

### Why not trigrams, or 4-grams?

**The cost grows fast and the returns collapse.**

- Adding bigrams took our vocabulary from a few thousand to 21,269 features. Trigrams would multiply it again.
- Longer n-grams are rarer, so most appear once or twice — not enough examples to learn a reliable weight from. That is overfitting waiting to happen.
- Sentiment negation is mostly a 2-word pattern (`not good`, `never again`, `too expensive`), so bigrams capture most of the available benefit.

`(1, 2)` is the standard sweet spot for sentiment analysis.

---

## 15. What is Logistic Regression?

**Logistic Regression is a model that predicts a probability between 0 and 1, used for classification.**

Despite the name, it classifies — it does not do regression. It is called "regression" because of how it works internally.

### How it works, in three steps

**Step 1 — give every feature a weight.** Training's whole job is finding these numbers. Ours learned:

| Feature | Weight |
| --- | --- |
| `great` | +10.49 |
| `good` | +8.14 |
| `not` | −9.33 |
| `bad` | −7.43 |
| `not good` | −3.45 |

Positive weight → pushes towards "Positive". Negative weight → pushes towards "Negative". Near zero → this word does not matter.

**Step 2 — add everything up.**

```
z = (weight₁ × feature₁) + (weight₂ × feature₂) + … + intercept
```

This total `z` can be any number, from very negative to very positive.

**Step 3 — squash it into a probability with the sigmoid function.**

```
probability = 1 / (1 + e^(−z))
```

The sigmoid is an S-shaped curve that maps any number to the range 0–1:

```
   1.0 ┤                    ╭──────────
       │                ╭───╯
   0.5 ┤ ─ ─ ─ ─ ─ ─╭──╯  ← z = 0 gives exactly 0.5
       │        ╭───╯
   0.0 ┤────────╯
       └────┬─────┬─────┬────
           −5     0    +5      z
```

Then: **probability > 0.5 → Positive, otherwise Negative.**

### A worked example from our model

For *"Fantastic quality and the delivery was really fast"* the model produced **P(positive) = 0.939**, and `src/predict.py` can show you the arithmetic:

```
fantastic      +0.9381
and            +0.4256
delivery       +0.3398
delivery was   +0.3398
...
```

Each line is that term's TF-IDF value × its learned weight. Add them all up (plus the intercept), push through the sigmoid, and you get 0.939. **This is why the project can explain its predictions: the explanation *is* the calculation**, not an approximation of it.

### How training finds the weights

It starts with all weights at zero and repeatedly nudges them to reduce a penalty called **log loss** (cross-entropy), which punishes confident wrong answers heavily. This nudging process is **gradient descent**; `solver="liblinear"` is the particular algorithm we use, chosen because it is efficient on small-to-medium sparse data. `max_iter=1000` simply gives it room to finish.

### Regularisation and `C`

Left alone, the model would give enormous weights to words that happen to appear in only one training review — memorising instead of learning. **Regularisation** adds a penalty for large weights, keeping the model simpler.

**`C` is the inverse of regularisation strength:**

- **small C** → strong penalty → weights squashed towards zero → simpler model, risks underfitting
- **large C** → weak penalty → weights free to grow → more flexible, risks overfitting

We chose `C=10` by 5-fold cross-validation **on the training set only**:

| C | 1 | 2 | 5 | **10** | 20 | 50 | 100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CV accuracy | 0.810 | 0.821 | 0.831 | **0.839** | 0.842 | 0.839 | 0.840 |

The curve flattens from `C=10` on, and everything above is within one standard deviation (±0.006). So we picked the **smallest `C` on the plateau** — same score, more regularisation, simpler model. *(If asked "why not C=20, it scored highest?" — because the difference is inside the noise, and between two equal models you take the simpler one.)*

---

## 16. Why Logistic Regression for text classification?

Six reasons, in order of how much they matter here:

**1. It is interpretable.** One weight per word. You can read the model and see it learned `great = +10.49` and `bad = −7.43`. You can explain any individual prediction by listing which words contributed what. For a portfolio project — and for any business that needs to justify decisions — this is enormous. A neural network gives you a number and a shrug.

**2. It works well on high-dimensional sparse data.** We have 20,000 features and 2,385 examples. That would break many algorithms; linear models handle it comfortably, because with that many dimensions the classes are usually close to linearly separable already.

**3. Text really is roughly linear.** Sentiment genuinely does behave additively: `"great"` pushes up, `"terrible"` pushes down, and a review with three positive words is more positive than one with one. A linear model matches the shape of the problem.

**4. It gives calibrated-ish probabilities.** `predict_proba` gives a genuine confidence, which the Streamlit app displays. (Support Vector Machines need extra work for this.)

**5. It is fast.** Trains in under a second, predicts instantly. Easy to iterate, cheap to serve.

**6. It is the standard baseline.** TF-IDF + Logistic Regression is what every practitioner tries first on text. If a fancier model cannot beat it, the fancier model is not worth its complexity. **Knowing your baseline is a professional skill.**

**The honest trade-off:** it cannot capture word order beyond bigrams, cannot understand that `"great"` and `"excellent"` are synonyms, and cannot detect sarcasm. A fine-tuned transformer would beat it. For this project, the interpretability and the fundamentals were the point.

---

## 17. What is Naive Bayes?

**Naive Bayes is a classifier that uses probability: it asks "which class makes this combination of words most likely?"**

We use **Multinomial** Naive Bayes, the variant designed for word counts.

### How it works

It is built on **Bayes' theorem**, which flips a conditional probability around:

```
P(Positive | words)  ∝  P(words | Positive) × P(Positive)
```

In plain English: *the chance this review is positive equals the chance a positive review would contain these words, times how common positive reviews are in general.*

Training is just counting. For each word, in each class: how often does it appear? So it learns things like *"`delicious` appears in 3% of positive reviews and 0.1% of negative ones"*. To classify, it multiplies those probabilities together for every word in the review and picks the class with the bigger result.

### Why "naive"?

Because it makes a **deliberately naive assumption: that every word is independent of every other word**, given the class.

That is plainly false. `"not"` and `"good"` appearing together is not a coincidence, and *"New York"* is not two independent words. Naive Bayes ignores all of that.

**And yet it works surprisingly well.** The reason is subtle and worth knowing: the assumption makes the *probability estimates* badly wrong, but classification only needs the *ranking* of the classes to be right. You can be quite wrong about the numbers and still pick the right winner.

### `alpha` — Laplace smoothing

`alpha=1.0` adds a small pretend count to every word. Without it, a word never seen in positive training reviews would have P = 0, and because Naive Bayes *multiplies* probabilities, a single zero would make the whole product zero — one unseen word could veto the entire prediction. Smoothing prevents that.

### Naive Bayes vs Logistic Regression

| | Naive Bayes | Logistic Regression |
| --- | --- | --- |
| Type | generative (models how data is produced) | discriminative (models the boundary) |
| Training | counting — extremely fast | iterative optimisation |
| Word independence | assumes it | does **not** assume it |
| Small data | often better | needs a bit more data |
| More data | plateaus early | keeps improving |
| Probabilities | poorly calibrated | better |
| Interpretability | good (log-probability ratios) | very good (one weight per word) |

**Our measured result:** Naive Bayes reached **81.07%** test accuracy vs **82.08%** for Logistic Regression — a 1-point gap. Notably, in an earlier run with different feature settings, **Naive Bayes was ahead**. It is a genuinely strong baseline, not a straw man.

---

## 18. Why compare different models?

**Because a single number means nothing on its own.** "82% accuracy" — is that good? You cannot tell without something to compare it to.

Five reasons to compare:

**1. To establish a baseline.** If Naive Bayes gets 81% in 0.01 seconds, then a complicated model getting 81.5% is not worth its complexity. Comparison is how you find out whether extra machinery earns its keep.

**2. No model is universally best** — the *No Free Lunch* theorem. The only way to know which one suits *your* data is to try more than one.

**3. It reveals when differences are noise.** Our two models differ by 1.0 percentage point, while cross-validation varies by ±1.2 points run to run. So the honest conclusion is **"Logistic Regression is slightly ahead, and more interpretable"** — not "Logistic Regression is better". Recognising that a gap is inside the noise is a genuinely senior instinct.

**4. Different models fail differently.** Comparing confusion matrices shows whether two models make the same mistakes. If they fail on different reviews, an ensemble might help.

**5. It justifies your final choice.** "I used Logistic Regression" is an assertion. "I compared it against Naive Bayes, it won by 1 point on the test set and is more interpretable, so I deployed it" is engineering.

**Why we chose Logistic Regression as the deployed model:** it scored slightly higher, *and* its one-weight-per-word structure is what makes the "which words drove this prediction?" feature in the app possible.

---

## 19. Accuracy

**Accuracy = the fraction of predictions that were correct.**

```
             correct predictions          TP + TN
accuracy = ───────────────────────  =  ─────────────────
              all predictions          TP + TN + FP + FN
```

**Our result: 0.8208 — 490 of 597 test reviews correct.**

```
(246 + 244) / 597 = 490 / 597 = 0.8208
```

**Strength:** everyone understands it instantly. Best single number for a non-technical audience.

### Why accuracy alone is not enough

**The problem is class imbalance.** Imagine 1,000 reviews, 950 positive and 50 negative. A model with one line of code — `return "Positive"` — scores:

```
950 / 1000 = 95% accuracy
```

95%! And it is completely useless: it **never once identifies an unhappy customer**, which is the entire business reason for building it. Its recall on the negative class is **0**.

Accuracy hides this because it lumps both classes together. Precision, recall, F1 and the confusion matrix expose it immediately. (`tests/test_evaluate.py` contains this exact scenario as a test, asserting accuracy = 0.9 while F1-macro < 0.5.)

**Two more reasons accuracy can mislead:**

- **It ignores the cost of different mistakes.** In medicine, missing a disease is far worse than a false alarm; accuracy treats them as equal.
- **It ignores confidence.** Being 51% sure and being 99% sure both count as one correct answer. (ROC-AUC captures this; ours is 0.8952.)

**For our project specifically:** our dataset is ~50/50 balanced, so accuracy *is* a fair headline number here — 50% is the coin-flip baseline, so 82% is real learning. But we still report the full set, because on the next dataset it might not be.

---

## 20. Precision

**Precision answers: "of everything I labelled positive, how much really was positive?"**

```
                 TP               true positives
precision = ───────────  =  ─────────────────────────────
              TP + FP        everything I called positive
```

**Our result (positive class): 0.8215.** We called 297 reviews positive; 244 of them really were.

```
244 / (244 + 53) = 244 / 297 = 0.8215
```

**Precision is about false alarms.** High precision = when I say "positive", you can trust me.

**Memory hook:** precision looks at the **column** of the confusion matrix — everything I *predicted* as positive.

**When precision matters most:** when acting on a wrong positive is expensive. A spam filter needs high precision — sending a real job offer to the spam folder is much worse than letting one spam email through. Same for arresting people, or approving loans.

**How to game it (and why that is bad):** only predict "positive" when you are 99.9% certain. You would predict positive twice a year with perfect precision — and miss almost everything. That is why precision is never reported alone.

---

## 21. Recall

**Recall answers: "of everything that really was positive, how much did I find?"**

```
              TP              true positives
recall = ───────────  =  ───────────────────────────
           TP + FN        everything that really was positive
```

**Our result (positive class): 0.8188.** There were 298 genuinely positive reviews; we found 244.

```
244 / (244 + 54) = 244 / 298 = 0.8188
```

**Recall is about misses.** High recall = I do not let many cases slip past.

Recall is also called **sensitivity** or the **true positive rate**.

**Memory hook:** recall looks at the **row** of the confusion matrix — everything that *actually was* positive.

**When recall matters most:** when *missing* a case is expensive. Cancer screening needs high recall — a false alarm means one more test, but a missed tumour can be fatal. Fraud detection, and safety alerts, are the same.

**How to game it:** predict "positive" for everything. Perfect recall (you missed nothing!) and terrible precision.

### The precision/recall trade-off

They pull against each other, and the dial between them is the **decision threshold** — we use 0.5.

| Threshold | Effect |
| --- | --- |
| Raise to 0.9 | only very confident positives → **precision ↑, recall ↓** |
| Lower to 0.1 | almost everything called positive → **recall ↑, precision ↓** |

**Which to favour is a business decision, not a maths one.** For flagging angry customers you would favour **recall** — better to over-flag than to let a furious customer go unanswered.

**In our results, precision (0.8215) and recall (0.8188) are almost identical**, and the same is true for the negative class. On a balanced dataset with a 0.5 threshold, that is the expected and desirable outcome: the model is not biased towards either sentiment.

---

## 22. F1-score

**F1 is one number that combines precision and recall — specifically, their harmonic mean.**

```
        2 × precision × recall
F1 = ───────────────────────────
         precision + recall
```

**Our result (positive class): 0.8202.**

```
2 × (0.8215 × 0.8188) / (0.8215 + 0.8188) = 0.8202
```

### Why the *harmonic* mean and not a simple average?

**Because the harmonic mean punishes imbalance.** Take precision = 1.0 and recall = 0.02 (the "only predict when 99.9% sure" model):

- Simple average: (1.0 + 0.02) / 2 = **0.51** — looks mediocre but acceptable
- F1: 2 × (1.0 × 0.02) / (1.0 + 0.02) = **0.039** — correctly brutal

The harmonic mean sits close to the *smaller* of the two numbers. **You cannot get a good F1 by maximising one metric and ignoring the other** — which is exactly the behaviour you want to prevent.

### Macro vs weighted average

| Average | How | When to use |
| --- | --- | --- |
| **Macro** | plain mean of each class's F1 | when every class matters equally, **even a small one** |
| **Weighted** | mean weighted by class size | when you want the big classes to dominate |
| **Micro** | pool all TP/FP/FN together | equals accuracy in binary classification |

**We report macro F1 = 0.8208.** On our balanced data macro and weighted are identical, but macro is the safer habit: on imbalanced data it refuses to let a large class hide a failure on a small one.

**When to lead with F1 instead of accuracy:** imbalanced data, or when false positives and false negatives are both genuinely costly.

---

## 23. Confusion matrix

**A confusion matrix is a table showing exactly which mistakes the model made.** It is the most informative single output in classification, because every other metric can be calculated from it.

### Our actual matrix (Logistic Regression, 597 test reviews)

|  | **Predicted Negative** | **Predicted Positive** |
| --- | --- | --- |
| **Actual Negative** | **246** ✅ True Negative | **53** ❌ False Positive |
| **Actual Positive** | **54** ❌ False Negative | **244** ✅ True Positive |

### The four cells

| Cell | Name | Meaning here |
| --- | --- | --- |
| **TP** = 244 | True Positive | positive review, correctly called positive |
| **TN** = 246 | True Negative | negative review, correctly called negative |
| **FP** = 53 | False Positive (**Type I error**) | negative review we wrongly called positive — *a false alarm; we missed an unhappy customer* |
| **FN** = 54 | False Negative (**Type II error**) | positive review we wrongly called negative — *we flagged a happy customer as a complaint* |

**The trick for remembering the names:** the second word is **what you predicted**; the first word is **whether you were right**. "False Positive" = you predicted Positive, and you were wrong.

### Everything else comes from this table

```
accuracy  = (244 + 246) / 597           = 0.8208
precision =  244 / (244 + 53)           = 0.8215
recall    =  244 / (244 + 54)           = 0.8188
F1        = 2·P·R / (P + R)             = 0.8202
```

### What ours tells us

- **107 total errors** (53 + 54) out of 597.
- The errors are **split almost evenly** — 53 vs 54. The model is not systematically over-cheerful or over-harsh. On a balanced dataset with a 0.5 threshold, that is exactly what you want to see.
- If instead we had seen, say, 10 FP and 97 FN, we would know the model was biased towards "negative" and could fix it by moving the threshold.

**Which error is worse depends on the business.** For "find me the angry customers", a **false positive is worse** — that is a complaint you never saw. So in production you might lower the threshold below 0.5 to catch more negatives, accepting more false alarms.

The picture version is saved to `reports/confusion_matrix.png` by `src/evaluate.py`.

---

## 24. Overfitting

**Overfitting is when a model memorises its training data instead of learning general patterns. It scores brilliantly on data it has seen and poorly on anything new.**

*Analogy:* a student who memorises the answers to last year's exam paper. Ask them the same questions and they are perfect. Change the numbers and they are lost. They learned the answers, not the subject.

### How to spot it

**Compare training score with test score. A big gap means overfitting.**

### Our own numbers — the honest version

| | Logistic Regression |
| --- | --- |
| Train accuracy | **1.0000** |
| Cross-validation accuracy | 0.8377 |
| **Test accuracy** | **0.8208** |

**Train 1.00 vs test 0.82 — an 18-point gap.** The model classifies its training data perfectly. **Yes, this is overfitting, and you should say so plainly if asked.**

Here is why we accepted it:

1. **It is near-inevitable at this shape.** With 20,000 features and 2,385 examples, there are more dimensions than data points, and a linear model can almost always separate the training set exactly. Text classification nearly always looks like this.
2. **The two honest estimates agree.** Cross-validation said 0.8377, the untouched test set said 0.8208 — close. If the test score had collapsed to 0.60, the gap would be a real problem. The model *did* learn something that generalises.
3. **We tried reducing it and it made things worse.** Turning regularisation up to `C=1` narrows the gap but drops real accuracy from 0.839 to 0.810. A smaller gap with a worse model is not an improvement.

So the correct framing: **the gap is a symptom of a small dataset, not of bad modelling.** More data is the fix — not more regularisation.

**One piece of real evidence of overfitting to own:** the model gives the word `"and"` a weight of **+4.53**. `"and"` has no sentiment. It scored highly because in this particular corpus, positive sentences happen to string clauses together (*"good value and fast delivery"*). That is the model learning a quirk of 2,385 sentences rather than learning English — exactly what more data would wash out.

### Causes and cures

| Cause | Cure |
| --- | --- |
| Too little data | **get more data** — usually the best fix |
| Too many features | `min_df`, `max_features`, feature selection |
| Model too flexible | more regularisation (lower `C`) |
| Tuned on the test set | tune with cross-validation instead (what we did) |
| Duplicate rows across the split | deduplicate (what we do) |

---

## 25. Underfitting

**Underfitting is the opposite problem: the model is too simple to capture the pattern, so it does badly on training data AND on test data.**

*Analogy:* a student who did not revise at all. Bad on the practice paper, bad on the exam. Consistently bad.

### How to spot it

**Low training score.** That is the giveaway. If a model cannot even fit the data it was shown, the problem is capacity, not generalisation.

| Symptom | Diagnosis |
| --- | --- |
| Train 0.99, test 0.82 | **overfitting** (our case) |
| Train 0.65, test 0.64 | **underfitting** |
| Train 0.85, test 0.84 | healthy |

### Causes and cures

| Cause | Cure |
| --- | --- |
| Model too simple | use a more flexible model |
| Too much regularisation | **raise `C`** |
| Too few features | add features — e.g. bigrams, or lower `min_df` |
| Over-aggressive preprocessing | stop deleting useful information |
| Under-trained | more iterations (`max_iter`) |

**We actually observed underfitting in this project.** At `C=1`, cross-validation accuracy was only 0.810 versus 0.839 at `C=10`. The regularisation penalty was so strong it crushed the weights and the model could not use the evidence available to it. Raising `C` fixed it — which is the textbook cure.

### The bias–variance trade-off

This is the concept underneath both:

- **Bias** = error from wrong assumptions → **underfitting**. The model is too rigid.
- **Variance** = error from sensitivity to the particular training data → **overfitting**. The model is too twitchy.

```
error
  │╲                                    ╱   ← total error
  │ ╲          ╱╲                     ╱
  │  ╲       ╱    ╲                 ╱
  │   ╲    ╱        ╲__          ╱      ← variance (rises with complexity)
  │    ╲ ╱             ╲ ___  ╱
  │     ╳                    ⌣          ← the sweet spot
  │    ╱ ╲___                            ← bias (falls with complexity)
  └──────────────────────────────────────
    simple  →  model complexity  →  complex
```

You cannot minimise both at once; you look for the sweet spot. **That is literally what our `C` grid search was doing** — walking from high-bias (`C=1`, underfit) to high-variance (`C=100`) and picking the best point in between.

---

## 26. Training vs testing data

| | Training set | Test set |
| --- | --- | --- |
| Size | 2,385 (80%) | 597 (20%) |
| Purpose | the model learns from it | measure how good the model is |
| Seen by the model? | yes, repeatedly | **no — once, at the very end** |
| Used to pick hyperparameters? | yes, via cross-validation | **never** |

**The rule: the test set is sacred.** You look at it once, at the end, to produce the number you report. The moment you start tuning against it, it stops being unseen and your reported score becomes optimistic — that is test-set leakage (§8).

### So how do you tune anything?

**Cross-validation, inside the training set.** This is the piece beginners usually miss, and it is worth being able to explain clearly.

**5-fold cross-validation** splits the *training* data into 5 equal parts, then:

```
Round 1:  [TEST] [train] [train] [train] [train]   -> score 1
Round 2:  [train] [TEST] [train] [train] [train]   -> score 2
Round 3:  [train] [train] [TEST] [train] [train]   -> score 3
Round 4:  [train] [train] [train] [TEST] [train]   -> score 4
Round 5:  [train] [train] [train] [train] [TEST]   -> score 5

Result = average of the 5 scores (and their standard deviation)
```

Every row gets used for both training and validating, just never at the same time. Two benefits:

1. **A more stable estimate** than one single split — plus a standard deviation, which tells you how much the number wobbles.
2. **The real test set is never touched.**

**Our numbers:** `0.8377 ± 0.0119`. That ±0.0119 is why we can say a 1-point gap between two models is inside the noise.

*(A third option you should be able to name: a separate **validation set** — a three-way train/validation/test split. Cross-validation is generally preferred on small datasets, because a 20% validation slice of 2,385 rows would be too small to measure anything reliably.)*

---

## 27. Feature engineering

**Feature engineering is turning raw data into the numbers a model can actually learn from.** It is usually where most of the real gains in a classical ML project come from — far more than swapping models.

**A feature** is one measurable input. In this project, **one feature = one term in the vocabulary**. `"great"` is a feature. `"not good"` is a feature. We have 20,000 of them.

### What we did, in order

1. **Text cleaning** (§6) — deciding what to remove *is* feature engineering, because every deletion removes a possible feature. Keeping `"not"` was a feature-engineering decision.
2. **Tokenisation** — splitting text into words. scikit-learn's default pattern keeps sequences of 2+ word characters, which is why `"don't"` becomes `"don"` (the `"t"` is dropped as too short).
3. **N-grams** (§14) — creating bigram features so negation is visible.
4. **TF-IDF weighting** (§9) — deciding *how much* each feature counts.
5. **Vocabulary limits** — `min_df=1` and `max_features=20000` decide which features are allowed to exist at all.
6. **L2 normalisation** — making review length irrelevant.

### The decisions, and the evidence for each

This is the part to be proud of: **every choice was measured with cross-validation on the training set, not assumed.**

| Decision | Chose | Evidence |
| --- | --- | --- |
| n-gram range | `(1, 2)` | beat `(1,1)` by 0.2–0.4 points; and `not good` was genuinely learned at −3.45 |
| `min_df` | `1` | CV: 1 → 0.839, 2 → 0.829, 3 → 0.819 |
| `max_features` | `20000` | full vocab is 21,269; the cap changed CV accuracy by 0.0000 — kept as a safety rail |
| `sublinear_tf` | **off** | cost 0.2 points; these are single sentences, so nothing repeats to dampen |
| Contraction expansion | **no** | moved CV by less than one standard deviation |
| Stopword removal | **no** | would destroy `"not"` — the strongest negative feature at −9.33 |

**The `min_df=1` choice is the one to volunteer in a viva**, because it goes against standard advice. The standard advice assumes a large corpus, where a once-seen word is probably a typo. Our training set is 2,385 short sentences, where a once-seen word is quite likely a real sentiment word like `"flawless"`. **Context beats convention, and measurement beats both.**

### Ideas we did not use (good answers for "what else could you do?")

- **Word embeddings** (Word2Vec, GloVe) — vectors that place `"great"` and `"excellent"` near each other, so the model can generalise across synonyms. TF-IDF cannot do this at all.
- **Character n-grams** — robust to typos and useful for other languages.
- **Hand-built features** — review length, exclamation-mark count, ALL-CAPS ratio, emoji.
- **Negation scoping** — rewriting `"not good"` as `"NOT_good"` across a whole clause.
- **Feature selection** — `SelectKBest` with chi-squared to keep only the most informative terms.

---

## 28. Model serialization

**Serialization is saving a trained Python object to a file so you can load it back later without retraining.** Also called "pickling" or "persisting" a model.

### Why it is necessary

Training and prediction have opposite requirements:

- **Training** happens rarely and can be slow.
- **Prediction** happens constantly and must be instant.

Without serialization, the Streamlit app would have to retrain the model every time someone clicked a button — reloading the CSV, refitting TF-IDF, refitting Logistic Regression. Absurd, and it would also mean the deployed model could silently differ from the one you measured.

### How we do it

```python
# in src/train.py — save
joblib.dump(primary_model, config.MODEL_PATH)
joblib.dump(vectorizer,    config.VECTORIZER_PATH)

# in src/predict.py — load
model      = joblib.load(config.MODEL_PATH)
vectorizer = joblib.load(config.VECTORIZER_PATH)
```

### ⚠️ You must save the vectorizer too — this is the key insight

**This is the most common beginner mistake in NLP projects, and a favourite viva question.**

The model is just a list of 20,000 weights. Weight number 4231 might mean `"not good"` — but the model does not know that. **The mapping from words to column numbers lives inside the vectorizer.**

If you saved only the model and rebuilt the vectorizer later, you would get a different vocabulary in a different order. Weight 4231 would then be applied to some unrelated word, and every prediction would be silently wrong. Not crashed — *wrong*, which is worse.

They are a matched pair. `tests/test_predict.py` asserts exactly this:

```python
assert model.coef_.shape[1] == len(vectorizer.vocabulary_)
```

The same logic means **`clean_text()` must be applied identically at training and prediction time.** Train on lowercased text and predict on raw text, and the features no longer line up. In this project both paths call the same function, and a test asserts `"THIS IS GREAT!!!"` and `"this is great"` give the same answer.

### joblib vs pickle

Both save Python objects. **joblib is preferred for scikit-learn** because it handles large NumPy arrays much more efficiently. It is also what the scikit-learn docs recommend.

### Things to be aware of

- **Security:** never load a `.joblib`/`.pickle` file from an untrusted source. Loading can execute arbitrary code.
- **Version drift:** a model saved with scikit-learn 1.9 may warn or break when loaded with a very different version. This is why `reports/metrics.json` records the versions used.
- **Not for long-term archival:** for that, save the training code and data so you can rebuild. Our `random_state=42` means `python -m src.train` reproduces the identical model.
- **`models/` is in `.gitignore`** — binary files that change every run bloat git history, and they are one command away from being rebuilt.

---

## 29. How Streamlit works

**Streamlit turns a Python script into a web app. You write top-to-bottom Python; it renders a web page.** No HTML, CSS, JavaScript, routing or callbacks required.

### The execution model — the important bit

**Streamlit re-runs your entire script from top to bottom on every single interaction.** Every click, every keystroke, every dropdown change: the whole file runs again.

That is what makes the code so simple — there is no event system, just a script where `st.button()` returns `True` on the run where it was clicked. But it has one big consequence:

> **Anything expensive must be cached, or it will happen on every click.**

Without caching, our app would reload both `.joblib` files from disk every time a user typed a character.

### The two caches we use

```python
@st.cache_resource      # for live objects: models, DB connections
def get_artifacts():
    return load_artifacts()

@st.cache_data          # for plain data: the metrics.json contents
def get_metrics():
    ...
```

| Decorator | For | Behaviour |
| --- | --- | --- |
| `st.cache_resource` | models, connections | one shared instance across all users |
| `st.cache_data` | DataFrames, dicts, JSON | a copy per caller, so mutating it is safe |

We also have `@lru_cache` on `load_artifacts()` in `predict.py`, so the CLI gets the same benefit without Streamlit.

### The widgets in our app

| Widget | Purpose |
| --- | --- |
| `st.text_area` | where the review is typed |
| `st.button` | "Analyze Sentiment"; returns `True` on the run it was clicked |
| `st.tabs` | separates single-review from batch-CSV mode |
| `st.file_uploader` | the CSV upload |
| `st.selectbox` | pick an example, or pick the text column |
| `st.metric` | the big confidence number |
| `st.progress` | the probability bars |
| `st.success` / `st.error` / `st.info` / `st.warning` | coloured result boxes |
| `st.dataframe` | the word-contribution table and batch results |
| `st.download_button` | download batch predictions |
| `st.stop()` | abort the run early if the model is missing |

### Running it

```bash
streamlit run app.py       # serves on http://localhost:8501
```

**One design decision worth mentioning in a viva:** the sidebar's performance figures are **read from `reports/metrics.json`**, not typed into the app. That way the displayed accuracy always matches what the model actually scored — the UI cannot drift into advertising a number the model never achieved.

**Why Streamlit for this project:** it is pure Python, takes minutes rather than days, and is ideal for demos and internal tools. **When not to use it:** it is not for high-traffic public products or fine-grained UI control — for that you would put a FastAPI endpoint behind a proper front end.

---

## 30. Limitations

Knowing your model's weaknesses is a strength. This is the honest list.

**1. No neutral class.** The biggest practical limitation. Training data has only positive and negative, so *"It arrived on Tuesday"* is forced into one of the two. Fixing it needs three-way labelled data, not a code change.

**2. Sarcasm and irony fail completely.** *"Great, it broke on day one"* contains `great` (+10.49), our strongest positive feature. A bag-of-words model has no mechanism for tone. This is not a bug to be fixed — it is inherent to the approach.

**3. The dataset is small and the sentences are short.** 2,982 sentences, 12.2 words on average. Real reviews are longer and mix opinions: *"lovely screen but terrible battery"* just gets summed up. **This is the main ceiling on our 82%.**

**4. Word order is mostly lost.** Bigrams catch `"not good"`, but nothing catches *"I thought it would be bad, but it was wonderful"*.

**5. Out-of-vocabulary words are invisible.** A word the model never saw contributes nothing at all. Typos and new slang are simply skipped. The app honestly reports how many words it recognised, so you can spot when a prediction is really a guess.

**6. No synonym understanding.** `"great"` and `"excellent"` are completely unrelated columns to this model. It must learn each one separately from examples. Word embeddings would fix this.

**7. English only.**

**8. Mixed domains.** Trained on movies + restaurants + electronics, so it is general but mediocre at each. `"long"` is good for a battery and bad for a queue; one model cannot hold both.

**9. It learned some corpus quirks.** `"and"` has a weight of +4.53 despite carrying no sentiment. Honest evidence of overfitting to a small corpus.

**10. Confidence is not reliability.** The model reports 93% confidence on sentences it sometimes gets wrong. Logistic Regression probabilities are reasonable but not calibrated guarantees.

**11. Things we did not measure, and therefore do not claim:** performance on long reviews, on other domains, on other languages, latency under load, or any business impact. All unmeasured, so no claims are made.

---

## 31. Future improvements

In order of value for effort — which is itself the answer to "what would you do next?"

**1. Get more and longer data.** By far the highest impact. The 82% is mostly a data ceiling, not a model ceiling. The same code on 50,000 full-length reviews would likely reach the high 80s. *If you are asked "how would you improve this?", lead with this.*

**2. Add a neutral class.** Needs three-way labelled data. The code barely changes — `predict_proba` just returns three columns instead of two — but it makes the tool far more honest in practice.

**3. Formalise the tuning with `GridSearchCV` on a `Pipeline`.** We tuned by hand in a notebook. Wrapping the vectorizer and classifier in a `Pipeline` and searching it makes the process reproducible *and* makes data leakage structurally impossible, since the pipeline refits the vectorizer inside every CV fold automatically.

**4. Try a couple more models.** `LinearSVC` is often a shade better than Logistic Regression on TF-IDF text; plain `CountVectorizer` would confirm TF-IDF is pulling its weight. Cheap experiments, real information.

**5. Calibrate the probabilities** with `CalibratedClassifierCV`, so "90% confident" genuinely means "right about 90% of the time". That matters if anyone acts on the confidence score.

**6. Do proper error analysis.** Print the 107 misclassified reviews and read them. This almost always teaches you more than another hyperparameter sweep — you find out whether the failures are sarcasm, mixed opinions, or mislabelled data.

**7. Handle negation explicitly.** Rewrite `"not good at all"` as `"not_good not_at not_all"` so negation propagates through a clause instead of only the adjacent word.

**8. Deploy it properly.** Wrap `predict.py` in a FastAPI endpoint, containerise with Docker, host the Streamlit app on Streamlit Community Cloud, and log real predictions so you can monitor for drift.

**9. Then, and only then, consider transformers.** A fine-tuned DistilBERT or a similar model would very likely beat this, and would handle word order and sarcasm far better, because it reads words in context rather than as a bag. The costs: you lose the ability to explain a prediction by reading a weight, you need a GPU and much more time, and the model becomes megabytes instead of kilobytes. **For this project, the classical approach was the point** — and having a solid, measured, interpretable baseline is exactly what tells you whether the complexity of a transformer is justified.

---

## Extra concepts worth knowing

Short definitions for things that come up around this project.

**Supervised vs unsupervised learning.** Supervised = training data comes with the right answers (our case: every review has a label). Unsupervised = no labels; you look for structure, e.g. clustering reviews by topic.

**Classification vs regression.** Classification predicts a category (Positive/Negative). Regression predicts a number (a star rating from 1.0 to 5.0). Ours is classification.

**Hyperparameter vs parameter.** A **parameter** is learned by the model during training (our 20,000 weights). A **hyperparameter** is set by *you* before training (`C`, `min_df`, `ngram_range`). You tune hyperparameters with cross-validation.

**Sigmoid function.** `1 / (1 + e^(−z))` — squashes any number into 0–1 so it can be read as a probability. The final step in Logistic Regression.

**Log loss (cross-entropy).** The penalty Logistic Regression minimises during training. It punishes confident wrong answers far more than uncertain ones.

**Gradient descent.** The optimisation method: nudge the weights in the direction that reduces the loss, repeatedly, until it stops improving.

**Regularisation (L1 vs L2).** A penalty on large weights to prevent overfitting. **L2** (our default) shrinks all weights smoothly. **L1** drives some weights to exactly zero, which effectively performs feature selection.

**Sparse vs dense matrix.** Sparse stores only non-zero values and their positions; dense stores every cell. Our matrix is 0.10% non-zero, so sparse saves roughly 1,000× the memory.

**ROC-AUC.** "Area under the ROC curve" — the probability that the model ranks a randomly chosen positive review above a randomly chosen negative one. It judges *ranking* across all thresholds, so unlike accuracy it does not depend on the 0.5 cut-off. **Ours is 0.8952.** (0.5 = random guessing, 1.0 = perfect.)

**Decision threshold.** The cut-off applied to the probability — we use 0.5. Changing it trades precision against recall without retraining anything.

**Class imbalance.** When one class vastly outnumbers another. Fixes: `class_weight="balanced"`, resampling (SMOTE), or simply reporting metrics that expose the problem. **Our data is balanced, so we did not need any of these** — but you should be able to say what you *would* do.

**Baseline model.** The simplest thing that could work, used for comparison. Here, "always predict positive" would score ~50% on our balanced data; our 82% is meaningful against that.

**Model drift.** Real-world language changes (new products, new slang), so a deployed model slowly gets worse. The cure is monitoring and periodic retraining.

**Tokenisation.** Splitting text into units (tokens). scikit-learn's default keeps runs of 2+ word characters, which is why single letters and punctuation vanish.

---

## Cheat sheet: the numbers to remember

Memorise this table. Most viva questions about results are answered from it.

| Thing | Value |
| --- | --- |
| Dataset | UCI Sentiment Labelled Sentences (Amazon, Yelp, IMDb) |
| Rows downloaded → usable | 3,000 → **2,982** (18 dropped) |
| Class balance | 1,492 negative / 1,490 positive (≈50/50) |
| Average review length | 12.2 words |
| Split | 80/20 stratified, `random_state=42` |
| Train / test size | **2,385 / 597** |
| Features | TF-IDF, unigrams + bigrams, `min_df=1`, `max_features=20000` |
| Vocabulary | **20,000** terms (capped from 21,269) |
| Matrix sparsity | 48,425 non-zero cells = **0.10%** |
| Primary model | Logistic Regression, `C=10`, `liblinear` |
| Baseline model | Multinomial Naive Bayes, `alpha=1.0` |
| **Test accuracy (LR)** | **0.8208** (490/597 correct) |
| Test accuracy (NB) | 0.8107 |
| Precision / Recall / F1 (positive) | 0.8215 / 0.8188 / 0.8202 |
| F1 macro | 0.8208 |
| ROC-AUC | 0.8952 |
| Cross-validation (LR) | 0.8377 ± 0.0119 |
| Train accuracy (LR) | 1.0000 — **overfitting, and we say so** |
| Confusion matrix | TN 246, FP 53, FN 54, TP 244 |
| Total errors | 107 of 597 |
| Strongest positive / negative feature | `great` +10.49 / `not` −9.33 |
| Proof bigrams worked | `good` +8.14 vs **`not good` −3.45** |
| Tests | **86 passing** |

**If you remember only five things:**

1. **The pipeline:** clean → split → TF-IDF → train → evaluate → save → serve.
2. **TF-IDF** = frequent in this review × rare across all reviews, so distinctive words win.
3. **Fit the vectorizer on train only** — otherwise data leakage, and your score is a lie.
4. **Accuracy alone misleads** on imbalanced data; report precision, recall, F1 and the confusion matrix.
5. **82.08% on 597 unseen reviews, and I can tell you exactly which words drove any prediction.**
