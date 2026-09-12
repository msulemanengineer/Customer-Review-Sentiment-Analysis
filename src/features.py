"""
Feature extraction: turning text into numbers with TF-IDF.

WHY WE NEED THIS
Logistic Regression is maths: it multiplies each input number by a weight and
adds them up. It cannot multiply the word "terrible" by anything. So before
training we must convert each review into a fixed-length vector of numbers.
That conversion is called feature extraction.

WHAT TF-IDF DOES
TF-IDF = Term Frequency x Inverse Document Frequency.
  * TF  -- how often a word appears in THIS review. A word used a lot in a
           review is probably important to that review.
  * IDF -- how rare the word is across ALL reviews. Words like "the" appear
           everywhere, so they get a small weight. Words like "refund" appear
           in few reviews, so they get a large weight.
  * score = TF x IDF, so a word scores high only when it is frequent *here*
    and rare *overall* -- which is exactly what makes it informative.

The result is a sparse matrix: one row per review, one column per vocabulary
term. Almost every cell is 0 (a 12-word review has 12 non-zero columns out of
thousands), so scikit-learn stores only the non-zero values instead of the
full grid. That is what "sparse" means, and it is why this scales.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_features: int = 20_000,
) -> TfidfVectorizer:
    """Create the TF-IDF vectorizer.

    Every parameter below is here for a reason, and the reasons were CHECKED
    with 5-fold cross-validation on the training set (never on the test set).
    The full grid is in notebooks/sentiment_analysis.ipynb.

    ngram_range=(1, 2)
        Use single words (unigrams) AND two-word pairs (bigrams). This is the
        cheapest way to let a bag-of-words model notice negation: "not" and
        "good" separately look positive-ish, but the bigram "not good" is its
        own feature the model can learn to score negatively.
        Measured: (1, 2) beat (1, 1) by roughly 0.2-0.4 accuracy points.

    min_df=1
        Keep every term, even ones that appear in a single review.
        This is NOT the usual default advice, so it needs justifying: our
        training set is only ~2,400 short one-sentence reviews, so plenty of
        genuinely useful sentiment words ("refund", "flawless") appear exactly
        once. Measured CV accuracy: min_df=1 -> 0.839, min_df=2 -> 0.829,
        min_df=3 -> 0.819. The data says keep them.
        On a much larger corpus min_df=2 usually does help, because then a
        once-seen word really is a typo rather than a rare real word.

    max_features=20000
        Keep at most the 20,000 most frequent terms. With this dataset the
        full vocabulary is 21,269 terms, so the cap only removes the ~1,300
        rarest ones and CV accuracy is unchanged to 4 decimal places. It stays
        in as a cheap safety rail: if you swap in a 500k-review dataset the
        matrix cannot silently explode.

    strip_accents="unicode"
        "cafe" and "caf\u00e9" collapse into one feature instead of two.

    lowercase=False
        clean_text() already lowercased everything, so doing it again would
        just waste time. Kept explicit so the decision is visible in the code.

    NOT USED: sublinear_tf. It replaces the raw count with 1 + log(count) to
    dampen repeated words. Our reviews are single sentences, so almost no word
    repeats inside one review and there is nothing to dampen -- measured, it
    changed CV accuracy by -0.2 points. We left it off rather than carry a
    parameter that does nothing here.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        strip_accents="unicode",
        lowercase=False,
    )


def fit_transform_train(vectorizer: TfidfVectorizer, train_texts):
    """Learn the vocabulary + IDF weights from the TRAINING text only.

    `fit` is what builds the vocabulary and computes the IDF values.
    It must see training data only -- see transform_test() below.
    """
    return vectorizer.fit_transform(train_texts)


def transform_test(vectorizer: TfidfVectorizer, texts):
    """Convert new text using the ALREADY-FITTED vectorizer.

    DATA LEAKAGE WARNING -- this is the classic mistake in NLP projects.
    If you call fit_transform() on the full dataset before splitting, the IDF
    weights and the vocabulary are computed using the test reviews too. The
    model then has indirect knowledge of data it is supposed to have never
    seen, and your test score comes out too optimistic.

    Correct order: split first, fit on train, transform test.
    Words in the test set that the vectorizer never saw are simply ignored --
    which is also what happens in production with a brand-new review.
    """
    return vectorizer.transform(texts)


def top_features_for_class(vectorizer, model, n: int = 15):
    """Return the n most positive and n most negative terms the model learned.

    This is the "explainability" bit. For binary Logistic Regression there is
    one weight (coefficient) per feature. A large positive weight pushes the
    prediction towards class 1 (Positive); a large negative weight pushes it
    towards class 0 (Negative). Reading these is a quick sanity check that the
    model learned sentiment and not some artefact of the data.
    """
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]                     # shape: (n_features,)
    order = weights.argsort()                    # ascending: negative first

    most_negative = [(feature_names[i], float(weights[i])) for i in order[:n]]
    most_positive = [(feature_names[i], float(weights[i])) for i in order[-n:][::-1]]
    return most_positive, most_negative
