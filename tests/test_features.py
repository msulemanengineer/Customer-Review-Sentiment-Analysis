"""Tests for TF-IDF feature extraction."""

import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from src.features import (
    build_vectorizer,
    fit_transform_train,
    top_features_for_class,
    transform_test,
)

TRAIN_TEXTS = [
    "great phone i love it",
    "terrible phone it broke",
    "good value and fast delivery",
    "bad value and slow delivery",
    "not good at all",
    "really good quality",
]
TRAIN_LABELS = [1, 0, 1, 0, 0, 1]


@pytest.fixture
def fitted():
    """A vectorizer fitted on TRAIN_TEXTS, plus the resulting matrix."""
    vectorizer = build_vectorizer()
    matrix = fit_transform_train(vectorizer, TRAIN_TEXTS)
    return vectorizer, matrix


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def test_vectorizer_uses_the_parameters_we_documented():
    vectorizer = build_vectorizer()
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.min_df == 1
    assert vectorizer.max_features == 20_000
    # clean_text() already lowercases; doing it twice would waste time.
    assert vectorizer.lowercase is False


def test_vectorizer_builds_bigrams():
    """Bigrams are how a bag-of-words model can notice negation."""
    vectorizer, _ = build_vectorizer(), None
    matrix = fit_transform_train(vectorizer, TRAIN_TEXTS)
    vocabulary = vectorizer.get_feature_names_out()

    assert "not good" in vocabulary          # two-word feature
    assert "good" in vocabulary              # single-word feature
    assert matrix.shape[1] == len(vocabulary)


# ---------------------------------------------------------------------------
# the output matrix
# ---------------------------------------------------------------------------

def test_output_is_a_sparse_matrix(fitted):
    _, matrix = fitted
    # Dense storage would be mostly zeros; sparse is why this scales.
    assert issparse(matrix)
    assert matrix.nnz < matrix.shape[0] * matrix.shape[1]


def test_one_row_per_document(fitted):
    _, matrix = fitted
    assert matrix.shape[0] == len(TRAIN_TEXTS)


def test_tfidf_values_are_l2_normalised(fitted):
    """TfidfVectorizer normalises each row to length 1 by default.

    That keeps a long review from outweighing a short one just because it has
    more words.
    """
    _, matrix = fitted
    row_lengths = np.sqrt(matrix.multiply(matrix).sum(axis=1)).ravel()
    assert np.allclose(np.asarray(row_lengths).ravel(), 1.0)


def test_tfidf_values_are_non_negative(fitted):
    # Multinomial Naive Bayes cannot accept negative features, so this
    # property is what lets us feed the same matrix to both models.
    _, matrix = fitted
    assert matrix.min() >= 0


def test_rare_word_scores_higher_than_common_word(fitted):
    """The 'IDF' half of TF-IDF in one assertion.

    "phone" appears in 2 of our 6 documents; "quality" in only 1. Both appear
    once in their own document, so the rarer word must get the bigger weight.
    """
    vectorizer, _ = fitted
    idf = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    assert idf["quality"] > idf["phone"]


# ---------------------------------------------------------------------------
# fit on train, transform test (the data-leakage rule)
# ---------------------------------------------------------------------------

def test_transform_test_keeps_the_same_columns(fitted):
    vectorizer, train_matrix = fitted
    test_matrix = transform_test(vectorizer, ["great value", "terrible quality"])
    # Same number of columns = same feature meaning. If this ever differed,
    # the model's weights would be lined up against the wrong words.
    assert test_matrix.shape[1] == train_matrix.shape[1]


def test_unseen_words_are_ignored_not_added(fitted):
    """Words the vectorizer never saw during fit are dropped silently.

    That is correct behaviour: in production a brand-new review will contain
    unknown words, and we cannot invent a weight for them.
    """
    vectorizer, _ = fitted
    vocabulary_size = len(vectorizer.vocabulary_)

    row = transform_test(vectorizer, ["zzzquux unseenwordhere"])
    assert row.nnz == 0                                # nothing matched
    assert len(vectorizer.vocabulary_) == vocabulary_size  # vocab unchanged


def test_transform_before_fit_raises():
    """You cannot transform with a vectorizer that was never fitted."""
    with pytest.raises(NotFittedError):
        transform_test(build_vectorizer(), ["anything"])


def test_transform_is_deterministic(fitted):
    vectorizer, _ = fitted
    first = transform_test(vectorizer, ["great phone"])
    second = transform_test(vectorizer, ["great phone"])
    assert (first != second).nnz == 0


# ---------------------------------------------------------------------------
# interpretability helper
# ---------------------------------------------------------------------------

def test_top_features_returns_sensible_weights(fitted):
    vectorizer, matrix = fitted
    model = LogisticRegression(solver="liblinear", C=10.0, random_state=42)
    model.fit(matrix, TRAIN_LABELS)

    top_positive, top_negative = top_features_for_class(vectorizer, model, n=3)

    assert len(top_positive) == 3
    assert len(top_negative) == 3
    # Positive weights push towards class 1, negative weights towards class 0.
    assert all(weight > 0 for _, weight in top_positive)
    assert all(weight < 0 for _, weight in top_negative)
    # Sorted strongest-first.
    assert top_positive[0][1] >= top_positive[-1][1]
