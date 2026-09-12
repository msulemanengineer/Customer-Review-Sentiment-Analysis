"""Tests for model loading and prediction.

These tests need the trained artifacts in models/. If they are missing the
whole module is skipped with a clear message instead of failing confusingly.

    python -m src.download_data
    python -m src.train
    pytest -v
"""

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src import config
from src.predict import (
    explain_prediction,
    load_artifacts,
    predict_batch,
    predict_sentiment,
)

pytestmark = pytest.mark.skipif(
    not (config.MODEL_PATH.exists() and config.VECTORIZER_PATH.exists()),
    reason="Trained artifacts not found. Run `python -m src.train` first.",
)

CLEARLY_POSITIVE = [
    "Absolutely love this product, it works perfectly.",
    "Great quality and excellent value for money.",
    "The food was delicious and the service was wonderful.",
]
CLEARLY_NEGATIVE = [
    "Terrible quality, it broke after one day.",
    "Awful experience, waste of money.",
    "The worst purchase I have ever made.",
]


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def test_load_artifacts_returns_model_and_vectorizer():
    model, vectorizer = load_artifacts()
    assert isinstance(model, LogisticRegression)
    assert isinstance(vectorizer, TfidfVectorizer)


def test_loaded_vectorizer_is_fitted():
    _, vectorizer = load_artifacts()
    # A fitted TfidfVectorizer has these attributes; an unfitted one does not.
    assert hasattr(vectorizer, "vocabulary_")
    assert hasattr(vectorizer, "idf_")
    assert len(vectorizer.vocabulary_) > 0


def test_model_and_vectorizer_agree_on_feature_count():
    """The single most important consistency check in the project.

    The model has one weight per feature. If the saved vectorizer produced a
    different number of columns than the model was trained on, every
    prediction would be garbage -- so we assert they match.
    """
    model, vectorizer = load_artifacts()
    assert model.coef_.shape[1] == len(vectorizer.vocabulary_)


def test_load_artifacts_is_cached():
    """lru_cache should hand back the very same objects, not reload them."""
    first_model, first_vectorizer = load_artifacts()
    second_model, second_vectorizer = load_artifacts()
    assert first_model is second_model
    assert first_vectorizer is second_vectorizer


def test_missing_artifact_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        load_artifacts(
            model_path=tmp_path / "nope.joblib",
            vectorizer_path=tmp_path / "nope2.joblib",
        )


def test_saved_artifacts_survive_a_reload_round_trip(tmp_path):
    """joblib.dump -> joblib.load must give an equally usable model.

    This is what "model serialisation" has to guarantee: the reloaded object
    produces identical predictions, otherwise the app and the training script
    would disagree.
    """
    model, vectorizer = load_artifacts()
    copy_path = tmp_path / "copy.joblib"
    joblib.dump(model, copy_path)
    reloaded = joblib.load(copy_path)

    row = vectorizer.transform(["great product, works perfectly"])
    assert reloaded.predict(row)[0] == model.predict(row)[0]


# ---------------------------------------------------------------------------
# single prediction
# ---------------------------------------------------------------------------

def test_prediction_has_the_expected_shape():
    result = predict_sentiment("This product is great.")
    for key in (
        "label", "sentiment", "confidence", "probability_positive",
        "probability_negative", "cleaned_text", "is_empty", "known_words",
    ):
        assert key in result


def test_probabilities_sum_to_one():
    result = predict_sentiment("Good enough for the price.")
    total = result["probability_positive"] + result["probability_negative"]
    assert total == pytest.approx(1.0)


def test_confidence_matches_the_predicted_class():
    result = predict_sentiment("I really love it.")
    expected = (
        result["probability_positive"] if result["label"] == 1
        else result["probability_negative"]
    )
    assert result["confidence"] == pytest.approx(expected)
    # The winning class always has at least half the probability mass.
    assert 0.5 <= result["confidence"] <= 1.0


@pytest.mark.parametrize("text", CLEARLY_POSITIVE)
def test_clearly_positive_reviews_are_predicted_positive(text):
    result = predict_sentiment(text)
    assert result["label"] == 1, f"got {result['sentiment']} for: {text}"
    assert result["sentiment"] == "Positive"


@pytest.mark.parametrize("text", CLEARLY_NEGATIVE)
def test_clearly_negative_reviews_are_predicted_negative(text):
    result = predict_sentiment(text)
    assert result["label"] == 0, f"got {result['sentiment']} for: {text}"
    assert result["sentiment"] == "Negative"


def test_prediction_is_deterministic():
    """Same input must always give the same output -- no hidden randomness."""
    text = "The screen is lovely but the battery is poor."
    first = predict_sentiment(text)
    second = predict_sentiment(text)
    assert first == second


def test_case_and_punctuation_do_not_change_the_answer():
    """clean_text() runs before the vectorizer, so these must agree."""
    plain = predict_sentiment("this product is great")
    shouty = predict_sentiment("THIS PRODUCT IS GREAT!!!")
    assert plain["label"] == shouty["label"]
    assert plain["confidence"] == pytest.approx(shouty["confidence"])


def test_empty_input_is_handled_not_guessed():
    for text in ("", "   ", "!!!???", "***"):
        result = predict_sentiment(text)
        assert result["is_empty"] is True
        assert result["label"] is None
        assert result["sentiment"] == "Unknown"


def test_out_of_vocabulary_input_is_flagged():
    """Nonsense words -> nothing recognised -> known_words == 0.

    The app uses this to warn the user that the prediction is a guess.
    """
    result = predict_sentiment("zzqqxx wubbleflib grondlewax")
    assert result["is_empty"] is False
    assert result["known_words"] == 0


def test_known_words_counts_recognised_terms():
    result = predict_sentiment("great product")
    assert result["known_words"] > 0


# ---------------------------------------------------------------------------
# batch prediction
# ---------------------------------------------------------------------------

def test_batch_returns_one_result_per_input():
    texts = CLEARLY_POSITIVE + CLEARLY_NEGATIVE
    results = predict_batch(texts)
    assert len(results) == len(texts)


def test_batch_matches_single_prediction():
    """The fast path and the simple path must never disagree."""
    texts = ["Great value.", "Broke immediately.", "Lovely design."]
    batch = predict_batch(texts)
    for text, batch_result in zip(texts, batch):
        single = predict_sentiment(text)
        assert batch_result["label"] == single["label"]
        assert batch_result["confidence"] == pytest.approx(single["confidence"])


def test_batch_keeps_row_order_with_empty_rows_mixed_in():
    """An unusable row must not shift the other results up a slot."""
    results = predict_batch(["Great product!", "", "Terrible product!", "!!!"])
    assert results[0]["label"] == 1
    assert results[1]["is_empty"] is True
    assert results[2]["label"] == 0
    assert results[3]["is_empty"] is True


def test_batch_of_only_empty_rows():
    results = predict_batch(["", "  ", "???"])
    assert all(result["is_empty"] for result in results)


def test_batch_of_nothing():
    assert predict_batch([]) == []


# ---------------------------------------------------------------------------
# explanation helper
# ---------------------------------------------------------------------------

def test_explain_returns_terms_from_the_review():
    drivers = explain_prediction("the food was delicious", top_n=5)
    assert drivers
    terms = [term for term, _ in drivers]
    # Every returned term must actually come from the input text.
    assert all(
        all(word in "the food was delicious" for word in term.split())
        for term in terms
    )


def test_explain_is_sorted_by_strength():
    drivers = explain_prediction("great food but terrible service", top_n=6)
    strengths = [abs(contribution) for _, contribution in drivers]
    assert strengths == sorted(strengths, reverse=True)


def test_explain_empty_input_returns_nothing():
    assert explain_prediction("") == []
    assert explain_prediction("!!!") == []
