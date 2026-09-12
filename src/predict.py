"""
Make predictions with the saved model.

This is the "inference" side of the project. Training happens once and is
slow-ish; prediction happens many times and must be fast, so we load the two
saved artifacts and reuse them:

    tfidf_vectorizer.joblib -> turns text into the same numbers as training
    sentiment_model.joblib  -> turns those numbers into a label

The order matters and must match training exactly:
    raw text -> clean_text() -> vectorizer.transform() -> model.predict()

Run from the command line:
    python -m src.predict "the battery died after two days"
    python -m src.predict            (starts a small interactive prompt)
"""

from __future__ import annotations

import sys
from functools import lru_cache

import joblib

from src import config
from src.preprocessing import clean_text


@lru_cache(maxsize=1)
def load_artifacts(model_path=None, vectorizer_path=None):
    """Load (model, vectorizer) from disk.

    lru_cache means the files are read once per process. Streamlit reruns the
    whole script on every click, so without caching we would reload the model
    from disk on every single interaction.
    """
    model_path = model_path or config.MODEL_PATH
    vectorizer_path = vectorizer_path or config.VECTORIZER_PATH

    for path in (model_path, vectorizer_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing artifact: {path}\n"
                "Train the model first:  python -m src.train"
            )

    return joblib.load(model_path), joblib.load(vectorizer_path)


def predict_sentiment(text: str) -> dict:
    """Predict the sentiment of one review.

    Returns a dict:
        {
          "label": 1,
          "sentiment": "Positive",
          "confidence": 0.93,          # probability of the predicted class
          "probability_positive": 0.93,
          "probability_negative": 0.07,
          "cleaned_text": "great phone lovely screen",
          "is_empty": False,           # True if cleaning left nothing usable
          "known_words": 4,            # how many words the model recognised
        }

    `confidence` is the model's own probability, not a guarantee of being
    right. A model can be confidently wrong -- especially on sarcasm, or on
    words it has never seen.
    """
    model, vectorizer = load_artifacts()

    cleaned = clean_text(text)
    if not cleaned:
        # Nothing left to classify (empty input, or only punctuation/emoji).
        return {
            "label": None,
            "sentiment": "Unknown",
            "confidence": 0.0,
            "probability_positive": 0.0,
            "probability_negative": 0.0,
            "cleaned_text": "",
            "is_empty": True,
            "known_words": 0,
        }

    row = vectorizer.transform([cleaned])          # 1 x n_features sparse row
    label = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]    # [P(negative), P(positive)]

    return {
        "label": label,
        "sentiment": config.LABEL_NAMES[label],
        "confidence": float(probabilities[label]),
        "probability_positive": float(probabilities[1]),
        "probability_negative": float(probabilities[0]),
        "cleaned_text": cleaned,
        "is_empty": False,
        # If this is 0, every word was out-of-vocabulary and the prediction is
        # really just the model's default bias -- worth telling the user.
        "known_words": int(row.nnz),
    }


def predict_batch(texts) -> list[dict]:
    """Predict many reviews at once.

    One vectorizer.transform() call on the whole list is much faster than
    calling predict_sentiment() in a loop, because the work happens inside
    NumPy/SciPy instead of in Python.
    """
    model, vectorizer = load_artifacts()

    cleaned = [clean_text(text) for text in texts]
    # Remember which rows survived cleaning so we can put them back in order.
    usable_indexes = [i for i, text in enumerate(cleaned) if text]

    results: list[dict] = [
        {
            "label": None, "sentiment": "Unknown", "confidence": 0.0,
            "probability_positive": 0.0, "probability_negative": 0.0,
            "cleaned_text": "", "is_empty": True, "known_words": 0,
        }
        for _ in texts
    ]
    if not usable_indexes:
        return results

    rows = vectorizer.transform([cleaned[i] for i in usable_indexes])
    labels = model.predict(rows)
    probabilities = model.predict_proba(rows)

    for position, index in enumerate(usable_indexes):
        label = int(labels[position])
        probability = probabilities[position]
        results[index] = {
            "label": label,
            "sentiment": config.LABEL_NAMES[label],
            "confidence": float(probability[label]),
            "probability_positive": float(probability[1]),
            "probability_negative": float(probability[0]),
            "cleaned_text": cleaned[index],
            "is_empty": False,
            "known_words": int(rows[position].nnz),
        }
    return results


def explain_prediction(text: str, top_n: int = 6) -> list[tuple[str, float]]:
    """Show which words pushed the prediction, strongest first.

    For each term present in this review we multiply its TF-IDF value by the
    model's weight for that term. The product is that term's contribution to
    the final score: positive values argue for "Positive", negative values
    argue for "Negative". Summing all of them (plus the intercept) gives
    exactly the number the sigmoid turns into the probability -- so this is a
    faithful explanation, not an approximation.
    """
    model, vectorizer = load_artifacts()

    cleaned = clean_text(text)
    if not cleaned:
        return []

    row = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]

    contributions = [
        (feature_names[column], float(row[0, column] * weights[column]))
        for column in row.indices
    ]
    contributions.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return contributions[:top_n]


def _interactive() -> None:
    print("Sentiment analyser. Type a review, or 'quit' to exit.\n")
    while True:
        try:
            text = input("review> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        result = predict_sentiment(text)
        print(f"  -> {result['sentiment']} "
              f"(confidence {result['confidence']:.1%})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        review = " ".join(sys.argv[1:])
        outcome = predict_sentiment(review)
        print(f"Review    : {review}")
        print(f"Sentiment : {outcome['sentiment']}")
        print(f"Confidence: {outcome['confidence']:.1%}")
        print(f"P(positive)={outcome['probability_positive']:.3f}  "
              f"P(negative)={outcome['probability_negative']:.3f}")
        drivers = explain_prediction(review)
        if drivers:
            print("Top drivers:")
            for term, contribution in drivers:
                direction = "positive" if contribution > 0 else "negative"
                print(f"  {term:<20} {contribution:+.4f}  ({direction})")
    else:
        _interactive()
