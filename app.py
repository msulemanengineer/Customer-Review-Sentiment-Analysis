"""
Streamlit user interface for the sentiment model.

Run with:
    streamlit run app.py

HOW STREAMLIT WORKS (worth knowing for the viva)
Streamlit re-runs this entire Python file from top to bottom every time the
user interacts with the page -- every click, every keystroke in a text box.
That model is what makes the code so simple (no callbacks, no routing, just a
script), but it also means anything expensive must be cached, otherwise we
would reload the model from disk on every single click.

We cache in two places:
  * @st.cache_resource on the loader below, and
  * @lru_cache inside src/predict.load_artifacts()

The app NEVER trains. It only loads models/sentiment_model.joblib and
models/tfidf_vectorizer.joblib, which `python -m src.train` produced.
"""

from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

from src import config
from src.predict import explain_prediction, load_artifacts, predict_batch, predict_sentiment

st.set_page_config(
    page_title="Customer Review Sentiment Analysis",
    page_icon="💬",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def get_artifacts():
    """Load the model once and keep it in memory across reruns.

    st.cache_resource is the right decorator for things that are not data but
    live objects (a model, a database connection). Streamlit keeps one shared
    copy instead of one per user session.
    """
    return load_artifacts()


@st.cache_data(show_spinner=False)
def get_metrics() -> dict | None:
    """Read the measured scores that `python -m src.train` wrote.

    We read them from disk instead of typing them into this file, so the app
    can never display a number the model did not actually achieve.
    st.cache_data (not cache_resource) is right here: this is plain data.
    """
    if not config.METRICS_PATH.exists():
        return None
    return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))


def confidence_note(result: dict) -> str:
    """Plain-English explanation of what the model just did."""
    if result["is_empty"]:
        return (
            "There were no usable words left after cleaning, so the model has "
            "nothing to judge. Try typing an actual sentence."
        )
    if result["known_words"] == 0:
        return (
            "The model did not recognise **any** of these words from its "
            "training vocabulary, so this prediction is basically a guess. "
            "This is the classic out-of-vocabulary problem."
        )

    confidence = result["confidence"]
    if confidence >= 0.90:
        strength = "very confident"
    elif confidence >= 0.70:
        strength = "fairly confident"
    else:
        strength = "not very confident -- this review sits close to the boundary"

    return (
        f"The model is **{strength}** ({confidence:.0%}). It scored the words "
        f"it recognised and the total leaned "
        f"**{result['sentiment'].lower()}**. Remember this is a probability, "
        "not proof -- the model can be confidently wrong, especially on "
        "sarcasm."
    )


# ===========================================================================
# Header
# ===========================================================================
metrics = get_metrics()

st.title("💬 Customer Review Sentiment Analysis")
if metrics:
    st.caption(
        f"TF-IDF + {metrics['primary_model']}, trained on "
        f"{metrics['dataset']['usable_rows']:,} labelled review sentences "
        "from Amazon, Yelp and IMDb."
    )
else:
    st.caption("TF-IDF + Logistic Regression on labelled customer reviews.")

# Fail early and clearly if the model has not been trained yet.
try:
    model, vectorizer = get_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.info(
        "Open a terminal in the project folder and run:\n\n"
        "```\npython -m src.download_data\npython -m src.train\n```"
    )
    st.stop()

with st.sidebar:
    st.header("About this model")
    st.markdown(
        f"""
**Pipeline**
1. Clean the text (lowercase, strip URLs/HTML/punctuation)
2. TF-IDF -> {len(vectorizer.vocabulary_):,} features (unigrams + bigrams)
3. Logistic Regression -> probability
4. Probability > 0.5 -> **Positive**
        """
    )

    if metrics:
        scores = metrics["results"][metrics["primary_model"]]
        st.markdown(
            f"""
**Measured test-set performance**

| Metric | Value |
| --- | --- |
| Accuracy | **{scores['accuracy']:.1%}** |
| F1 (macro) | {scores['f1_macro']:.3f} |
| ROC-AUC | {scores['roc_auc']:.3f} |

Measured on {metrics['dataset']['test_rows']} unseen reviews.
Numbers are read from `reports/metrics.json`, written by
`python -m src.train` -- nothing here is typed in by hand.
            """
        )
    else:
        st.caption(
            "No reports/metrics.json found, so no scores to show. "
            "Run `python -m src.train` to generate it."
        )

    st.divider()
    st.caption(
        "Only two sentiments are supported: Positive and Negative. "
        "The training data has no neutral class, so a neutral review will "
        "always be forced into one of the two."
    )

tab_single, tab_batch = st.tabs(["Single review", "Batch CSV"])

# ===========================================================================
# Tab 1 -- one review at a time
# ===========================================================================
with tab_single:
    EXAMPLES = {
        "-- type my own --": "",
        "Positive example": "The battery lasts all day and the screen is beautiful.",
        "Negative example": "Arrived scratched and stopped charging after a week.",
        "Negation example": "The design is not good and the sound is not great either.",
        "Hard / borderline example": "It works, I suppose, but I expected more for the price.",
    }

    choice = st.selectbox("Try an example, or write your own:", list(EXAMPLES))
    review_text = st.text_area(
        "Customer review",
        value=EXAMPLES[choice],
        height=130,
        placeholder="e.g. The delivery was fast but the product feels cheap.",
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not review_text.strip():
            st.warning("Please enter a review first.")
        else:
            result = predict_sentiment(review_text)

            if result["is_empty"]:
                st.warning(confidence_note(result))
            else:
                # --- headline verdict -------------------------------------
                if result["label"] == 1:
                    st.success(f"### ✅ {result['sentiment']}")
                else:
                    st.error(f"### ❌ {result['sentiment']}")

                left, right = st.columns(2)
                left.metric("Confidence", f"{result['confidence']:.1%}")
                right.metric("Words recognised", result["known_words"])

                # --- probabilities ----------------------------------------
                st.write("**Probability breakdown**")
                st.progress(
                    result["probability_positive"],
                    text=f"Positive: {result['probability_positive']:.1%}",
                )
                st.progress(
                    result["probability_negative"],
                    text=f"Negative: {result['probability_negative']:.1%}",
                )

                # --- explanation ------------------------------------------
                st.info(confidence_note(result))

                drivers = explain_prediction(review_text, top_n=6)
                if drivers:
                    st.write("**Which words drove this prediction?**")
                    driver_frame = pd.DataFrame(
                        [
                            {
                                "term": term,
                                "contribution": round(contribution, 4),
                                "pushes towards": (
                                    "Positive" if contribution > 0 else "Negative"
                                ),
                            }
                            for term, contribution in drivers
                        ]
                    )
                    st.dataframe(driver_frame, hide_index=True,
                                 width="stretch")
                    st.caption(
                        "Contribution = the term's TF-IDF value x the weight "
                        "Logistic Regression learned for it. Adding all of "
                        "them up (plus the intercept) is exactly how the "
                        "model reached its answer."
                    )

                with st.expander("What the model actually saw after cleaning"):
                    st.code(result["cleaned_text"] or "(nothing left)")

# ===========================================================================
# Tab 2 -- many reviews from a CSV
# ===========================================================================
with tab_batch:
    st.write(
        "Upload a CSV with a text column. Every row is scored and you can "
        "download the results."
    )
    st.caption(
        "There is a ready-made file to try in this repo: "
        "`data/sample_reviews.csv`."
    )

    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
        except Exception as error:                      # noqa: BLE001
            st.error(f"Could not read that CSV: {error}")
            st.stop()

        if frame.empty:
            st.warning("That CSV has no rows.")
            st.stop()

        # Let the user say which column holds the text, defaulting to
        # 'review' if it exists.
        columns = list(frame.columns)
        default_index = (
            columns.index(config.TEXT_COLUMN)
            if config.TEXT_COLUMN in columns
            else 0
        )
        text_column = st.selectbox(
            "Which column holds the review text?", columns, index=default_index
        )

        if st.button("Analyze all rows", type="primary"):
            results = predict_batch(frame[text_column].astype(str).tolist())

            output = frame.copy()
            output["predicted_sentiment"] = [r["sentiment"] for r in results]
            output["confidence"] = [round(r["confidence"], 4) for r in results]
            output["probability_positive"] = [
                round(r["probability_positive"], 4) for r in results
            ]

            positive_count = sum(1 for r in results if r["label"] == 1)
            negative_count = sum(1 for r in results if r["label"] == 0)
            unknown_count = sum(1 for r in results if r["label"] is None)

            st.success(f"Scored {len(results)} rows.")
            columns_row = st.columns(3)
            columns_row[0].metric("Positive", positive_count)
            columns_row[1].metric("Negative", negative_count)
            columns_row[2].metric("Unreadable", unknown_count)

            st.dataframe(output, width="stretch")

            # If the CSV already has true labels, show the real accuracy.
            # This is honest evaluation, not a made-up number.
            if config.LABEL_COLUMN in frame.columns:
                from src.preprocessing import normalize_label

                # reset_index so the true labels line up with our results by
                # position. Without this, a CSV whose index is not 0..n-1
                # would compare the wrong rows against each other.
                truth = (
                    frame[config.LABEL_COLUMN]
                    .map(normalize_label)
                    .reset_index(drop=True)
                )
                predicted = pd.Series([r["label"] for r in results])
                comparable = truth.notna() & predicted.notna()

                if comparable.any():
                    correct = int(
                        (truth[comparable] == predicted[comparable]).sum()
                    )
                    total = int(comparable.sum())
                    st.info(
                        f"This file has a `{config.LABEL_COLUMN}` column, so we "
                        f"can check: **{correct}/{total} correct "
                        f"({correct / total:.1%})** on these rows."
                    )

            buffer = io.StringIO()
            output.to_csv(buffer, index=False)
            st.download_button(
                "Download results as CSV",
                data=buffer.getvalue(),
                file_name="sentiment_predictions.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "Portfolio project. Classical ML only (TF-IDF + Logistic Regression), "
    "no deep learning. Trained on short English review sentences, so accuracy "
    "on long reviews or other domains will be lower."
)
