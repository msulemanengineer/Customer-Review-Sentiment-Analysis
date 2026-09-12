"""
Train the sentiment model end to end.

Run with:
    python -m src.train

What this script does, in order:
    1. load + clean the dataset             (src/preprocessing.py)
    2. stratified 80/20 train/test split
    3. fit TF-IDF on the TRAINING text only  (src/features.py)
    4. train Logistic Regression (primary) and Multinomial Naive Bayes
    5. cross-validate both on the training set
    6. score both on the untouched test set  (src/evaluate.py)
    7. save the Logistic Regression model + the vectorizer with joblib
    8. save the measured numbers to reports/metrics.json and a PNG of the
       confusion matrix

Nothing here is random except the split, and that is pinned with
RANDOM_STATE, so running it twice gives identical numbers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB

from src import config, evaluate, features, preprocessing


def build_models() -> dict:
    """The two models we compare.

    LogisticRegression -- our primary model.
        It learns one weight per TF-IDF feature, adds them all up, then
        squashes the total through the sigmoid function to get a probability
        between 0 and 1. That works very well for text because the number of
        features is huge (tens of thousands) while the relationship we need is
        close to linear: "awful" pushes the score down, "delicious" pushes it
        up.
        * solver="liblinear" is built for small/medium sparse problems.
        * max_iter=1000 just gives the optimiser room to converge.
        * C=10.0 is the inverse of the regularisation strength: a SMALL C
          regularises hard (weights forced towards zero, simpler model), a
          LARGE C lets the weights grow. We picked 10 by cross-validating on
          the training set only:
              C=1 -> 0.810, C=2 -> 0.821, C=5 -> 0.831,
              C=10 -> 0.839, C=20 -> 0.842, C=50 -> 0.839, C=100 -> 0.840
          The curve flattens from C=10 onwards and everything above it sits
          inside one standard deviation (+/- 0.006), so we take the smallest
          C on the plateau -- same score, more regularisation, simpler model.

    MultinomialNB -- the baseline we compare against.
        A probabilistic model built on word frequencies. It is extremely fast
        and famously strong on text, but it assumes every word is independent
        of every other word, which is obviously false for language. Comparing
        the two tells us whether the extra flexibility of Logistic Regression
        actually buys anything on this dataset.
    """
    return {
        "Logistic Regression": LogisticRegression(
            solver="liblinear",
            C=10.0,
            max_iter=1000,
            random_state=config.RANDOM_STATE,
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
    }


def main(data_path: Path | None = None) -> dict:
    print("=" * 62)
    print("CUSTOMER REVIEW SENTIMENT ANALYSIS -- TRAINING")
    print("=" * 62)

    # --- 1. data -----------------------------------------------------------
    print("\n[1/7] Loading data")
    df = preprocessing.load_dataset(data_path)

    word_counts = df["clean_review"].str.split().str.len()
    print(f"\nAverage review length: {word_counts.mean():.1f} words "
          f"(shortest {word_counts.min()}, longest {word_counts.max()})")
    if "source" in df.columns:
        print("Rows per source:")
        print(df["source"].value_counts().to_string())

    # --- 2. split ----------------------------------------------------------
    # We split BEFORE touching the vectorizer. The test set has to stay
    # completely unseen, otherwise the score is meaningless.
    # stratify=y keeps the same positive/negative ratio in both halves, so the
    # test set is a fair miniature of the whole dataset.
    print(f"\n[2/7] Splitting {1 - config.TEST_SIZE:.0%} train / "
          f"{config.TEST_SIZE:.0%} test (stratified)")
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        df["clean_review"],
        df[config.LABEL_COLUMN],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[config.LABEL_COLUMN],
    )
    print(f"  train: {len(x_train_text)} reviews")
    print(f"  test : {len(x_test_text)} reviews")

    # --- 3. features -------------------------------------------------------
    print("\n[3/7] Fitting TF-IDF on the training text only")
    vectorizer = features.build_vectorizer()
    x_train = features.fit_transform_train(vectorizer, x_train_text)
    x_test = features.transform_test(vectorizer, x_test_text)

    density = x_train.nnz / (x_train.shape[0] * x_train.shape[1])
    print(f"  vocabulary size : {len(vectorizer.vocabulary_)} terms")
    print(f"  train matrix    : {x_train.shape[0]} x {x_train.shape[1]}")
    print(f"  non-zero cells  : {x_train.nnz} ({density:.2%} of the matrix)")
    print("  -> this is why we store a sparse matrix, not a dense one")

    # --- 4. + 5. train and cross-validate ----------------------------------
    print(f"\n[4/7] Training models and running {config.CV_FOLDS}-fold "
          "cross-validation on the training set")
    folds = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                            random_state=config.RANDOM_STATE)

    models = build_models()
    comparison = {}
    for name, model in models.items():
        # Cross-validation splits the TRAINING data into 5 parts and trains 5
        # times, validating on a different part each time. It gives a more
        # stable estimate than one single split, and it never touches the test
        # set.
        cv_scores = cross_val_score(model, x_train, y_train, cv=folds,
                                    scoring="accuracy")
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1]
        metrics = evaluate.compute_metrics(y_test, y_pred, y_proba)

        # Training accuracy vs test accuracy is our overfitting check:
        # a big gap means the model memorised the training data.
        train_accuracy = float(model.score(x_train, y_train))

        metrics["cv_accuracy_mean"] = float(cv_scores.mean())
        metrics["cv_accuracy_std"] = float(cv_scores.std())
        metrics["train_accuracy"] = train_accuracy
        comparison[name] = metrics

        print(f"\n  {name}")
        print(f"    CV accuracy    : {cv_scores.mean():.4f} "
              f"(+/- {cv_scores.std():.4f})")
        print(f"    train accuracy : {train_accuracy:.4f}")
        print(f"    test accuracy  : {metrics['accuracy']:.4f}")
        print(f"    test F1 (macro): {metrics['f1_macro']:.4f}")

    # --- 6. detailed report for the primary model --------------------------
    primary_name = "Logistic Regression"
    primary_model = models[primary_name]
    y_pred_primary = primary_model.predict(x_test)

    print(f"\n[5/7] Detailed evaluation -- {primary_name} (primary model)")
    evaluate.print_report(primary_name, y_test, y_pred_primary)
    evaluate.plot_confusion_matrix(
        y_test, y_pred_primary,
        title=f"Confusion Matrix -- {primary_name}",
        save_path=config.CONFUSION_MATRIX_PATH,
    )

    top_positive, top_negative = features.top_features_for_class(
        vectorizer, primary_model, n=15)
    print("\nMost POSITIVE terms the model learned:")
    print("  " + ", ".join(f"{term} ({weight:+.2f})"
                           for term, weight in top_positive))
    print("Most NEGATIVE terms the model learned:")
    print("  " + ", ".join(f"{term} ({weight:+.2f})"
                           for term, weight in top_negative))

    # --- 7. save -----------------------------------------------------------
    print("\n[6/7] Saving model artifacts")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # joblib serialises the fitted Python objects to disk ("model
    # serialisation"). The Streamlit app loads these instead of retraining.
    # The vectorizer MUST be saved too: a model whose weight number 4231 means
    # "not good" is useless without the mapping that produced column 4231.
    joblib.dump(primary_model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    print(f"  model      -> {config.MODEL_PATH}")
    print(f"  vectorizer -> {config.VECTORIZER_PATH}")

    print("\n[7/7] Saving metrics")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "file": (data_path or config.DATASET_PATH).name,
            "usable_rows": int(len(df)),
            "train_rows": int(len(x_train_text)),
            "test_rows": int(len(x_test_text)),
            "positive_rows": int((df[config.LABEL_COLUMN] == 1).sum()),
            "negative_rows": int((df[config.LABEL_COLUMN] == 0).sum()),
        },
        "features": {
            "vectorizer": type(vectorizer).__name__,
            "ngram_range": list(vectorizer.ngram_range),
            "min_df": vectorizer.min_df,
            "max_features": vectorizer.max_features,
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
        },
        "hyperparameters": {
            name: {
                key: value
                for key, value in model.get_params().items()
                if key in {"C", "solver", "max_iter", "alpha"}
            }
            for name, model in models.items()
        },
        "primary_model": primary_name,
        "results": comparison,
        "top_positive_terms": top_positive,
        "top_negative_terms": top_negative,
        "versions": {
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    evaluate.save_metrics(payload)

    best = max(comparison, key=lambda name: comparison[name]["f1_macro"])
    print("\n" + "=" * 62)
    print(f"Done. Deployed model: {primary_name} "
          f"(test accuracy {comparison[primary_name]['accuracy']:.4f})")
    print(f"Highest test F1 (macro) in this run: {best}")
    print("=" * 62)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the sentiment model.")
    parser.add_argument(
        "--data", dest="data_path", default=None,
        help="Path to a CSV with 'review' and 'label' columns "
             "(default: data/reviews.csv)",
    )
    args = parser.parse_args()
    main(Path(args.data_path) if args.data_path else None)
