"""Phase 04 — TF-IDF support-message classifier with a held-out test set."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_PATH = Path(__file__).resolve().parent / "data" / "support_messages.csv"


def load_dataset(path: Path):
    frame = pd.read_csv(path)
    return frame["message"], frame["label"]


def train_classifier(path: Path):
    texts, labels = load_dataset(path)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LinearSVC": LinearSVC(),
        "MultinomialNB": MultinomialNB(),
    }
    # The handout ends on logistic regression, so that model answers new messages.
    chosen = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    chosen.fit(train_texts, train_labels)
    predictions = chosen.predict(test_texts)
    print("Chosen model: LogisticRegression")
    print("Accuracy:", accuracy_score(test_labels, predictions))
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, zero_division=0))

    print("Same split, other classifiers from the handout:")
    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)
    for name, model in candidates.items():
        model.fit(train_features, train_labels)
        score = accuracy_score(test_labels, model.predict(test_features))
        print(f"  {name}: {score:.2f}")

    new_text = ["I forgot my login password"]
    print("\nNew text:", new_text[0])
    print("Prediction:", chosen.predict(new_text)[0])
    return chosen


def main():
    print("=== Phase 04 — TF-IDF classification ===")
    print("Dataset:", DATA_PATH.name)
    train_classifier(DATA_PATH)
    print("fit() is training. predict() on a message that was not in the fit call is inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
