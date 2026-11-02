"""Phase 01 — classify short reviews with word counts and Naive Bayes.

Training fits the vectorizer and the classifier on labeled text.
Inference only transforms new text and asks the trained model for a label.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def lecture_example():
    """The four-sentence example from the Week 03 handout."""
    texts = [
        "I love this course",
        "This course is excellent",
        "I hate this course",
        "This course is terrible",
    ]
    labels = ["positive", "positive", "negative", "negative"]

    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(texts)
    model = MultinomialNB()
    model.fit(features, labels)

    new_text = ["I really love this course"]
    new_features = vectorizer.transform(new_text)
    prediction = model.predict(new_features)
    print("Lecture example")
    print("Prediction:", prediction[0])


def main():
    print("=== Phase 01 — text classification ===")
    lecture_example()
    print("Training is model.fit on labeled text. Inference is model.predict on new text.")
    print("The scored train/test evaluation is in phase 04, where the dataset is large enough to split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
