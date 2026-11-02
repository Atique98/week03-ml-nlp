"""Run the Week 03 lab from phase 01 through phase 06."""

import phase01_text_classification
import phase02_preprocessing
import phase03_tokenization
import phase04_tfidf_classifier
import phase05_embeddings
import phase06_semantic_search


def main():
    phases = [
        phase01_text_classification.main,
        phase02_preprocessing.main,
        phase03_tokenization.main,
        phase04_tfidf_classifier.main,
        phase05_embeddings.main,
        phase06_semantic_search.main,
    ]
    for run_phase in phases:
        result = run_phase()
        if result not in (None, 0):
            return result
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
