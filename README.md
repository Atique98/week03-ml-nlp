# Week 03 — Machine Learning, NLP and LLM Foundations

Separate submission from the Week 02 API lab. This folder covers text classification, preprocessing, tokenization, embeddings, and semantic search.

## Setup

Use Python 3.10 or newer.

```bash
python3 -m pip install -r requirements.txt
python3 run_lab.py
```

The first run downloads `bert-base-uncased`, `gpt2`, and `sentence-transformers/all-MiniLM-L6-v2`. One phase at a time:

```bash
python3 phase01_text_classification.py
python3 phase02_preprocessing.py
python3 phase03_tokenization.py
python3 phase04_tfidf_classifier.py
python3 phase05_embeddings.py
python3 phase06_semantic_search.py
```

Saved output is in `evidence/full_run.txt`.

## What each phase does

| Phase | File | Handout idea |
| --- | --- | --- |
| 01 | `phase01_text_classification.py` | CountVectorizer and Multinomial Naive Bayes |
| 02 | `phase02_preprocessing.py` | Strip, lowercase, and remove punctuation |
| 03 | `phase03_tokenization.py` | BERT, GPT-2, character pieces, and SentencePiece |
| 04 | `phase04_tfidf_classifier.py` | TF-IDF classifier on `data/support_messages.csv` |
| 05 | `phase05_embeddings.py` | Cosine similarity of sentence embeddings |
| 06 | `phase06_semantic_search.py` | Rank documents for three queries |

## Explanations

**Training and inference.** Training is `fit`. The vectorizer learns its vocabulary and the classifier learns from the labels. Inference is `predict` on text that was not used to fit the model. New text is converted with `transform`, so the vocabulary stays the one learned during training.

**Preprocessing.** The same word should not become several vocabulary entries because of capitals, extra spaces, or punctuation. Cleaning makes those variants match before counting or classification.

**Special tokens and subwords.** BERT adds `[CLS]` at the start of a sequence and `[SEP]` at the end. GPT-2 uses a different tokenization and does not add those two markers. Subword tokenizers split an uncommon word into smaller known pieces. BERT splits `tokenization` into `token` and `##ization` instead of replacing the whole word with an unknown token.

**Word counts and embeddings.** A count vector only records which words appear. Two sentences about passwords can score as unrelated if they do not share words. An embedding places sentences with similar meaning near each other. Cosine similarity near 1 means a close meaning. A score near 0 means the sentences are about different things.
