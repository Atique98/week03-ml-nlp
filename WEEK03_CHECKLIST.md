# Week 03 Checklist

Completed in this folder. This is a separate submission from the API lab.

## 1. Machine Learning — Text Classification

- [x] Import the required libraries from scikit-learn. `phase01_text_classification.py`, `phase04_tfidf_classifier.py`
- [x] Create a small text dataset with corresponding labels. Phase 01 reviews and `data/support_messages.csv`
- [x] Convert text into numerical features using CountVectorizer or TF-IDF.
- [x] Split the dataset into training and testing sets.
- [x] Train a text classification model.
- [x] Generate predictions for new text.
- [x] Evaluate the model using accuracy and a classification report.
- [x] Explain the difference between training and inference. `README.md`

## 2. Natural Language Processing — Text Preprocessing

- [x] Implement a text-cleaning function in Python. `clean_text` in `phase02_preprocessing.py`
- [x] Remove unnecessary whitespace.
- [x] Convert text to lowercase where appropriate.
- [x] Remove punctuation where appropriate.
- [x] Test the preprocessing function using multiple sentences.
- [x] Explain why text preprocessing is useful. `README.md`

## 3. Tokenization

- [x] Install and import the required tokenizer libraries. `requirements.txt`
- [x] Load a pretrained tokenizer such as BERT's tokenizer.
- [x] Convert a sentence into tokens.
- [x] Convert tokens into token IDs.
- [x] Decode token IDs back into text.
- [x] Compare token counts for two different tokenizers. BERT and GPT-2
- [x] Explain special tokens and subword tokenization. `README.md`

## 4. Embeddings and Semantic Similarity

- [x] Load a sentence-transformer model.
- [x] Generate embeddings for multiple sentences.
- [x] Calculate cosine similarity between two sentences.
- [x] Compare semantically similar and unrelated sentences.
- [x] Interpret the similarity scores.
- [x] Explain the difference between word counts and semantic embeddings. `README.md`

## 5. Semantic Search

- [x] Create a small collection of text documents.
- [x] Generate and store document embeddings.
- [x] Convert a user query into an embedding.
- [x] Calculate similarity between the query and documents.
- [x] Rank documents by similarity score.
- [x] Display the most relevant documents.
- [x] Test the search system using at least three different queries.

## 6. Code Quality and Submission

- [x] Organize the implementation into clearly named sections or functions.
- [x] Use meaningful variable and function names.
- [x] Run the code from beginning to end without unhandled errors. `python3 run_lab.py`
- [x] Add comments where they clarify important logic.
- [x] Capture screenshots or output evidence. `evidence/full_run.txt`
- [x] Prepare a README with setup and execution instructions.
- [x] Submit the source code and required evidence.
