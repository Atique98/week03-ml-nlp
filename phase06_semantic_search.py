"""Phase 06 — rank a small document collection by semantic similarity."""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOCUMENTS = [
    "Students can register courses through the student portal.",
    "The examination office publishes examination schedules.",
    "The finance office handles tuition fee issues.",
    "The library provides digital research resources.",
]
QUERIES = [
    "Where can I register my courses?",
    "When is the exam schedule published?",
    "I have a problem paying tuition.",
]


def search(model, document_embeddings, query):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, document_embeddings)[0]
    return sorted(zip(DOCUMENTS, scores), key=lambda item: item[1], reverse=True)


def main():
    print("=== Phase 06 — semantic search ===")
    model = SentenceTransformer(MODEL_NAME)
    document_embeddings = model.encode(DOCUMENTS)
    for query in QUERIES:
        print(f"\nQuery: {query}")
        for document, score in search(model, document_embeddings, query):
            print(f"{score:.4f} -> {document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
