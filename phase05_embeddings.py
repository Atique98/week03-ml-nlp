"""Phase 05 — sentence embeddings and cosine similarity."""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print("=== Phase 05 — embeddings and semantic similarity ===")
    model = SentenceTransformer(MODEL_NAME)
    sentences = [
        "How do I reset my password?",
        "I forgot my password.",
        "The weather is very hot today.",
    ]
    embeddings = model.encode(sentences)
    print("Sentences:", len(sentences))
    print("Embedding shape:", embeddings.shape)

    password_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    weather_similarity = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
    print("Password similarity:", round(float(password_similarity), 4))
    print("Weather similarity:", round(float(weather_similarity), 4))
    print("A score near 1 means the sentences are about the same thing.")
    print("The password sentences match even though the wording is not identical.")
    print("The weather sentence stays near 0 because the meaning is different.")
    print("Word counts only match shared words. Embeddings compare meaning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
