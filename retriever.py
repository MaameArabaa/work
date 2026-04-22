from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


class HybridVectorStore:
    def __init__(self):
        self.embeddings = None
        self.texts = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

    def add(self, embeddings, texts):
        self.embeddings = np.array(embeddings)
        self.texts = texts
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query, k=3, alpha=0.7):
        """
        alpha controls balance:
        alpha = 0.7 means:
        70% semantic similarity
        30% keyword similarity
        """

        # Semantic similarity
        query_embedding = model.encode([query])
        semantic_scores = cosine_similarity(query_embedding, self.embeddings)[0]

        # Keyword similarity
        query_tfidf = self.vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]

        # Combine scores
        combined_scores = alpha * semantic_scores + (1 - alpha) * keyword_scores

        # Get top-k results
        top_k_idx = np.argsort(combined_scores)[-k:][::-1]

        results = [self.texts[i] for i in top_k_idx]
        scores = [combined_scores[i] for i in top_k_idx]
        semantic_top = [semantic_scores[i] for i in top_k_idx]
        keyword_top = [keyword_scores[i] for i in top_k_idx]

        return results, scores, semantic_top, keyword_top


def retrieve(query, store, k=3, alpha=0.7):
    return store.search(query, k=k, alpha=alpha)