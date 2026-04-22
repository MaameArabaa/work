from data_loader import load_pdf, load_csv, csv_to_text_chunks
from chunking import chunk_text
from embedding import embed_chunks
from retriever import HybridVectorStore, retrieve
from prompt import build_prompt
from generator import generate_response


def generate_pure_llm_response(query):
    prompt = f"""
Answer the question.

Question:
{query}

Answer:
"""
    return generate_response(prompt)


def build_rag_system():
    pdf_text = load_pdf("budget.pdf")
    pdf_chunks = chunk_text(pdf_text, chunk_size=500, overlap=100)

    csv_df = load_csv("ghana_election_result.csv")
    csv_chunks = csv_to_text_chunks(csv_df)

    all_chunks = pdf_chunks + csv_chunks
    embeddings = embed_chunks(all_chunks)

    store = HybridVectorStore()
    store.add(embeddings, all_chunks)

    return store


def evaluate(store, queries):
    for q in queries:
        print("\nQuery:", q)

        results, _, _, _ = retrieve(q, store, k=3, alpha=0.7)

        rag_prompt = build_prompt(q, results)
        rag_answer = generate_response(rag_prompt, results, q)

        pure_answer = generate_pure_llm_response(q)

        print("\nRAG:", rag_answer)
        print("\nPure LLM:", pure_answer)
        print("=" * 60)


if __name__ == "__main__":
    queries = [
        "What does the budget say about GETFund?",
        "Which region had the highest vote count?",
        "What was the purpose of the Daakye Trust Programme?"
    ]

    store = build_rag_system()
    evaluate(store, queries)