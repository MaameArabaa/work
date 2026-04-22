from data_loader import load_pdf, load_csv, csv_to_text_chunks
from chunking import chunk_text
from embedding import embed_chunks
from retriever import HybridVectorStore, retrieve
from prompt import build_prompt
from generator import generate_response
from logger import log_message

print("Loading PDF...")
pdf_text = load_pdf("budget.pdf")

print("Loading CSV...")
csv_df = load_csv("ghana_election_result.csv")
csv_chunks = csv_to_text_chunks(csv_df)

print("Chunking PDF...")
pdf_chunks = chunk_text(pdf_text, chunk_size=500, overlap=100)

# Combine both
all_chunks = pdf_chunks + csv_chunks

print("Embedding...")
embeddings = embed_chunks(all_chunks)

print("Building retriever...")
store = HybridVectorStore()
store.add(embeddings, all_chunks)

print("System ready!\n")

while True:
    query = input("You: ")

    if query.lower() in ["exit", "quit"]:
        break

    log_message("USER QUERY", query)

    results, scores, _, _ = retrieve(query, store, k=3, alpha=0.7)

    print("\nRetrieved Chunks:")
    for i, r in enumerate(results):
        print(f"{i+1}. {r[:200]}...")

    prompt = build_prompt(query, results)

    # 🔥 IMPORTANT UPDATED CALL
    response = generate_response(prompt, results, query)

    log_message("RESPONSE", response)

    print("\nAI:", response)
    print("=" * 50)