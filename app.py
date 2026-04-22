import streamlit as st
from data_loader import load_pdf, load_csv, csv_to_text_chunks
from chunking import chunk_text
from embedding import embed_chunks
from retriever import HybridVectorStore, retrieve
from prompt import build_prompt
from generator import generate_response

# Page setup
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("Academic City RAG Chatbot")

# Sidebar
st.sidebar.title("Settings")
k = st.sidebar.slider("Top-k", 1, 5, 3)
alpha = st.sidebar.slider("Alpha", 0.0, 1.0, 0.7)

# Build system
@st.cache_resource
def build_system():
    pdf_text = load_pdf("budget.pdf")
    pdf_chunks = chunk_text(pdf_text, chunk_size=500, overlap=100)

    csv_df = load_csv("ghana_election_result.csv")
    csv_chunks = csv_to_text_chunks(csv_df)

    all_chunks = pdf_chunks + csv_chunks
    embeddings = embed_chunks(all_chunks)

    store = HybridVectorStore()
    store.add(embeddings, all_chunks)

    return store

store = build_system()

# User input
query = st.text_input("Ask a question")

# Prevent bad input
if query and len(query.strip()) < 10:
    st.warning("Please enter a complete question.")
    st.stop()

# Main logic
if query:
    results, scores, _, _ = retrieve(query, store, k=5, alpha=0.6)

    prompt = build_prompt(query, results)
    response = generate_response(prompt, results, query)

    # 🔥 Ensure response is never empty
    if not response or response.strip() == "":
        response = "I don't know."

    # ✅ FINAL ANSWER BOX (FIXED FOR DARK MODE)
    st.markdown("## 💡 Final Answer")

    st.markdown(
        f"""
        <div style="
            padding:15px;
            border-radius:10px;
            background-color:#f0f2f6;
            color:#000000;
            font-size:16px;
            font-weight:500;
            border:1px solid #d3d3d3;">
            {response}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Retrieved chunks
    st.subheader("Retrieved Chunks")
    for i, chunk in enumerate(results):
        with st.expander(f"Chunk {i+1}"):
            st.write(chunk)
            st.write(f"Score: {scores[i]:.4f}")

    # Prompt display
    st.subheader("Prompt Sent to Model")
    st.text_area("", prompt, height=200)