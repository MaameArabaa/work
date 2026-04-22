import streamlit as st
from data_loader import load_pdf, load_csv, csv_to_text_chunks
from chunking import chunk_text
from embedding import embed_chunks
from retriever import HybridVectorStore, retrieve
from prompt import build_prompt
from generator import generate_response

st.set_page_config(
    page_title="Academic City RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------
# CUSTOM STYLING
# ----------------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0f172a;
        }

        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, h4, h5, h6, p, label, div {
            color: white;
        }

        .hero-box {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            padding: 28px;
            border-radius: 20px;
            border: 1px solid #475569;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 8px;
            color: white;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: #cbd5e1;
        }

        .metric-card {
            background-color: #1e293b;
            padding: 18px;
            border-radius: 16px;
            border: 1px solid #334155;
            text-align: center;
            margin-bottom: 10px;
        }

        .metric-label {
            font-size: 0.9rem;
            color: #94a3b8;
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: white;
        }

        div[data-testid="stTextInput"] input {
            background-color: #1e293b !important;
            color: white !important;
            border-radius: 12px !important;
            border: 1px solid #475569 !important;
            padding: 0.9rem !important;
        }

        div[data-testid="stTextInput"] input::placeholder {
            color: #94a3b8 !important;
        }

        .answer-box {
            padding: 18px;
            border-radius: 16px;
            background-color: #1e293b;
            color: white;
            font-size: 16px;
            font-weight: 500;
            border: 1px solid #475569;
            box-shadow: 0 6px 18px rgba(0,0,0,0.2);
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #1e293b;
            color: white;
            border-radius: 12px;
            border: 1px solid #334155;
            padding: 10px 16px;
        }

        .stTabs [aria-selected="true"] {
            background-color: #7c3aed !important;
            color: white !important;
        }

        div[data-testid="stExpander"] {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
        }

        textarea {
            background-color: #1e293b !important;
            color: white !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #334155;
        }

        section[data-testid="stSidebar"] * {
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    k = st.slider("Top-k retrieval", 1, 5, 3)
    alpha = st.slider("Alpha", 0.0, 1.0, 0.7, 0.1)
    st.markdown("---")
    st.caption("This assistant answers only from the budget PDF and election CSV.")

# ----------------------------
# BUILD SYSTEM
# ----------------------------
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

    return store, len(pdf_chunks), len(csv_chunks), len(all_chunks)

store, pdf_count, csv_count, total_count = build_system()

# ----------------------------
# HEADER
# ----------------------------
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">Academic City RAG Assistant</div>
        <div class="hero-subtitle">
            Ask grounded questions using only the budget PDF and election CSV dataset.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# METRICS
# ----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">PDF Chunks</div>
            <div class="metric-value">{pdf_count}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">CSV Chunks</div>
            <div class="metric-value">{csv_count}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Chunks</div>
            <div class="metric-value">{total_count}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# QUERY INPUT
# ----------------------------
query = st.text_input("Ask a question", placeholder="What does the budget say about GETFund?")

if query and len(query.strip()) < 10:
    st.warning("Please enter a more complete question.")
    st.stop()

# ----------------------------
# MAIN LOGIC
# ----------------------------
if query:
    with st.spinner("Generating answer..."):
        results, scores, _, _ = retrieve(query, store, k=k, alpha=alpha)
        prompt = build_prompt(query, results)
        response = generate_response(prompt, results, query)

        if not response or response.strip() == "":
            response = "No relevant information found."

    st.markdown("## 💡 Final Answer")
    st.markdown(
        f"""
        <div class="answer-box">
            {response}
        </div>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs(["Retrieved Chunks", "Prompt", "Scores"])

    with tab1:
        for i, chunk in enumerate(results):
            with st.expander(f"Chunk {i+1}", expanded=(i == 0)):
                st.write(chunk)

    with tab2:
        st.text_area("Prompt sent to model", prompt, height=260)

    with tab3:
        for i, score in enumerate(scores):
            safe_score = max(0.0, min(1.0, float(score)))
            st.write(f"Chunk {i+1}: {score:.4f}")
            st.progress(safe_score)