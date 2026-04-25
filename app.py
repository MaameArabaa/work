import streamlit as st
import os

from data_loader import load_pdf, load_csv, csv_to_text_chunks
from chunking import chunk_text
from embedding import embed_chunks
from retriever import HybridVectorStore, retrieve
from prompt import build_prompt
from generator import generate_response

st.set_page_config(
    page_title="Academic City RAG Assistant",
    layout="wide"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b1324;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.3rem;
        }

        h1, h2, h3, label, p, div {
            color: #ffffff;
        }

        /* Hide default Streamlit header spacing */
        header {
            visibility: hidden;
        }

        .hero-card {
            background: #24344a;
            border: 1px solid #40536d;
            border-radius: 0px 0px 24px 24px;
            padding: 34px 36px;
            margin-bottom: 30px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        }

        .hero-title {
            font-size: 36px;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 18px;
        }

        .hero-subtitle {
            font-size: 18px;
            color: #d8e3f3;
        }

        .metric-card {
            background: #1d2a3d;
            border: 1px solid #33465f;
            border-radius: 18px;
            padding: 28px 20px;
            text-align: center;
            box-shadow: 0 8px 18px rgba(0,0,0,0.18);
        }

        .metric-label {
            font-size: 16px;
            color: #93b5d8;
            margin-bottom: 10px;
        }

        .metric-value {
            font-size: 28px;
            font-weight: 800;
            color: #ffffff;
        }

        div[data-testid="stTextInput"] label {
            color: #ffffff !important;
            font-weight: 700;
            font-size: 16px;
        }

        div[data-testid="stTextInput"] input {
            background-color: #1d2a3d !important;
            color: #ffffff !important;
            border: 1px solid #52657f !important;
            border-radius: 14px !important;
            padding: 14px 18px !important;
            font-size: 18px !important;
        }

        div[data-testid="stTextInput"] input::placeholder {
            color: #8fb0d0 !important;
        }

        .answer-box {
            padding: 18px;
            border-radius: 14px;
            background-color: #f0f2f6;
            color: #000000;
            font-size: 16px;
            font-weight: 500;
            border: 1px solid #d3d3d3;
            margin-bottom: 20px;
        }

        div[data-testid="stExpander"] {
            background-color: #1d2a3d;
            border: 1px solid #33465f;
            border-radius: 12px;
        }

        textarea {
            background-color: #1d2a3d !important;
            color: white !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #111827;
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
st.sidebar.title("Settings")
k = st.sidebar.slider("Top-k", 1, 5, 3)
alpha = st.sidebar.slider("Alpha", 0.0, 1.0, 0.7)

# ----------------------------
# HELPER
# ----------------------------
def find_file(possible_names):
    for name in possible_names:
        if os.path.exists(name):
            return name
    return None

# ----------------------------
# BUILD SYSTEM
# ----------------------------
@st.cache_resource
def build_system():
    pdf_path = find_file(["budget.pdf", "Budget.pdf"])
    csv_path = find_file([
        "ghana_election_result.csv",
        "Ghana_Election_Result.csv",
        "election_results.csv"
    ])

    if not pdf_path:
        st.error("❌ budget.pdf not found in repository")
        st.stop()

    if not csv_path:
        st.error("❌ CSV file not found. Check filename in GitHub.")
        st.stop()

    pdf_text = load_pdf(pdf_path)
    pdf_chunks = chunk_text(pdf_text, chunk_size=500, overlap=100)

    csv_df = load_csv(csv_path)
    csv_chunks = csv_to_text_chunks(csv_df)

    all_chunks = pdf_chunks + csv_chunks
    embeddings = embed_chunks(all_chunks)

    store = HybridVectorStore()
    store.add(embeddings, all_chunks)

    return store, len(pdf_chunks), len(csv_chunks), len(all_chunks)

store, pdf_count, csv_count, total_count = build_system()

# ----------------------------
# HEADER CARD
# ----------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Academic City RAG Assistant</div>
        <div class="hero-subtitle">
            Ask grounded questions using only the budget PDF and election CSV dataset.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# METRIC CARDS
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

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------
# USER INPUT
# ----------------------------
query = st.text_input(
    "Ask a question",
    placeholder="What does the budget say about GETFund?"
)

if query and len(query.strip()) < 10:
    st.warning("Please enter a complete question.")
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

    st.subheader("Retrieved Chunks")
    for i, chunk in enumerate(results):
        with st.expander(f"Chunk {i+1}"):
            st.write(chunk)
            st.write(f"Score: {scores[i]:.4f}")

    st.subheader("Prompt Sent to Model")
    st.text_area("", prompt, height=200)