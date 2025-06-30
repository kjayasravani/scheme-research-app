import os
import pickle
import logging
from datetime import datetime
import base64
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv

from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq

# ---- Setup Logging ----
os.makedirs("logs", exist_ok=True)
log_filename = "logs/app.log"
logging.basicConfig(
    filename=log_filename,
    filemode="a",
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

def log(message, level="info"):
    if "logs" not in st.session_state:
        st.session_state.logs = []

    st.session_state.logs.append(message)

    if level == "info":
        logging.info(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "error":
        logging.error(message)

# ---- Custom CSS ----
def apply_custom_styles():
    st.markdown("""
    <style>
        .stApp {
            background-color: #0f0f0f;
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }
        .stSidebar {
            background-color: #1a1a1a;
        }
        section.main > div {
            background-color: #1a1a1a;
            padding: 2rem;
            border-radius: 10px;
        }
        h1, h2, h3, h4 {
            color: #FFB6C1;
        }
        .stTextInput > div > div > input, .stTextArea textarea {
            background-color: #262626;
            color: #ffffff;
            border: 1px solid #444;
            border-radius: 6px;
        }
        .stButton > button {
            background-color: #FFB6C1;
            color: #000000;
            border: none;
            padding: 0.5rem 1.2rem;
            border-radius: 6px;
            font-weight: bold;
            transition: 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #e6a9bd;
        }
        .stMarkdown, .stText, .stDataFrame {
            color: #ffffff;
        }
        .css-1d391kg {
            color: white;
        }
        .footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #333;
            color: #888;
            text-align: center;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)

# ---- Environment Variables ----
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---- Apply Theme ----
apply_custom_styles()

# ---- Page Config ----
st.set_page_config(page_title="Scheme Research Application", layout="wide")

# ---- Centered Logo + Title ----
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("logo.png")

st.markdown(
    f"""
    <div style='text-align: center; margin-top: -30px; margin-bottom: 30px;'>
        <img src='data:image/png;base64,{logo_base64}' width='120' style='margin-bottom: 10px;' />
        <h1 style='color: #FFB6C1; font-size: 36px; font-weight: bold; margin: 0;'>Scheme Research Application</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ---- Sidebar ----
st.sidebar.header("Upload PDFs or Enter URLs")
uploaded_pdfs = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

url_mode = st.sidebar.radio("Add URLs", ["None", "Paste URLs", "Upload URL File"])
urls = []
if url_mode == "Paste URLs":
    raw = st.sidebar.text_area("Enter URLs (one per line)")
    urls = [u.strip() for u in raw.splitlines() if u.strip()]
elif url_mode == "Upload URL File":
    url_file = st.sidebar.file_uploader("Upload .txt file with URLs", type="txt")
    if url_file:
        urls = [line.strip() for line in url_file.read().decode("utf-8").splitlines() if line.strip()]

process_btn = st.sidebar.button("Process Data")

# ---- Utils ----
@st.cache_resource
def get_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data
def read_pdf(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        return " ".join([page.get_text() for page in doc])

# ---- Load LLM ----
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

# ---- Document Processing ----
if process_btn:
    docs = []

    for pdf in uploaded_pdfs or []:
        text = read_pdf(pdf)
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": pdf.name}))
    if uploaded_pdfs:
        log(f"✅ {len(uploaded_pdfs)} PDF(s) uploaded.")

    if urls:
        url_docs = UnstructuredURLLoader(urls=urls).load()
        docs.extend(url_docs)
        log(f"🔗 {len(urls)} URL(s) loaded.")

    if not docs:
        st.error("No valid content found in PDFs or URLs.")
        log("❌ No valid content found to process.", level="error")
    else:
        log("⚙️ Starting document processing...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embedder = get_embedder()
        faiss_index = FAISS.from_documents(chunks, embedding=embedder)

        with open("faiss_combined.pkl", "wb") as f:
            pickle.dump((faiss_index, docs), f)

        st.success("Documents processed and indexed.")
        log("📌 Vector index created successfully.")

# ---- Q&A Section ----
st.subheader("Ask a Question")
query = st.text_input("Type your question below:")

if query:
    if not os.path.exists("faiss_combined.pkl"):
        st.warning("Please process the documents first.")
    else:
        with open("faiss_combined.pkl", "rb") as f:
            faiss_index, all_docs = pickle.load(f)

        top_docs = faiss_index.similarity_search(query, k=3)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        answer = chain.run(input_documents=top_docs, question=query)

        st.markdown("#### Answer")
        st.write(answer)
        log(f"❓ User asked: {query}")

        if st.button("Show Source Snippets"):
            st.markdown("##### Source Snippets")
            for i, doc in enumerate(top_docs):
                st.markdown(f"**Doc {i+1}:** `{doc.metadata.get('source', 'Unknown')}`")
                st.write(doc.page_content[:300] + "...")

# ---- Summary Section ----
st.subheader("Generate Structured Summary")
if st.button("Generate Summary"):
    if not os.path.exists("faiss_combined.pkl"):
        st.warning("Please process documents first.")
    else:
        with open("faiss_combined.pkl", "rb") as f:
            faiss_index, all_docs = pickle.load(f)

        chain = load_qa_chain(llm=llm, chain_type="stuff")

        prompts = {
            "🟢 Scheme Benefits": "List the benefits provided under this scheme.",
            "🟡 Application Process": "Explain how one can apply for this scheme.",
            "🔵 Eligibility Criteria": "Mention who is eligible for the scheme.",
            "🟣 Documents Required": "List the documents needed for the scheme.",
        }

        output = []
        for title, prompt in prompts.items():
            result = chain.run(input_documents=all_docs, question=f"{prompt} If not found, say 'Not mentioned'.")
            output.append(f"**{title}**\n\n{result.strip()}\n")

        st.markdown("### Summary")
        for section in output:
            st.markdown(section)

        log("📝 Structured summary generated.")


# ---- Footer ----
st.markdown('<div class="footer">Developed by <strong>K Jaya Sravani</strong> — Scheme Research Application</div>', unsafe_allow_html=True)
