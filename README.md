# Scheme Research Application

A smart research tool for querying and summarizing Indian government scheme documents (PDFs or URLs) using LLaMA3 via Groq API, LangChain, HuggingFace Embeddings, and FAISS Vector Store. Built with Streamlit.

### 🔗 Live Demo
👉 [scheme-research-app.streamlit.app](https://scheme-research-app.streamlit.app/)

---

##  Features

- 💬 Ask questions from uploaded PDFs or web links
- 📂 View source document snippets
- 🧠 LLaMA3 via Groq backend for high-speed reasoning
- 📊 Generate structured summaries (Benefits, Eligibility, Process, Documents)
- 🧠 Semantic search using HuggingFace + FAISS
- 🧾 Built-in logging and error tracking
- 🎨 Custom UI theme with branding

---

##  Tech Stack

| Component          | Library                     |
|--------------------|-----------------------------|
| UI                 | Streamlit                   |
| Embeddings         | `sentence-transformers`     |
| Vector Store       | FAISS                       |
| PDF Reader         | PyMuPDF (`fitz`)            |
| Language Model     | Groq + LLaMA3 via LangChain |
| URL Parsing        | LangChain URL Loader        |
| Logging            | Python logging              |
| Environment Mgmt   | `python-dotenv`             |


##  Folder Structure
```
scheme-research-app/
│
├── main.py # Main Streamlit app
├── logo.png # App logo
├── faiss_combined.pkl # Pickled vector DB (after processing)
├── requirements.txt # All dependencies
├── .env # Environment variables (not pushed to GitHub)
├── .gitignore # Ignore logs, venv, .env
└── logs/
└── app.log # Runtime logs
```

---

## ⚙️ Setup Instructions (Local)
1. Clone the repository

```
git clone https://github.com/your-username/scheme-research-app.git
cd scheme-research-app
```

2. Create a virtual environment

```
python -m venv venv
```
3. Install dependencies

```
pip install -r requirements.txt
```
4. Create a .env file for your API key

```
GROQ_API_KEY=your_actual_groq_key
```
5. Run the app

```
streamlit run main.py
```

---

# 🚀 Deployment (Streamlit Cloud)
1. Push code to GitHub

2. Go to Streamlit Cloud

3. Create new app → connect GitHub repo

4. Add GROQ_API_KEY in the "Secrets" section

5. Done! Your app will auto-deploy

---
# 🔐 Secrets & Security
1. Store your keys safely:

2. Never push .env to GitHub

3. Use .gitignore to ignore it

Store keys in Streamlit secrets (Settings → Secrets)

---
Developed By K Jaya Sravani
