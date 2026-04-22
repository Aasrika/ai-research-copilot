# 🧠 AI Research Copilot

A multi-agent **Retrieval-Augmented Generation (RAG)** system designed to analyze, interpret, and query research papers with grounded, citation-backed answers.

---

## 🚀 Overview

AI Research Copilot helps users **ask questions over research papers** and receive **factually grounded answers with citations**, reducing hallucinations and improving interpretability.

Built as a modular system with **retrieval, reasoning, and critic agents**, it simulates a research assistant pipeline similar to real-world AI systems.

---

## ✨ Key Features

* 📄 **PDF Parsing & Smart Chunking**

  * Extracts structured sections (methods, results, etc.)
  * Preserves metadata like page number and section

* 🔍 **Semantic Retrieval (FAISS + MMR)**

  * Dense vector search using SentenceTransformers
  * Maximal Marginal Relevance (MMR) for diverse retrieval

* 🤖 **LLM-Based Grounded Answering**

  * Groq-powered inference (LLaMA 3.x)
  * Answers strictly based on retrieved sources

* 📌 **Citation-Based Responses**

  * Inline citations: `[Source N, Page P]`
  * Full transparency of reasoning

* 🔁 **Critic & Retry Mechanism**

  * Evaluates answer quality (score + verdict)
  * Automatically retries with feedback

* 📊 **Evaluation Dashboard**

  * Pass rate, hallucination rate, latency
  * Failure analysis and weak runs tracking

* 🎨 **Interactive Streamlit UI**

  * Clean multi-page interface
  * Query, explore sources, and view metrics

---

## 🧱 System Architecture

```text
User Query
   ↓
Retriever (FAISS + MMR)
   ↓
Answering Agent (LLM + Prompting)
   ↓
Critic Agent (Evaluation + Feedback)
   ↓
Retry Loop (if needed)
   ↓
Final Answer + Citations
```

---

## 🛠️ Tech Stack

* **LLMs:** Groq (LLaMA 3.x)
* **Frameworks:** LangChain, LangGraph
* **Vector DB:** FAISS
* **Embeddings:** SentenceTransformers (MiniLM)
* **Backend:** Python
* **Frontend:** Streamlit
* **Evaluation:** Custom metrics + logging system

---

## 📂 Project Structure

```text
ai-research-copilot/
│
├── backend/
│   ├── agents/          # Answering + Critic agents
│   ├── core/            # Retrieval, config, processing
│   ├── evaluation/      # Metrics + logging
│
├── frontend/
│   ├── app.py           # Main Streamlit app
│   ├── pages/           # Dashboard + additional views
│
├── data/                # Research papers (PDFs)
├── test_graph.py        # Pipeline testing
├── test_ideas.py        # Idea generation agent
```

---

## ⚙️ Setup & Run (No Docker)

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ai-research-copilot.git
cd ai-research-copilot
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set API Key

```bash
set GROQ_API_KEY=your_api_key   # Windows
```

---

## ▶️ Run the Application

### Streamlit UI

```bash
streamlit run frontend/app.py
```

Open: http://localhost:8501

---

### (Optional) Run API

```bash
uvicorn backend.api.main:app --reload --port 8000
```

---

## 🧪 Testing

```bash
python test_graph.py
```

---

## 📊 Evaluation Metrics

The system tracks:

* ✅ Pass Rate
* 📉 Critic Score (0–10)
* ⚠️ Hallucination Rate
* ⏱️ Latency Breakdown
* 🔁 Retry Count

---

## 🎯 Key Highlights (Resume-Ready)

* Built a **multi-agent RAG system** with retrieval, reasoning, and evaluation loops
* Reduced hallucinations using **grounded prompting + critic feedback**
* Designed an **end-to-end pipeline with real-time evaluation dashboard**
* Implemented **semantic search with FAISS + MMR for improved retrieval diversity**
* Developed an **interactive UI for research exploration and analysis**

---

## 🚧 Future Improvements

* Cross-paper reasoning & comparison
* Better section-aware retrieval
* Caching & performance optimization
* Support for larger document collections

---

## 📌 Status

✅ Phase 1: Core pipeline
✅ Phase 2: Grounded RAG system
🔄 Phase 3 (Planned): Advanced reasoning + scaling

---

## 👩‍💻 Author

**Aasrika Kambhampati**
B.Tech Computer Science (AI/ML)

---

## ⭐ If you found this useful

Give this repo a star — it helps visibility!
