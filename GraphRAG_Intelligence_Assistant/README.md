# GraphRAG Intelligence Assistant

A local-first learning project that combines:
- Document ingestion (TXT and PDF)
- Text chunking
- TF-IDF retrieval as a lightweight local baseline
- Heuristic entity and relationship extraction
- A NetworkX knowledge graph
- Optional local LLM answer generation through Ollama
- Streamlit UI with evidence and graph-relationship display

## Important prototype limitations

This is an educational MVP, not a production GraphRAG implementation:
- Entity and relationship extraction uses simple heuristics and a small domain vocabulary.
- TF-IDF is lexical retrieval, not dense semantic embeddings.
- PDF page attribution is approximate in this version.
- Graph edges are inferred heuristically and must be verified.
- The app does not implement authentication, multi-user access controls, or production security.

## Requirements

- Windows 10/11
- Python 3.10 or 3.11 recommended
- Optional: Ollama for local LLM generation

## Setup (Windows PowerShell)

```powershell
cd GraphRAG_Intelligence_Assistant
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501), then choose **Load sample documents** in the sidebar.

## Optional local LLM using Ollama

1. Install Ollama from https://ollama.com/
2. In a separate terminal, pull a model that fits your laptop:
   ```powershell
   ollama pull llama3.2:latest
   ```
3. Start/run Ollama, then in the app check **Generate answers with Ollama**.
4. Enter the exact model tag shown by `ollama list`.

If a model is too slow for your available RAM, use a smaller quantized model. The app can still demonstrate retrieval without an LLM.

## Project structure

```text
GraphRAG_Intelligence_Assistant/
├── app.py
├── graph_rag.py
├── requirements.txt
├── README.md
└── data/
    ├── index_store.json       # created when you index/save
    └── sample_docs/
        ├── incident_report.txt
        ├── architecture_notes.txt
        └── troubleshooting_runbook.txt
```

## How the pipeline works

1. Upload PDF/TXT documents or load the included samples.
2. Split document text into overlapping chunks.
3. Identify known entities and some capitalized terms.
4. Create graph nodes and heuristic relationships.
5. Retrieve relevant chunks using TF-IDF.
6. Retrieve graph entities related to the question.
7. Optionally send evidence to a local Ollama model for answer generation.
8. Review the evidence and graph relationships shown by the UI.

## Next upgrades

1. Replace heuristic extraction with structured LLM extraction and validation.
2. Replace TF-IDF with sentence-transformer embeddings and FAISS/ChromaDB.
3. Add Neo4j for persistent graph storage and Cypher queries.
4. Add hybrid retrieval, reranking, and multi-hop graph traversal.
5. Add source-level permission filtering, prompt-injection tests, and audit logs.
6. Add evaluation datasets to measure retrieval quality and groundedness.

## Interview summary

"I built a Graph RAG prototype that ingests technical documents, splits them into chunks, extracts entities and candidate relationships, constructs a NetworkX knowledge graph, retrieves relevant passages using TF-IDF, and optionally uses a local Ollama LLM to generate evidence-grounded answers. I also expose source passages and graph relationships in a Streamlit dashboard. The next production steps are validated entity extraction, dense/hybrid retrieval, Neo4j persistence, evaluation, and access controls."
