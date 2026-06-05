# rag-demo

A minimal, end-to-end Retrieval-Augmented Generation (RAG) app using:

- **Ollama** — local LLM inference (`llama3.2`) and embeddings (`nomic-embed-text`)
- **Chroma** — local vector database (SQLite-backed, no separate service)
- **LangChain** — orchestration glue
- **FastAPI** — optional HTTP API wrapper

This is a learning project. The goal is to understand the full RAG stack — chunking, embedding, retrieval, generation — not to build production infrastructure.

---

## Architecture

```
Documents → Chunker → Embedder (nomic-embed-text via Ollama) → Chroma (on disk)
                                                                      ↓
User query → Embedder (nomic-embed-text) → similarity search → top-k chunks
                                                                      ↓
                                                           LLM (llama3.2 via Ollama)
                                                                      ↓
                                                                   Answer
```

Two separate Ollama calls happen per query:
1. Embed the question → vector
2. Generate answer given retrieved chunks as context

The LLM never sees raw documents — only the top-k chunks the retriever selects.

---

## Prerequisites

- [Ollama](https://ollama.com) running locally
- Python 3.11+

---

## Quickstart

```bash
# 1. Clone and set up
git clone https://github.com/youruser/rag-demo.git
cd rag-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Pull the required models (first time only)
python setup_models.py

# 3. Add your documents
# Drop .txt, .md, or .pdf files into ./docs/
# Sample docs are already there to get you started.

# 4. Ingest (chunk + embed + store)
python ingest.py

# 5. Query
python query.py "your question here"

# 6. Optional: run as an HTTP API
uvicorn api:app --reload --port 8080
```

---

## Project structure
setup_models.py    # Run once: pull required Ollama models
├── 
```
rag-demo/
├── ingest.py          # Load docs → chunk → embed → store in Chroma
├── query.py           # CLI: embed question → retrieve → generate answer
├── api.py             # FastAPI wrapper around query.py logic
├── config.py          # Central config (models, paths, chunk sizes)
├── requirements.txt
├── docs/              # Put your documents here (.txt, .md, .pdf)
│   └── sample/        # Sample corpus included for testing
├── scripts/
│   └── reset.sh       # Wipe chroma_db/ and start fresh
└── README.md
```

---

## Configuration

All tunable parameters live in `config.py`:

```python
OLLAMA_BASE_URL   = "http://localhost:11434"
EMBED_MODEL       = "nomic-embed-text"
CHAT_MODEL        = "llama3.2"
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50
TOP_K             = 4
CHROMA_DIR        = "./chroma_db"
DOCS_DIR          = "./docs"
```

If Ollama is running on a different machine (e.g. over Tailscale), change `OLLAMA_BASE_URL`.

---

## Detailed walkthrough

See [WALKTHROUGH.md](WALKTHROUGH.md) for a step-by-step explanation of what each component does and why.

---

## Swapping components

| Component | Default | Alternatives |
|---|---|---|
| Vector DB | Chroma | pgvector, Qdrant, Weaviate |
| Embeddings | nomic-embed-text (Ollama) | OpenAI ada-002, mxbai-embed-large |
| LLM | llama3.2 (Ollama) | Any Ollama model, OpenAI, vLLM endpoint |
| Loader | LangChain DirectoryLoader | Unstructured, custom parsers |

Because everything speaks standard interfaces, swapping any layer is a one-line config change.

---

## Resetting

```bash
bash scripts/reset.sh   # deletes chroma_db/, then re-run ingest.py
```
