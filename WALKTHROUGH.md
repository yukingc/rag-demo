# WALKTHROUGH.md — How this RAG app works

This document explains what each component does, why decisions were made, and what to
observe as you experiment. Read this alongside the code.

---

## 1. The problem RAG solves

LLMs are frozen at their training cutoff. Ask `llama3.2` about your internal documents,
your homelab setup, or anything specific to your context — it either hallucinates or says
it doesn't know.

RAG solves this by injecting relevant document snippets into the prompt at query time:

> "Given these specific passages from your docs, answer the question."

The model's job becomes reading comprehension, not memory recall.

---

## 2. Ingest phase (`ingest.py`)

### Loading

`DirectoryLoader` recursively finds `.txt`, `.md`, and `.pdf` files under `./docs/`.
Each file becomes one or more `Document` objects with:
- `page_content`: the text
- `metadata`: source file path, page number (for PDFs), etc.

### Chunking

`RecursiveCharacterTextSplitter` splits documents hierarchically:
1. Try splitting on `\n\n` (paragraph breaks)
2. If chunks are still too large, split on `\n`
3. Then on `. ` (sentences)
4. Finally on individual characters

This preserves natural text boundaries better than hard character splitting.

**Key parameters:**
- `chunk_size=500` — target chunk size in characters (not tokens; characters are faster to compute)
- `chunk_overlap=50` — each chunk shares 50 characters with the next, so context isn't lost at boundaries

Try changing `chunk_size` to 200 vs 1000 and re-running to see how retrieval quality changes.

### Embedding

`OllamaEmbeddings(model="nomic-embed-text")` calls Ollama's embedding endpoint for each chunk.
Each chunk becomes a 768-dimensional float vector — a point in high-dimensional space.

Semantically similar text ends up close together. "How do I restart a service?" and
"Restarting services in Docker" will have high cosine similarity even though they share
no words.

**This is the slowest part of ingestion.** For large corpora, consider batching or async embedding.

### Storage

`Chroma.from_documents()` stores both:
- The vectors (for similarity search)
- The original chunk text + metadata (to return with results)

Chroma persists everything to `./chroma_db/` as SQLite files. Delete this directory and
re-run `ingest.py` to start over.

---

## 3. Query phase (`query.py`)

### Embedding the question

The user's question is embedded with the **same model** used at ingestion (`nomic-embed-text`).
This is mandatory — mixing models makes similarity scores meaningless.

### Similarity search

Chroma computes cosine similarity between the query vector and all stored chunk vectors,
returning the top-k closest matches (default: 4).

This is **not** keyword search. The query "how do I reset the database?" will match a chunk
containing "wipe the Chroma directory and re-run ingest" even though no words overlap.

### Generation

The top-k chunks are injected into the prompt:

```
You are a helpful assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, say so — do not make things up.

Context:
[chunk 1 text]
[chunk 2 text]
[chunk 3 text]
[chunk 4 text]

Question: how do I reset the database?

Answer:
```

`llama3.2` generates a response grounded in the provided context.

### The `return_source_documents=True` flag

This makes `query.py` print which chunks were retrieved alongside the answer.
**Always look at the sources.** If the answer is wrong, check:
- Were the right chunks retrieved? (retrieval problem — try different chunk sizes)
- Were the right chunks retrieved but the answer still wrong? (generation problem — try a better prompt or larger model)

These are different failure modes with different fixes.

---

## 4. The API (`api.py`)

FastAPI wraps the query logic as an HTTP service. The chain is built once at startup
(loading Chroma and initialising the Ollama client) and reused for every request.

```bash
uvicorn api:app --reload --port 8080

# Try it
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "what is chunking?"}'

# Swagger UI
open http://localhost:8080/docs
```

The `/health` endpoint is useful for monitoring and as a readiness probe if you
containerise this later.

---

## 5. What to experiment with

### Change chunk size
Edit `CHUNK_SIZE` in `config.py`, run `scripts/reset.sh`, re-run `ingest.py`, then query again.
Smaller chunks: more precise retrieval, less context per chunk.
Larger chunks: more context, but retrieval is less targeted.

### Change TOP_K
Increase `TOP_K` to retrieve more chunks. The LLM gets more context but the prompt gets longer
(and slower, and more expensive if using a paid API).

### Add your own documents
Drop any `.txt`, `.md`, or `.pdf` into `./docs/` and re-run `ingest.py`.

### Point at a remote Ollama
Change `OLLAMA_BASE_URL` in `config.py` to a Tailscale hostname:
```python
OLLAMA_BASE_URL = "http://mac-mini.mulley-bonito.ts.net:11434"
```

### Swap the chat model
Change `CHAT_MODEL` to any model you have pulled in Ollama:
```python
CHAT_MODEL = "mistral"
CHAT_MODEL = "llama3.2:1b"   # faster, smaller
```

### Swap the vector DB to pgvector
1. `pip install langchain-postgres psycopg2-binary`
2. In `ingest.py` and `query.py`, replace `Chroma` with `PGVector`
3. Point at your Postgres instance

Everything else stays the same — same embeddings, same retrieval interface.

---

## 6. Limitations of this demo

- **No re-ranking**: retrieved chunks are used as-is. Production systems often add a
  cross-encoder re-ranker to improve ordering.
- **No streaming**: the API returns the full answer at once. Add `StreamingResponse` in FastAPI
  + `streaming=True` in the Ollama client for streaming.
- **Single collection**: all docs go into one Chroma collection. For multi-tenant or
  multi-corpus use, you'd namespace by collection.
- **No auth**: the FastAPI server has no authentication. Fine for local use, add it before
  exposing publicly.
