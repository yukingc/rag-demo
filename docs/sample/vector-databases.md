# Vector Databases

A vector database stores high-dimensional vectors (embeddings) and enables fast
approximate nearest-neighbor (ANN) search. This is the retrieval backbone of any RAG system.

## How similarity search works

When you search a vector DB:
1. Your query is embedded into a vector (same model used at ingestion).
2. The DB computes distance between the query vector and all stored vectors.
3. The top-k closest vectors (by cosine similarity or L2 distance) are returned.
4. The original text chunks associated with those vectors are retrieved.

Cosine similarity measures the angle between two vectors, not their magnitude.
Two texts can be very different lengths but semantically similar, and cosine similarity
handles this well.

## Chroma

Chroma is an open-source, embedded vector database. Key properties:

- **No separate process**: runs in-process with your Python app.
- **Persists to disk**: backed by SQLite + a custom index format.
- **Simple API**: designed to be the easiest possible vector DB to get started with.

```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("my_docs")
collection.add(documents=["chunk text"], embeddings=[[0.1, 0.2, ...]], ids=["doc1"])
results = collection.query(query_embeddings=[[0.1, 0.2, ...]], n_results=4)
```

LangChain's `Chroma` class wraps this and handles embedding automatically.

## pgvector

pgvector is a Postgres extension that adds a `vector` column type and cosine/L2/dot-product
index operators. Best choice if you already run Postgres in production.

```sql
CREATE EXTENSION vector;
CREATE TABLE documents (id serial, content text, embedding vector(768));
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
SELECT content FROM documents ORDER BY embedding <=> '[...]' LIMIT 4;
```

The `<=>` operator is cosine distance. Smaller = more similar.

## Choosing between them

| | Chroma | pgvector |
|---|---|---|
| Setup | Zero (in-process) | Needs Postgres |
| Scale | Up to ~1M vectors comfortably | Millions+ |
| Ops burden | None | Postgres maintenance |
| Best for | Dev, local, learning | Production |

For this project, Chroma is the right choice. Swapping to pgvector later is a
one-line change in config.py.
