# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique that improves LLM responses by providing
relevant context retrieved from a document corpus at query time, rather than relying solely on
the model's parametric knowledge baked in during training.

## Why RAG?

LLMs have a knowledge cutoff. They also hallucinate — generating plausible-sounding but
incorrect facts. RAG addresses both problems by grounding responses in actual documents.

The key insight: instead of asking a model "what do you know about X?", RAG asks
"given these specific passages, what is the answer to X?"

## The RAG pipeline

1. **Ingestion** (offline): Load documents → split into chunks → embed each chunk → store in a vector DB.
2. **Retrieval** (online): Embed the user query → find the nearest chunks by cosine similarity.
3. **Generation** (online): Pass retrieved chunks as context in the LLM prompt → generate grounded answer.

## Chunking strategies

Documents are split into overlapping chunks before embedding. Common strategies:

- **Fixed-size**: Split every N tokens. Simple but may cut across sentences.
- **Recursive character splitting**: Split on paragraph breaks, then sentences, then words. Preserves structure better.
- **Semantic chunking**: Split at topic boundaries detected by embedding similarity. More expensive but higher quality.

Chunk size is a key hyperparameter. Smaller chunks (200-300 tokens) give more precise retrieval
but may lack context. Larger chunks (800-1000 tokens) give more context but dilute the signal.
Overlap (typically 10-20% of chunk size) prevents losing context at boundaries.

## Embedding models

An embedding model maps text to a dense vector in a high-dimensional space. Semantically similar
text ends up close together (measured by cosine similarity or dot product).

For local use, `nomic-embed-text` (via Ollama) is a strong, compact choice. It produces
768-dimensional vectors and runs entirely on CPU if needed.

The embedding model used at ingestion time MUST be the same model used at query time.
Mixing models produces meaningless similarity scores.

## Vector databases

A vector database stores embeddings alongside the original text and supports approximate
nearest-neighbor (ANN) search at scale. Options:

- **Chroma**: Embedded, SQLite-backed, zero infrastructure. Good for development.
- **pgvector**: Postgres extension. Good for production if you already run Postgres.
- **Qdrant / Weaviate / Pinecone**: Purpose-built, scalable, more operational overhead.

For most learning projects and small corpora (< 100k chunks), Chroma is sufficient.
