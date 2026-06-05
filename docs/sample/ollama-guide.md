# Ollama

Ollama is a tool for running large language models locally. It handles model download,
quantization, and serving, exposing an OpenAI-compatible REST API on port 11434 by default.

## Key commands

```bash
# Pull a model (from command line)
ollama pull llama3.2
ollama pull nomic-embed-text
```

Or pull models from Python (useful if the `ollama` CLI isn't in your PATH):

```bash
# From the repo root, run:
python setup_models.py
```

Continuing with other commands:

```bash
# List downloaded models
ollama list

# Run interactively in the terminal
ollama run llama3.2

# Show model info
ollama show llama3.2
```

## API

Ollama exposes two endpoints relevant to RAG:

### Chat completions (OpenAI-compatible)
POST http://localhost:11434/v1/chat/completions

### Embeddings
POST http://localhost:11434/api/embeddings
Body: { "model": "nomic-embed-text", "prompt": "text to embed" }

The embeddings endpoint returns a single vector. LangChain's OllamaEmbeddings class
wraps this endpoint and handles batching.

## Models

- **llama3.2** (3B): Fast, low memory, good for general chat and RAG generation.
- **llama3.2:1b**: Even smaller. Useful on constrained hardware.
- **nomic-embed-text**: Embedding-only model. Produces 768-dim vectors. Does not generate text.
- **mxbai-embed-large**: Higher quality embeddings, larger model.

## Running on a remote machine

If Ollama runs on a different host (e.g. over Tailscale), set the base URL:

```python
OllamaEmbeddings(model="nomic-embed-text", base_url="http://hostname:11434")
Ollama(model="llama3.2", base_url="http://hostname:11434")
```

By default Ollama binds to 127.0.0.1. To expose it on all interfaces:
OLLAMA_HOST=0.0.0.0 ollama serve
