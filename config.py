# config.py — central configuration for rag-demo
# Change values here; everything else imports from this file.

# Ollama endpoint — change if Ollama is on another machine (e.g. Tailscale IP)
OLLAMA_BASE_URL = "http://localhost:11434"

# Models
EMBED_MODEL = "nomic-embed-text"  # embedding only, not for chat
CHAT_MODEL  = "llama3.2"          # generation model

# Chunking
CHUNK_SIZE    = 500   # approximate token count per chunk
CHUNK_OVERLAP = 50    # overlap between adjacent chunks (prevents context loss at boundaries)

# Retrieval
TOP_K = 4  # number of chunks to retrieve per query

# Paths
CHROMA_DIR = "./chroma_db"  # Chroma persists to disk here (SQLite-backed)
DOCS_DIR   = "./docs"       # directory to load documents from
