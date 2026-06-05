#!/usr/bin/env python3
"""
setup_models.py — Pull required Ollama models

Run this once before using the RAG system:
    python setup_models.py
"""

import ollama
import config

print("🔄 Pulling embedding model:", config.EMBED_MODEL)
ollama.pull(config.EMBED_MODEL)
print("✓ Embedding model ready\n")

print("🔄 Pulling chat model:", config.CHAT_MODEL)
ollama.pull(config.CHAT_MODEL)
print("✓ Chat model ready\n")

print("✅ All models ready! You can now run:")
print("   python ingest.py    (to ingest documents)")
print("   python api.py        (to start the API)")
