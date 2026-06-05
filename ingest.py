#!/usr/bin/env python3
"""
ingest.py — Load documents, chunk them, embed with Ollama, store in Chroma.

Run this once (or re-run after adding new documents):
    python ingest.py

What happens:
  1. Load all .txt, .md, .pdf files from DOCS_DIR
  2. Split into overlapping chunks
  3. Embed each chunk using nomic-embed-text via Ollama
  4. Persist vectors + text to Chroma on disk
"""

import sys
import time
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

import config


def load_documents(docs_dir: str) -> list:
    docs = []
    base = Path(docs_dir)

    if not base.exists():
        print(f"ERROR: docs directory '{docs_dir}' does not exist.")
        sys.exit(1)

    # .txt and .md files
    for pattern in ("**/*.txt", "**/*.md"):
        loader = DirectoryLoader(
            docs_dir,
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
        )
        docs.extend(loader.load())

    # .pdf files
    pdf_files = list(base.glob("**/*.pdf"))
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    return docs


def main():
    print(f"→ Loading documents from '{config.DOCS_DIR}' ...")
    docs = load_documents(config.DOCS_DIR)

    if not docs:
        print("No documents found. Add .txt, .md, or .pdf files to ./docs/ and re-run.")
        sys.exit(1)

    print(f"  Loaded {len(docs)} document(s).")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"→ Split into {len(chunks)} chunks "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")

    # Embed and store
    print(f"→ Embedding with '{config.EMBED_MODEL}' via Ollama at {config.OLLAMA_BASE_URL} ...")
    print("  (This may take a minute on first run.)")

    embeddings = OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )

    start = time.time()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DIR,
    )
    elapsed = time.time() - start

    print(f"→ Stored {len(chunks)} chunks in Chroma at '{config.CHROMA_DIR}' ({elapsed:.1f}s).")
    print("  Ready. Run: python query.py \"your question here\"")


if __name__ == "__main__":
    main()
