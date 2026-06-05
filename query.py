#!/usr/bin/env python3
"""
query.py — Ask a question against your document corpus.

Usage:
    python query.py "how does X work?"

What happens:
  1. Embed the question using nomic-embed-text via Ollama
  2. Find the top-k most similar chunks in Chroma (cosine similarity)
  3. Pass retrieved chunks as context to llama3.2
  4. Print the answer + which source chunks were used
"""

import sys
from pathlib import Path

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import config

# Prompt template — makes the LLM stay grounded in retrieved context
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, say so — do not make things up.

Context:
{context}

Question: {question}

Answer:""",
)


def build_qa_chain():
    chroma_path = Path(config.CHROMA_DIR)
    if not chroma_path.exists():
        print("ERROR: No Chroma database found. Run 'python ingest.py' first.")
        sys.exit(1)

    embeddings = OllamaEmbeddings(
        model=config.EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    vectorstore = Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.TOP_K},
    )
    llm = Ollama(
        model=config.CHAT_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )
    return chain


def run_query(question: str, verbose: bool = True) -> dict:
    chain = build_qa_chain()
    result = chain.invoke({"query": question})

    if verbose:
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(result["result"])

        print("\n" + "=" * 60)
        print(f"SOURCES  (top {config.TOP_K} chunks retrieved)")
        print("=" * 60)
        for i, doc in enumerate(result["source_documents"], 1):
            source = doc.metadata.get("source", "unknown")
            preview = doc.page_content[:300].replace("\n", " ")
            print(f"\n[{i}] {source}")
            print(f"    {preview}...")

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question here\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"\nQuestion: {question}")
    run_query(question)


if __name__ == "__main__":
    main()
