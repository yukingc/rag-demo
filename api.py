#!/usr/bin/env python3
"""
api.py — FastAPI HTTP wrapper around the RAG query logic.

Run:
    uvicorn api:app --reload --port 8080

Endpoints:
    POST /query       — ask a question, get an answer + sources
    GET  /health      — liveness check
    GET  /docs        — auto-generated Swagger UI (FastAPI built-in)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from query import build_qa_chain
import config

app = FastAPI(
    title="rag-demo",
    description="Minimal RAG API: Ollama + Chroma + LangChain",
    version="0.1.0",
)

# Build the chain once at startup (loads Chroma + initialises Ollama client)
qa_chain = None


@app.on_event("startup")
def startup():
    global qa_chain
    qa_chain = build_qa_chain()


class QueryRequest(BaseModel):
    question: str


class SourceDoc(BaseModel):
    source: str
    preview: str  # first 300 chars of the chunk


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDoc]
    model: str
    embed_model: str


@app.get("/health")
def health():
    return {"status": "ok", "chat_model": config.CHAT_MODEL, "embed_model": config.EMBED_MODEL}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    result = qa_chain.invoke({"query": req.question})

    sources = [
        SourceDoc(
            source=doc.metadata.get("source", "unknown"),
            preview=doc.page_content[:300].replace("\n", " "),
        )
        for doc in result["source_documents"]
    ]

    return QueryResponse(
        question=req.question,
        answer=result["result"],
        sources=sources,
        model=config.CHAT_MODEL,
        embed_model=config.EMBED_MODEL,
    )
