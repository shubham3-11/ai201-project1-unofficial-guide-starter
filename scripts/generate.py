#!/usr/bin/env python3
"""Milestone 5: grounded generation over retrieved chunks via Groq.

Exports:
    ask(query, k=TOP_K) -> {"answer": str, "sources": list[str], "chunks": list[dict]}

The system prompt enforces retrieval grounding: the LLM is instructed to answer
only from the supplied context and to decline when the context is insufficient.
Source attribution is also appended programmatically after generation so it is
guaranteed regardless of model behavior.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── config ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rutgers_housing"
GENERATION_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHROMA_PATH = _PROJECT_ROOT / "data" / "chroma_db"

SYSTEM_PROMPT = """\
You are an assistant helping prospective and current Rutgers New Brunswick \
students find off-campus housing information.

IMPORTANT RULES:
1. Answer ONLY using the information in the context documents provided.
2. Do NOT draw on your general training knowledge about housing, NJ, or Rutgers.
3. If the provided documents do not contain enough information to answer the \
question, respond with exactly: "I don't have enough information on that."
4. Be specific: quote or closely paraphrase what students or sources actually \
said. Do not generalize beyond what the context states.
5. Keep your answer concise (3–6 sentences) unless the question requires detail.
"""

# ── one-time model and store initialisation ───────────────────────────────────

load_dotenv(_PROJECT_ROOT / ".env")

_embedding_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None
_groq_client: Groq | None = None


def _get_resources() -> tuple[SentenceTransformer, chromadb.Collection, Groq]:
    global _embedding_model, _collection, _groq_client  # noqa: PLW0603

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    if _collection is None:
        if not _CHROMA_PATH.exists():
            print(
                "ERROR: ChromaDB not found. Run scripts/embed_and_retrieve.py first.",
                file=sys.stderr,
            )
            sys.exit(1)
        client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
        _collection = client.get_collection(COLLECTION_NAME)

    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            print("ERROR: GROQ_API_KEY is not set in .env", file=sys.stderr)
            sys.exit(1)
        _groq_client = Groq(api_key=api_key)

    return _embedding_model, _collection, _groq_client


# ── retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    k: int = TOP_K,
) -> list[dict]:
    query_embedding = model.encode([query], convert_to_list=True)[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(
            {
                "text": text,
                "source": meta.get("source", ""),
                "filename": meta.get("filename", ""),
                "url": meta.get("url", ""),
                "distance": round(dist, 4),
            }
        )
    return hits


# ── generation ────────────────────────────────────────────────────────────────

def _build_context_block(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source_label = chunk["source"] or chunk["filename"]
        parts.append(
            f"[Document {i}] Source: {source_label}\n"
            f"URL: {chunk['url']}\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _deduplicate_sources(chunks: list[dict]) -> list[str]:
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        label = chunk["source"] or chunk["filename"]
        if label not in seen:
            seen.add(label)
            url = chunk["url"]
            sources.append(f"{label} — {url}" if url else label)
    return sources


def generate_answer(query: str, chunks: list[dict], groq_client: Groq) -> str:
    context = _build_context_block(chunks)
    user_message = (
        f"Context documents:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer based only on the context above. "
        "If the context is insufficient, say exactly: "
        "\"I don't have enough information on that.\""
    )
    try:
        response = groq_client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return (
            f"[Generation error: {exc}]\n\n"
            "Retrieval succeeded — see the 'Retrieved from' section for source chunks."
        )


# ── public entry point ────────────────────────────────────────────────────────

def ask(query: str, k: int = TOP_K) -> dict:
    """Run the full RAG pipeline for a query.

    Returns:
        {
          "answer": str,           # grounded LLM response
          "sources": list[str],    # deduplicated source labels + URLs
          "chunks": list[dict],    # raw retrieved chunks with distances
        }
    """
    model, collection, groq_client = _get_resources()
    chunks = retrieve(query, collection, model, k=k)
    answer = generate_answer(query, chunks, groq_client)
    sources = _deduplicate_sources(chunks)
    return {"answer": answer, "sources": sources, "chunks": chunks}


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "What areas near Rutgers New Brunswick do students mention as cheaper options for off-campus housing?",
        "What safety-related advice do students give about choosing off-campus housing near Rutgers?",
        "What is the best restaurant in New Brunswick for sushi?",  # out-of-scope
    ]

    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)
        result = ask(query)
        print(f"\nANSWER:\n{result['answer']}")
        print("\nSOURCES:")
        for src in result["sources"]:
            print(f"  • {src}")
        print(f"\n[top chunk distances: {[c['distance'] for c in result['chunks']]}]")
