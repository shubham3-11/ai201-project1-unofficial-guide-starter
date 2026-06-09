#!/usr/bin/env python3
"""Milestone 4: embed chunks with all-MiniLM-L6-v2 and store in ChromaDB.

Usage:
  python scripts/embed_and_retrieve.py           # build store + run eval queries
  python scripts/embed_and_retrieve.py --rebuild  # drop existing collection and rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rutgers_housing"
TOP_K = 5

EVAL_QUERIES = [
    "What areas near Rutgers New Brunswick do students mention as cheaper options for off-campus housing?",
    "What platforms or resources do Rutgers students recommend for finding off-campus housing?",
    "What safety-related advice do students give about choosing off-campus housing near Rutgers?",
]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_chunks(chunks_path: Path) -> list[dict]:
    chunks = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_vector_store(
    chunks: list[dict],
    model: SentenceTransformer,
    chroma_path: Path,
    rebuild: bool = False,
) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(chroma_path))

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    if existing_count > 0 and not rebuild:
        print(f"Collection '{COLLECTION_NAME}' already has {existing_count} vectors — skipping embed.")
        print("Run with --rebuild to re-embed from scratch.")
        return collection

    print(f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_list=True)

    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [
        {
            "source": chunk.get("source", ""),
            "type": chunk.get("type", ""),
            "url": chunk.get("url", ""),
            "title": chunk.get("title", ""),
            "filename": chunk.get("filename", ""),
            "chunk_index": chunk.get("chunk_index", 0),
        }
        for chunk in chunks
    ]

    # ChromaDB add() handles batching internally; add all at once for simplicity
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Stored {collection.count()} vectors in ChromaDB at {chroma_path}")
    return collection


def retrieve(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    k: int = TOP_K,
) -> list[dict]:
    """Return top-k chunks for a query, each with text, metadata, and distance."""
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


def print_results(query: str, hits: list[dict]) -> None:
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)
    for rank, hit in enumerate(hits, start=1):
        print(f"\n[{rank}] distance={hit['distance']}  source={hit['source']}")
        print(f"     file={hit['filename']}")
        print(f"     url={hit['url']}")
        print("-" * 60)
        print(hit["text"])
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed chunks and test retrieval.")
    parser.add_argument("--rebuild", action="store_true", help="Drop and rebuild the collection.")
    args = parser.parse_args()

    root = project_root()
    chunks_path = root / "data" / "chunks.jsonl"
    chroma_path = root / "data" / "chroma_db"

    if not chunks_path.exists():
        print(f"ERROR: {chunks_path} not found. Run build_document_pipeline.py first.")
        return 1

    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path.name}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    collection = build_vector_store(chunks, model, chroma_path, rebuild=args.rebuild)

    print("\n--- Retrieval test: 3 evaluation queries ---")
    for query in EVAL_QUERIES:
        hits = retrieve(query, collection, model, k=TOP_K)
        print_results(query, hits)

    return 0


if __name__ == "__main__":
    sys.exit(main())
