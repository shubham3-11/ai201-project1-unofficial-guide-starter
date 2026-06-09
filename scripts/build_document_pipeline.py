#!/usr/bin/env python3
"""Milestone 3: load, clean, chunk, and save local documents for the RAG pipeline."""

from __future__ import annotations

import html
import json
import random
import re
import sys
from pathlib import Path

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_BODY_LENGTH_WARN = 200
MIN_TOTAL_CHUNKS_WARN = 50
MAX_TOTAL_CHUNKS_WARN = 2000

HTML_ARTIFACT_PATTERNS = ("<div", "</", "&nbsp;", "&amp;")

BOILERPLATE_LINE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^\s*share\s*$",
        r"^\s*upvote\s*$",
        r"^\s*downvote\s*$",
        r"^\s*reply\s*$",
        r"^\s*report\s*$",
        r"^\s*save\s*$",
        r"^\s*hide\s*$",
        r"^\s*give award\s*$",
        r"^\s*cookie(s)?\s*(notice|policy|settings)?\s*$",
        r"^\s*accept all cookies\s*$",
        r"^\s*we use cookies\b",
        r"^\s*privacy policy\s*$",
        r"^\s*terms of (service|use)\s*$",
        r"^\s*skip to (main )?content\s*$",
        r"^\s*sign in\s*$",
        r"^\s*log in\s*$",
        r"^\s*subscribe\s*$",
        r"^\s*menu\s*$",
        r"^\s*home\s*$",
        r"^\s*search\s*$",
        r"^\s*advertisement\s*$",
        r"^\s*sponsored\s*$",
        r"^\s*click here\s*$",
        r"^\s*read more\s*$",
        r"^\s*loading\.\.\.\s*$",
        r"^\s*javascript is (disabled|required)\s*$",
        r"^\s*©\s*\d{4}\b",
        r"^\s*all rights reserved\.?\s*$",
        r"^\s*follow us on\b",
        r"^\s*back to top\s*$",
    ]
]

LABEL_ONLY_LINE = re.compile(r"^\[[^\]]+\]\s*$")
METADATA_LINE = re.compile(r"^(SOURCE|TYPE|URL|TITLE)\s*:\s*(.*)$", re.IGNORECASE)
SEPARATOR_PATTERN = re.compile(r"^-{3,}\s*$", re.MULTILINE)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_txt_files(documents_dir: Path) -> list[dict]:
    """Load top-level .txt files from documents/, excluding README notes."""
    if not documents_dir.exists():
        print(f"WARNING: documents directory not found: {documents_dir}")
        return []

    files = sorted(
        p
        for p in documents_dir.glob("*.txt")
        if p.is_file() and not p.name.lower().startswith("readme")
    )

    if not files:
        print(f"WARNING: No .txt documents found in {documents_dir}")
        return []

    documents: list[dict] = []
    for path in files:
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"WARNING: Could not read {path.name}: {exc}")
            continue

        try:
            metadata, body = parse_document(raw_text, path.name)
        except ValueError as exc:
            print(f"WARNING: Skipping malformed document {path.name}: {exc}")
            continue

        documents.append(
            {
                "filename": path.name,
                "source": metadata.get("source", ""),
                "type": metadata.get("type", ""),
                "url": metadata.get("url", ""),
                "title": metadata.get("title", ""),
                "raw_body": body,
            }
        )

    return documents


def parse_document(text: str, filename: str) -> tuple[dict[str, str], str]:
    """Parse metadata header and body separated by a --- line."""
    match = SEPARATOR_PATTERN.search(text)
    if not match:
        print(f"WARNING: {filename} has no '---' separator; treating entire file as body.")
        header = ""
        body = text.strip()
    else:
        header = text[: match.start()].strip()
        body = text[match.end() :].strip()

    metadata: dict[str, str] = {}
    for line in header.splitlines():
        meta_match = METADATA_LINE.match(line.strip())
        if meta_match:
            key = meta_match.group(1).lower()
            metadata[key] = meta_match.group(2).strip()

    return metadata, body


def clean_text(text: str) -> str:
    """Clean document body while preserving useful labels and content."""
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    entity_replacements = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&#39;": "'",
        "&quot;": '"',
        "&lt;": "<",
        "&gt;": ">",
    }
    for entity, replacement in entity_replacements.items():
        cleaned = cleaned.replace(entity, replacement)

    cleaned = html.unescape(cleaned)

    kept_lines: list[str] = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue

        if any(pattern.search(stripped) for pattern in BOILERPLATE_LINE_PATTERNS):
            continue

        if re.fullmatch(r"[\W_]+", stripped):
            continue

        if re.fullmatch(r"(?:Share|Upvote|Reply|Report|Save|Hide){1,3}", stripped, re.IGNORECASE):
            continue

        stripped = re.sub(r"[ \t]+", " ", stripped)
        kept_lines.append(stripped)

    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [line.strip() for line in text.split("\n") if line.strip()]
    return paragraphs


def split_long_paragraph(paragraph: str, chunk_size: int, overlap: int) -> list[str]:
    if len(paragraph) <= chunk_size:
        return [paragraph]

    pieces: list[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(paragraph):
        end = min(start + chunk_size, len(paragraph))
        piece = paragraph[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(paragraph):
            break
        start += step
    return pieces


def build_chunk_units(text: str, chunk_size: int) -> list[str]:
    """Convert cleaned text into chunkable units, splitting oversized paragraphs."""
    units: list[str] = []
    for paragraph in split_paragraphs(text):
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
        else:
            units.extend(split_long_paragraph(paragraph, chunk_size, CHUNK_OVERLAP))
    return units


def is_readable_overlap(prefix: str) -> bool:
    """Return True when an overlap prefix can stand alone without looking cut off."""
    if not prefix:
        return False
    if prefix.startswith("["):
        return True
    first = prefix[0]
    if first.isupper() or first.isdigit():
        return True
    return False


def get_overlap_prefix(previous_chunk: str, overlap: int) -> str:
    """Take a readable suffix from the previous chunk, starting at a natural boundary."""
    if not previous_chunk or overlap <= 0:
        return ""

    suffix = previous_chunk[-overlap:]
    label_match = list(re.finditer(r"(?:^|\n)(\[[^\]]+\])", suffix))
    if label_match:
        return suffix[label_match[-1].start() :].lstrip("\n").strip()

    paragraph_break = suffix.rfind("\n\n")
    if paragraph_break != -1:
        return suffix[paragraph_break + 2 :].strip()

    sentence_break = max(suffix.rfind(". "), suffix.rfind("! "), suffix.rfind("? "))
    if sentence_break != -1:
        return suffix[sentence_break + 2 :].strip()

    word_break = suffix.find(" ")
    if word_break != -1:
        return suffix[word_break + 1 :].strip()

    return suffix.strip()


def apply_overlap(chunks: list[str], overlap: int, chunk_size: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for idx in range(1, len(chunks)):
        prev = overlapped[-1]
        prefix = get_overlap_prefix(prev, overlap)
        current = chunks[idx]

        if prefix and is_readable_overlap(prefix) and not current.startswith(prefix):
            merged = f"{prefix}\n\n{current}".strip()
            if len(merged) <= chunk_size:
                overlapped.append(merged)
            else:
                overlapped.append(current)
        else:
            overlapped.append(current)
    return overlapped


def is_low_value_chunk(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    lines = [line.strip() for line in stripped.split("\n") if line.strip()]
    if not lines:
        return True

    if all(LABEL_ONLY_LINE.match(line) for line in lines):
        return True

    content_chars = re.sub(r"\[[^\]]+\]", "", stripped)
    content_chars = re.sub(r"\s+", "", content_chars)
    return len(content_chars) < 20


def chunk_document(
    cleaned_body: str,
    metadata: dict[str, str],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Chunk cleaned text with paragraph-aware packing and metadata attached."""
    units = build_chunk_units(cleaned_body, chunk_size)
    if not units:
        return []

    packed: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current_parts, current_len
        if current_parts:
            packed.append("\n\n".join(current_parts).strip())
        current_parts = []
        current_len = 0

    for unit in units:
        separator_len = 2 if current_parts else 0
        projected = current_len + separator_len + len(unit)
        if current_parts and projected > chunk_size:
            flush_current()
        current_parts.append(unit)
        current_len = len("\n\n".join(current_parts))

    flush_current()
    packed = apply_overlap(packed, overlap, chunk_size)

    chunks: list[dict] = []
    search_from = 0
    stem = Path(metadata["filename"]).stem

    for index, chunk_text in enumerate(packed):
        if is_low_value_chunk(chunk_text):
            continue

        start_char = cleaned_body.find(chunk_text, search_from)
        if start_char == -1:
            start_char = cleaned_body.find(chunk_text)
        end_char = start_char + len(chunk_text) if start_char >= 0 else len(chunk_text)
        if start_char >= 0:
            search_from = max(start_char + 1, search_from)

        chunk_id = f"{stem}_chunk_{index:04d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": metadata.get("source", ""),
                "type": metadata.get("type", ""),
                "url": metadata.get("url", ""),
                "title": metadata.get("title", ""),
                "filename": metadata.get("filename", ""),
                "chunk_index": index,
                "start_char": start_char,
                "end_char": end_char,
            }
        )

    return chunks


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def contains_html_artifacts(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in HTML_ARTIFACT_PATTERNS)


def print_diagnostics(
    raw_documents: list[dict],
    cleaned_documents: list[dict],
    chunks: list[dict],
) -> None:
    print("\n=== Document Pipeline Diagnostics ===")
    print(f"Documents loaded: {len(raw_documents)}")
    print(f"Cleaned documents: {len(cleaned_documents)}")
    print(f"Total chunks generated: {len(chunks)}")

    print("\nChunks per source file:")
    per_file: dict[str, int] = {}
    for chunk in chunks:
        per_file[chunk["filename"]] = per_file.get(chunk["filename"], 0) + 1
    for filename in sorted(per_file):
        print(f"  - {filename}: {per_file[filename]}")

    if cleaned_documents:
        sample = cleaned_documents[0]
        preview = sample["clean_body"][:600]
        print("\nSample cleaned document preview:")
        print(f"  filename: {sample['filename']}")
        print(f"  source: {sample.get('source', '')}")
        print(f"  title: {sample.get('title', '')}")
        print(f"  preview ({len(preview)} chars shown):")
        print("-" * 60)
        print(preview)
        if len(sample["clean_body"]) > 600:
            print("...")
        print("-" * 60)

    if chunks:
        sample_size = min(5, len(chunks))
        sample_chunks = random.sample(chunks, sample_size)
        print(f"\nFive random representative chunks ({sample_size} shown):")
        for idx, chunk in enumerate(sample_chunks, start=1):
            print(f"\n[{idx}] chunk_id: {chunk['chunk_id']}")
            print(f"    filename: {chunk['filename']}")
            print(f"    title/source: {chunk.get('title') or chunk.get('source')}")
            print(f"    character length: {len(chunk['text'])}")
            print("    chunk text:")
            print("-" * 60)
            print(chunk["text"])
            print("-" * 60)

    print("\n=== Quality Checks ===")
    if not raw_documents:
        print("WARNING: No documents were loaded.")
    if len(chunks) < MIN_TOTAL_CHUNKS_WARN:
        print(
            f"WARNING: Total chunks ({len(chunks)}) is fewer than {MIN_TOTAL_CHUNKS_WARN}. "
            "Consider reducing chunk size or adding more source text."
        )
    if len(chunks) > MAX_TOTAL_CHUNKS_WARN:
        print(
            f"WARNING: Total chunks ({len(chunks)}) exceeds {MAX_TOTAL_CHUNKS_WARN}. "
            "Review chunk size or source duplication."
        )

    short_docs = [
        doc for doc in cleaned_documents if len(doc.get("clean_body", "")) < MIN_BODY_LENGTH_WARN
    ]
    for doc in short_docs:
        print(
            f"WARNING: Document body is very short ({len(doc['clean_body'])} chars): "
            f"{doc['filename']}"
        )

    artifact_chunks = [c for c in chunks if contains_html_artifacts(c["text"])]
    if artifact_chunks:
        print(f"WARNING: {len(artifact_chunks)} chunk(s) contain obvious HTML artifacts.")
        for chunk in artifact_chunks[:5]:
            print(f"  - {chunk['chunk_id']} ({chunk['filename']})")
    else:
        print("HTML artifact check: PASS")

    empty_or_fragment = [c for c in chunks if len(c["text"].strip()) < 40]
    if empty_or_fragment:
        print(f"WARNING: {len(empty_or_fragment)} chunk(s) may be fragments (< 40 chars).")
    else:
        print("Chunk self-containment check: PASS (no very short chunks detected)")


def main() -> int:
    root = project_root()
    documents_dir = root / "documents"
    data_dir = root / "data"

    raw_documents = load_txt_files(documents_dir)
    raw_records = [
        {
            "filename": doc["filename"],
            "source": doc["source"],
            "type": doc["type"],
            "url": doc["url"],
            "title": doc["title"],
            "raw_body": doc["raw_body"],
        }
        for doc in raw_documents
    ]
    write_jsonl(data_dir / "raw_documents.jsonl", raw_records)

    cleaned_documents: list[dict] = []
    all_chunks: list[dict] = []

    for doc in raw_documents:
        clean_body = clean_text(doc["raw_body"])
        cleaned_doc = {
            "filename": doc["filename"],
            "source": doc["source"],
            "type": doc["type"],
            "url": doc["url"],
            "title": doc["title"],
            "clean_body": clean_body,
        }
        cleaned_documents.append(cleaned_doc)

        doc_chunks = chunk_document(
            clean_body,
            {
                "filename": doc["filename"],
                "source": doc["source"],
                "type": doc["type"],
                "url": doc["url"],
                "title": doc["title"],
            },
        )
        all_chunks.extend(doc_chunks)

    write_jsonl(
        data_dir / "clean_documents.jsonl",
        cleaned_documents,
    )
    write_jsonl(data_dir / "chunks.jsonl", all_chunks)

    print_diagnostics(raw_documents, cleaned_documents, all_chunks)

    print("\nSaved outputs:")
    print(f"  - {data_dir / 'raw_documents.jsonl'}")
    print(f"  - {data_dir / 'clean_documents.jsonl'}")
    print(f"  - {data_dir / 'chunks.jsonl'}")

    return 0 if raw_documents else 1


if __name__ == "__main__":
    sys.exit(main())
