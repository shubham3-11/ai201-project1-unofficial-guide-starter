# The Unofficial Guide — Project 1

---

## Domain

This system covers off-campus housing near Rutgers University New Brunswick, aimed at incoming graduate and transfer students. It aggregates student-generated advice from Reddit threads, official Rutgers and third-party listing pages, and a College Confidential forum discussion.

This knowledge is valuable because the decision of where to live directly affects cost, commute, safety, and quality of life during grad school — yet official Rutgers resources only cover listings and legal FAQs. The lived student experience (which neighborhoods feel safe at night, what rent is realistic on a student budget, which Facebook groups actually work) is scattered across Reddit, subreddit wikis, and forum threads with no single authoritative source. A RAG system can surface and synthesize this distributed knowledge on demand.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Rutgers Off-Campus Housing Listings | Official marketplace | `documents/01_rutgers_listings.txt` |
| 2 | Rutgers Off-Campus Living FAQ | Official FAQ | `documents/02_rutgers_faq.txt` |
| 3 | Ultimate Guide to Off-Campus Housing, r/rutgers | Reddit student guide | `documents/03_reddit_ultimate_guide.txt` |
| 4 | Off campus housing thread, r/rutgers | Reddit Q&A thread | `documents/04_reddit_off_campus_housing.txt` |
| 5 | Steps for finding housing, r/rutgers | Reddit Q&A thread | `documents/05_reddit_steps_finding_housing.txt` |
| 6 | Safe Off Campus Housing, r/rutgers | Reddit Q&A thread | `documents/06_reddit_safe_housing.txt` |
| 7 | How expensive is off-campus in NJ?, r/rutgers | Reddit Q&A thread | `documents/07_reddit_expensive_off_campus.txt` |
| 8 | Good apartments for grad students, r/rutgers | Reddit Q&A thread | `documents/08_reddit_grad_apartments.txt` |
| 9 | Rutgers University Off-Campus Housing, RentCollegePads | Third-party listing aggregator | `documents/09_rentcollegepads_rutgers.txt` |
| 10 | On-campus vs. off-campus, College Confidential | Forum discussion | `documents/10_collegeconfidential_on_vs_off.txt` |

---

## Chunking Strategy

**Chunk size:** 500 characters (~120 tokens)

**Overlap:** 100 characters (~20% overlap)

**Why these choices fit your documents:** The corpus mixes short Reddit comments (1–3 sentences) with longer student guides and official FAQ sections. At 500 characters a single chunk typically holds one complete Reddit comment or one FAQ question-and-answer pair without splitting mid-sentence. That size is large enough for opinion-based text ("Cook/Douglass is cheaper," "use Places4Students") to stay semantically self-contained so embedding-based retrieval can match it on its own.

The 100-character overlap addresses a structural problem in Reddit threads: the question is in the original post and useful answers are in the replies. When posts and comments are stored as sequential text, overlap increases the chance that at least one chunk contains both the context and the answer without requiring exact boundary alignment. Preprocessing stripped HTML tags, boilerplate UI lines (Share / Upvote / Reply), cookie notices, and normalized whitespace before chunking. Each document was also given a hand-written `[PAGE SUMMARY]`, `[POST]`, `[COMMENT n]`, and `[KEY TAKEAWAYS]` structure to give every chunk a semantic label even if it was short.

**Final chunk count:** 82 chunks across 10 source documents

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (local inference, no API key, no rate limits)

**Why this model:** It produces 384-dimensional dense embeddings well-suited for short English text like Reddit comments, runs entirely locally without API cost or network latency, and achieves strong general semantic similarity for the kind of informal, first-person student language in this corpus. Chunks are short (≤500 chars) so a 384-dimension model is more than sufficient — longer context models would be wasted here.

**Production tradeoff reflection:** If deploying for real users with no cost constraint, the main tradeoffs to weigh would be: (1) **Domain-specific accuracy** — `bge-large-en-v1.5` or a hosted model like Voyage AI's `voyage-2` would likely rank student jargon and housing-specific terms better than MiniLM, at the cost of higher latency and API dependency. (2) **Context length** — MiniLM handles up to 256 tokens; for FAQ-style documents with longer answers, a model supporting 512–8192 tokens would avoid truncation loss. (3) **Multilingual support** — a significant share of incoming international grad students may search in their first language; `paraphrase-multilingual-MiniLM-L12-v2` or Cohere Embed Multilingual would handle non-English queries. (4) **Freshness** — Reddit threads are added each semester; a re-embedding pipeline and fast incremental indexing would be needed to keep the vector store current.

---

## Grounded Generation

**LLM:** Groq `llama-3.3-70b-versatile` via the Groq Python SDK  
**Vector store:** ChromaDB persistent local collection with cosine similarity  
**Retrieval:** Top-5 chunks per query

**System prompt grounding instruction:**

The system prompt enforces grounding through five explicit rules, not a polite suggestion:

```
You are an assistant helping prospective and current Rutgers New Brunswick
students find off-campus housing information.

IMPORTANT RULES:
1. Answer ONLY using the information in the context documents provided.
2. Do NOT draw on your general training knowledge about housing, NJ, or Rutgers.
3. If the provided documents do not contain enough information to answer the
   question, respond with exactly: "I don't have enough information on that."
4. Be specific: quote or closely paraphrase what students or sources actually
   said. Do not generalize beyond what the context states.
5. Keep your answer concise (3–6 sentences) unless the question requires detail.
```

The user message prepends the retrieved chunks labeled `[Document 1]`, `[Document 2]`, etc., each with its source name and URL, then ends with the query. Using `temperature=0.2` further reduces the model's tendency to improvise beyond the context.

**How source attribution is surfaced in the response:** Attribution is handled in two layers. First, the LLM is instructed to reference specific documents by number in its answer. Second, and more importantly, source names and URLs are extracted programmatically from the retrieved chunk metadata in `generate.py` and appended as a separate `sources` list after generation — so attribution is guaranteed regardless of whether the model remembers to cite inline.

---

## Evaluation Report

The system was tested on all five queries from `planning.md`. Responses were generated with the full pipeline (retrieve → generate via Groq). Response accuracy is evaluated against the expected answers in the planning doc.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What areas near Rutgers do students mention as cheaper options? | Cook/Douglass area, areas farther from College Ave; student-reported cost tradeoffs | Mentions living farther from College Ave and near Busch/Livingston as cheaper; does not name Cook/Douglass specifically | Partially relevant — key Cook/Douglass chunk exists in documents but was not in top 5 | Partially accurate |
| 2 | What platforms do students recommend for finding housing? | Places4Students, Rutgers Facebook housing groups, official Rutgers portal | Names Places4Students, Rutgers Facebook housing groups, and College Pads (RentCollegePads) | Relevant — correct source documents ranked first | Accurate |
| 3 | What safety-related advice do students give? | Neighborhood safety, proximity to bus routes, checking areas before signing | Check crime maps, visit at different times of day, ask current tenants; Highland Park described as calmer, College Ave as loud | Relevant — "Safe Off Campus Housing" thread ranked first | Accurate |
| 4 | How expensive is it to live off campus? | Student-reported rent ranges, cost varies by location, roommates reduce cost | $1200–$2000+/month for 1-bed; $700–800 per person shared; roommates are the main way to reduce cost | Relevant — cost-specific Reddit thread and grad apartment budget chunk ranked high | Accurate |
| 5 | What apartment complexes do grad students recommend? | Specific complex names and pros/cons from the grad-focused Reddit thread | "I don't have enough information on that." | Partially relevant — grad apartment thread retrieved for post chunk but not answer chunks | Inaccurate (grounding worked correctly; retrieval failed to surface answer chunks) |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:**  
*"What apartment complexes or housing options do grad students recommend near Rutgers NB?"*

**What the system returned:**  
`"I don't have enough information on that."` — the grounding mechanism worked correctly; the LLM declined to fabricate an answer. However, the correct answer (The Vue near New Brunswick Station, Aspire, Highland Park, The Standard) *does* exist in the documents and should have been surfaced.

**Root cause (tied to a specific pipeline stage — retrieval / chunking):**  
The `08_reddit_grad_apartments.txt` document was chunked into 5 pieces. Chunk 0 (the original Reddit post) contains the phrase "graduate school at Rutgers New Brunswick" and "near Busch or Livingston campuses" — the context establishing that this is a grad-student-specific question. Chunks 1–4 contain the actual answer: specific complex names (The Vue, Aspire, The Standard), Highland Park as a neighborhood, and general search sites (Apartments.com, HotPads). But those answer chunks do *not* repeat the phrase "grad student" — they focus on budgets, commutes, and amenity tradeoffs. As a result, the embedding for chunks 1–4 is semantically closer to general housing discussion than to the query "grad students recommend apartment complexes," and they were beaten out by other documents' chunks in the top-5 ranking. The retrieval returned the question post (chunk 0) for Q1 and Q4, where the broader housing phrasing matched better, but the answer chunks never ranked high enough for Q5.

The 100-character overlap carried a few words of the post into chunk 1 ("budget in the comments is under $2000 per month") but that overlap is about budget, not about the grad-student context that would have helped the embedding match Q5.

**What you would change to fix it:**  
Two complementary fixes: (1) **Prepend parent-post context to every reply chunk** at ingestion time — e.g., prepend a summary sentence like "Context: grad student seeking quiet apartment near Busch/Livingston under $2000/month." to each comment chunk so the embedding captures the thread's subject. (2) **Increase overlap** from 100 to 200+ characters so that the post's framing text (including "graduate school" and "Busch or Livingston") carries into the first comment chunk. A third option would be a **parent-document retrieval** pattern: embed summaries of the full thread but return individual comment chunks at generation time, decoupling the retrieval granularity from the embedding granularity.

---

## Spec Reflection

**One way the spec helped you during implementation:**  
The Chunking Strategy section of `planning.md` forced an explicit decision before writing any code: 500-character chunks with 100-character overlap, with reasoning tied to the document structure (Reddit comments vs. FAQ sections). When implementing `chunk_document()` in `build_document_pipeline.py`, that pre-written specification made it possible to give the AI tool a precise target — not "split the text into chunks" but "pack paragraph units up to 500 characters, then apply 100-character readable overlap at sentence or paragraph boundaries." The resulting code matched the spec on the first iteration because the spec was concrete enough to be unambiguous.

**One way your implementation diverged from the spec, and why:**  
The planning.md Architecture diagram showed a linear five-stage pipeline without a `[KEY TAKEAWAYS]` stage. During ingestion, it became clear that some source documents (Reddit threads, College Confidential) produced very short, disconnected chunks because each comment was 50–100 characters — too small for a meaningful embedding. To compensate, hand-written structural labels (`[POST]`, `[COMMENT n]`, `[KEY TAKEAWAYS]`) were added directly to the `.txt` source files before running the pipeline. This pre-processing step was not in the spec and was discovered during retrieval testing when distance scores for comment-heavy documents were high (weak matches). The fix moved some of the semantic signal upstream — into the documents themselves — rather than changing the chunking parameters.

---

## AI Usage

**Instance 1 — Ingestion and chunking pipeline (Milestone 3)**

- *What I gave the AI:* The Chunking Strategy section of `planning.md` (chunk size: 500 chars, overlap: 100 chars, document types: Reddit threads and official FAQ pages) along with the project's document file format (plain `.txt` files with a `SOURCE / TYPE / URL / TITLE` header block above a `---` separator line).
- *What it produced:* `build_document_pipeline.py` with `load_txt_files()`, `clean_text()`, `chunk_document()`, and `print_diagnostics()`. The overlap implementation used fixed character slicing on the previous chunk's tail.
- *What I changed or overrode:* The initial overlap used a raw character slice, which often cut mid-word or mid-sentence. I directed the AI to replace it with a boundary-aware `get_overlap_prefix()` function that walks back to the nearest sentence boundary, paragraph break, or label marker (`[...]`) before taking the overlap — so chunks would begin at a readable point rather than mid-phrase. I also added the `is_low_value_chunk()` filter after seeing that short boilerplate chunks (navigation labels, single-word lines) were being embedded as if they contained meaningful content.

**Instance 2 — Embedding, retrieval, and generation (Milestones 4–5)**

- *What I gave the AI:* The Retrieval Approach section of `planning.md` (`all-MiniLM-L6-v2`, ChromaDB, top-k=5) plus the chunk format from `chunks.jsonl` (fields: `chunk_id`, `text`, `source`, `url`, `filename`, `chunk_index`) and the grounding requirement from planning.md ("answer only from retrieved chunks, cite source URL").
- *What it produced:* `embed_and_retrieve.py` with `build_vector_store()` using ChromaDB's `add()` call, a `retrieve()` function returning distances, and `generate.py` with a system prompt and `ask()` entry point.
- *What I changed or overrode:* The first system prompt draft used hedged language: "Try to answer using the provided documents." I hardened it to five numbered rules with an explicit fallback phrase ("I don't have enough information on that.") and `temperature=0.2` to reduce model improvisation. I also directed the AI to add programmatic source attribution as a deduplicated list extracted from chunk metadata — the original code relied entirely on the LLM to cite sources inline, which is unreliable. For Milestone 4, I added a `--rebuild` flag and a skip-if-populated guard after the first run, because re-embedding 82 chunks on every script invocation added 3–4 seconds of unnecessary startup cost during testing.
