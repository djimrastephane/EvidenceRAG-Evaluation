# Top-K Chunks To LLM Flow

This note shows where retrieved chunks become LLM context in the current pipeline.

## High-Level Flow

```text
Question
  |
  v
API / runner
  |
  | calls
  v
SearchService.search(...)
[src/rag_pdf/services/search_service.py]

  |
  | 1. Encode question
  |    - MiniLM embedding
  |
  | 2. Retrieve candidates
  |    - dense search
  |    - BM25 search
  |
  | 3. Fuse + rerank
  |    - RRF / score fusion
  |    - optional cross-encoder
  |    - optional lexical/subsection boosts
  |
  | 4. Keep top-k
  |    - idx_list = fused_ranked[:k]
  |
  | 5. Build `results`
  |    each result contains:
  |    - chunk_id
  |    - pages
  |    - score
  |    - snippet
  |    - chunk_text
  v
results

  |
  | if include_generated_answer = False
  |------------------------------------> return retrieval results only
  |
  | if include_generated_answer = True
  v
_generate_local_answer(question, results, ...)
[src/rag_pdf/services/search_service.py]

  |
  v
_build_local_generation_prompt(question, results, ...)
[src/rag_pdf/services/search_service.py]

  |
  | loops through ranked `results`
  | and serializes chunk blocks like:
  |
  | [chunk_id=... pages=...]
  | <chunk text>
  |
  | subject to:
  | - max_context_chunks
  | - max_context_chars
  | - max_chunk_chars
  v
Prompt string

  |
  v
LocalLLMService.generate(prompt)
[src/rag_pdf/services/local_llm_service.py]

  |
  v
Local LLM response

  |
  v
Parse JSON answer + citations
[src/rag_pdf/services/search_service.py]

  |
  | validate citations against retrieved chunks
  | set:
  | - generation_status
  | - generation_confidence
  v
Final response
  |
  +-- retrieved top-k chunks
  +-- optional generated answer
  +-- optional citations
```

## Important Detail

`k` is not always the final number of chunks seen by the LLM.

After retrieval selects top `k`, prompt building may further reduce the context using:

- `max_context_chunks`
- `max_context_chars`
- `max_chunk_chars`

So the actual generation context is:

```text
top-k retrieved chunks
-> prompt-level trimming
-> final LLM context
```

## Main Code Locations

- `app/api/main.py`
  - passes `include_generated_answer` and generation limits into `SearchService.search(...)`
- `src/rag_pdf/services/search_service.py`
  - retrieval, fusion, top-k selection, prompt building, answer parsing, citation validation
- `src/rag_pdf/services/local_llm_service.py`
  - local LLM HTTP call

## Practical Takeaway

There is no separate script whose sole job is "send top-k chunks to the LLM".

That behavior is embedded inside `SearchService.search(...)`, which performs both:

- retrieval of top-k chunks
- optional generation over those chunks
