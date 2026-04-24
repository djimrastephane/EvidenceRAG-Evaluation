# Current RAG Pipeline Diagram

```mermaid
flowchart TB
    subgraph Offline["Offline preprocessing and indexing"]
        PDF[PDF report]
        Extract[Hybrid page extraction<br/>PyMuPDF + fallback/OCR checks]
        Clean[Boilerplate stripping<br/>header/footer cleanup]
        Classify[Page classification<br/>text pages vs table pages]
        Section[Section and subsection inference]
        TextChunks[Text chunking<br/>segment-aware token chunks]
        TableChunks[Table extraction and table chunks<br/>summary + markdown + structured facts]
        ChunkStore[chunks.parquet]
        Embed[Embed chunk text<br/>MiniLM with section context]
        Dense[(FAISS dense index<br/>L2-normalized embeddings)]
        Meta[(chunk_meta.parquet)]
        Facts[(table_facts.parquet)]

        PDF --> Extract --> Clean --> Classify --> Section
        Section --> TextChunks --> ChunkStore
        Section --> TableChunks --> ChunkStore
        TableChunks --> Facts
        ChunkStore --> Embed --> Dense
        ChunkStore --> Meta
    end

    subgraph Online["Online query-time retrieval and answer stage"]
        Query[Input query]
        QEmbed[Embed query<br/>MiniLM]
        DenseSearch[Dense search in FAISS<br/>cosine via normalized inner product]
        BM25[BM25 lexical scoring<br/>over chunk text]
        Fuse[RRF fusion<br/>dense + BM25]
        CERerank{Cross-encoder rerank<br/>enabled?}
        CE[Cross-encoder rerank<br/>top-N fused candidates]
        Heuristic[Post-fusion rerank/boosts<br/>table, overlap, numeric, subsection]
        TopK[Final top-k chunks<br/>with pages, sections, scores]
        Pred[Deterministic answer pick<br/>from retrieved evidence]
        Gen{Include generated<br/>answer?}
        Num{Numeric question<br/>with strong candidate?}
        NumExtract[Deterministic numeric extractor]
        LLM[Local grounded LLM<br/>Ollama-compatible]
        Cite[Validate citations<br/>against retrieved chunks]
        Answer[Grounded answer<br/>+ citations]

        Query --> QEmbed --> DenseSearch
        Query --> BM25
        Dense --> DenseSearch
        Meta --> DenseSearch
        ChunkStore --> BM25
        DenseSearch --> Fuse
        BM25 --> Fuse
        Fuse --> CERerank
        CERerank -- Yes --> CE --> Heuristic
        CERerank -- No --> Heuristic
        Heuristic --> TopK
        TopK --> Pred
        TopK --> Gen
        Gen -- No --> Pred
        Gen -- Yes --> Num
        Num -- Yes --> NumExtract --> Answer
        Num -- No --> LLM --> Cite --> Answer
    end
```

## Notes

- This reflects the current default retrieval path: hybrid dense + BM25 fusion, not dense-only.
- Cross-encoder reranking is optional, but post-fusion heuristic reranking is part of the service path.
- Table facts are produced during preprocessing, but chunk retrieval still operates over chunk text rather than `table_facts.parquet`.
- Generated answers are optional and citation-validated; retrieval can also be used without generation.

## Main code references

- Preprocessing: `scripts/preprocess_hybrid.py`
- Index build: `scripts/build_index.py`
- Hybrid retrieval and generation service: `src/rag_pdf/services/search_service.py`
- Fusion helpers: `src/rag_pdf/retrieval/canonical_hybrid.py`
- BM25 and shared retrieval utilities: `src/rag_pdf/retrieval/hybrid_utils.py`
- Local LLM wrapper: `src/rag_pdf/services/local_llm_service.py`
