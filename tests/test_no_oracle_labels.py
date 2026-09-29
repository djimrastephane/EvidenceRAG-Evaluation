"""Gold eval labels (expected_section / expected_subsection / expected_pages / answer_type)
must not influence default ranking or answer extraction.

Each test first checks the default path ignores the labels, then enables the oracle
explicitly to confirm the fixture is sensitive enough to detect a leak.
"""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pandas as pd

from rag_pdf.retrieval.canonical_hybrid import apply_post_fusion_rerank
from rag_pdf.retrieval.rerank import RerankConfig
from thesis_rag.retrieval_dense import dense_retrieve_legacy_style
from thesis_rag.retrieval_hybrid import hybrid_retrieve_legacy_style
from thesis_rag.schemas import ChunkRecord, QueryRecord

TEXT = "Revenue resource limit performance against target for the year."


def _chunks() -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id=f"doc:{i}",
            doc_id="doc",
            page_number=10 + i,
            chunk_index=i,
            text=TEXT,
            token_count=12,
            word_count=10,
            pages=[10 + i],
            section_title="PERFORMANCE REPORT" if i else "ACCOUNTABILITY REPORT",
            subsection_title=f"SUB {i}",
            segment_boundary_type="CROSS_PAGE_CONTINUATION" if i == 0 else None,
        )
        for i in range(3)
    ]


def _query(with_gold: bool) -> QueryRecord:
    return QueryRecord(
        query_id="Q_2025_FIN_01",
        query_text="What was the revenue resource limit?",
        doc_id="doc",
        gold_pages=[12],
        expected_section="PERFORMANCE REPORT" if with_gold else None,
        expected_subsection="SUB 2" if with_gold else None,
    )


class FakeBM25:
    def get_scores(self, _tokens):
        return [0.3, 0.2, 0.1]


class FakeIndex:
    def search(self, query_vectors: np.ndarray, top_k: int):
        scores = np.array([[0.50, 0.49, 0.48][:top_k]], dtype=np.float32)
        indices = np.array([[0, 1, 2][:top_k]], dtype=np.int64)
        return scores, indices


def _hybrid_order(query: QueryRecord, **kwargs) -> list[str]:
    _, _, hits = hybrid_retrieve_legacy_style(
        chunks=_chunks(),
        queries=[query],
        dense_scores=np.array([[0.50, 0.49, 0.48]], dtype=np.float32),
        dense_indices=np.array([[0, 1, 2]], dtype=np.int64),
        bm25=FakeBM25(),
        max_k_search=3,
        dense_weight=0.5,
        bm25_weight=2.0,
        rrf_k=20,
        **{"enable_lexical_rerank": False, **kwargs},
    )
    return [hit.chunk_id for hit in sorted(hits, key=lambda h: h.rank)]


def _dense_order(query: QueryRecord, **kwargs) -> list[str]:
    hits = dense_retrieve_legacy_style(
        FakeIndex(),
        _chunks(),
        [query],
        np.zeros((1, 4), dtype=np.float32),
        top_k=3,
        max_k_search=3,
        **kwargs,
    )
    return [hit.chunk_id for hit in sorted(hits, key=lambda h: h.rank)]


def test_hybrid_default_ignores_gold_labels() -> None:
    assert _hybrid_order(_query(True)) == _hybrid_order(_query(False)) == ["doc:0", "doc:1", "doc:2"]
    # Sensitivity: the explicit oracle does move the gold-subsection chunk to the top.
    assert _hybrid_order(_query(True), enable_subsection_boost=True)[0] == "doc:2"


def test_dense_default_ignores_gold_labels() -> None:
    assert _dense_order(_query(True)) == _dense_order(_query(False)) == ["doc:0", "doc:1", "doc:2"]
    assert _dense_order(_query(True), enable_subsection_boost=True)[0] == "doc:2"


def test_post_fusion_rerank_default_ignores_gold_labels() -> None:
    meta = pd.DataFrame([c.to_dict() for c in _chunks()])
    scores = {0: 0.119, 1: 0.114, 2: 0.109}
    common = dict(
        question="What was the revenue resource limit?",
        fused_ranked=[0, 1, 2],
        scores_map=scores,
        meta=meta,
        chunk_text_by_id={},
        rerank_cfg=RerankConfig(),
        enable_lexical_rerank=True,
    )
    baseline, _ = apply_post_fusion_rerank(**common)
    with_labels, _ = apply_post_fusion_rerank(
        **common, expected_section="PERFORMANCE REPORT", expected_subsection="SUB 2"
    )
    assert with_labels == baseline
    oracle, _ = apply_post_fusion_rerank(
        **common,
        expected_section="PERFORMANCE REPORT",
        expected_subsection="SUB 2",
        enable_subsection_boost=True,
        subsection_boost=0.05,
    )
    assert oracle[0] == 2


def test_hybrid_default_applies_no_section_penalty() -> None:
    # The cross-page out-of-section penalty reads the gold expected_section; default is off.
    query = replace(_query(True), expected_subsection=None)
    # The penalty is applied inside the lexical rerank pass, so enable it here.
    assert _hybrid_order(query, enable_lexical_rerank=True) == ["doc:0", "doc:1", "doc:2"]
    penalised = _hybrid_order(query, enable_lexical_rerank=True, cross_page_out_of_section_penalty=0.08)
    assert penalised[0] != "doc:0"


def test_search_service_answer_extraction_takes_no_gold_inputs() -> None:
    from rag_pdf.services import search_service

    params = inspect.signature(search_service.SearchService._predict_answer).parameters
    assert "expected_pages" not in params
    assert "query_id" not in params
    assert not hasattr(search_service, "ANSWER_EVAL_STRICT_PAGE")
    assert not hasattr(search_service, "ENABLE_SUBSECTION_BOOST")
