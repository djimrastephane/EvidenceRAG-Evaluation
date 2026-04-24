from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rag_pdf.question_router import route_question
from rag_pdf.retrieval.hybrid_utils import rrf_fuse, score_fuse
from rag_pdf.retrieval.rerank import (
    RerankConfig,
    normalize_text as rerank_normalize_text,
    numeric_density_boost,
    query_overlap_boost,
    segment_search_hit_boost,
    table_priority_boost,
)


def normalize_cross_encoder_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise cross-encoder scores to [0, 1]; return a zero array for empty or uniform inputs."""
    if scores.size == 0:
        return scores
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi <= lo:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32)


def fuse_ranked_lists(
    *,
    fusion_strategy: str,
    dense_ranked: list[int],
    bm25_ranked: list[int],
    dense_score_map: dict[int, float],
    bm25_score_map: dict[int, float],
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
) -> tuple[list[int], dict[int, float]]:
    """Fuse dense and BM25 ranked lists using RRF or weighted score fusion; return (ranked_indices, score_map)."""
    strategy = fusion_strategy if fusion_strategy in {"rrf", "score_fusion"} else "rrf"
    if strategy == "score_fusion":
        fused_ranked, fused_scores = score_fuse(
            dense_score_map=dense_score_map,
            bm25_score_map=bm25_score_map,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )
    else:
        fused_ranked, fused_scores = rrf_fuse(
            dense_ranked=dense_ranked,
            bm25_ranked=bm25_ranked,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )
    return fused_ranked, {int(idx): float(score) for idx, score in dict(fused_scores).items()}


def apply_post_fusion_rerank(
    *,
    question: str,
    fused_ranked: list[int],
    scores_map: dict[int, float],
    meta: pd.DataFrame,
    chunk_text_by_id: dict[str, str],
    rerank_cfg: RerankConfig,
    enable_lexical_rerank: bool,
    expected_section: str,
    expected_subsection: str,
    enable_subsection_boost: bool,
    subsection_boost: float,
    cross_page_out_of_section_penalty: float = 0.0,
) -> tuple[list[int], dict[int, float]]:
    """Apply lexical re-ranking boosts (table, overlap, numeric, segment, subsection) to the fused list."""
    ranked = list(fused_ranked)
    updated_scores = {int(idx): float(score) for idx, score in scores_map.items()}

    if enable_lexical_rerank and ranked:
        route = route_question(question)
        target_section = rerank_normalize_text(expected_section)
        for idx in ranked:
            row = meta.iloc[idx]
            cid = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
            ctext = chunk_text_by_id.get(cid, "")
            score = float(updated_scores.get(idx, 0.0))
            score += table_priority_boost(
                is_table_chunk=bool(row.get("is_table", False)),
                route_intent=route.intent if route is not None else "generic",
                config=rerank_cfg,
            )
            score += query_overlap_boost(
                question=question,
                chunk_text=ctext,
                config=rerank_cfg,
            )
            score += numeric_density_boost(
                question=question,
                chunk_text=ctext,
                config=rerank_cfg,
            )
            score += segment_search_hit_boost(
                question=question,
                segment_has_search_hit=bool(row.get("segment_has_search_hit", False)),
                config=rerank_cfg,
            )
            if cross_page_out_of_section_penalty > 0.0 and target_section:
                boundary_type = str(row.get("segment_boundary_type") or "").strip().upper()
                section_title = rerank_normalize_text(str(row.get("section_title", "") or ""))
                if boundary_type == "CROSS_PAGE_CONTINUATION" and section_title and section_title != target_section:
                    score -= float(cross_page_out_of_section_penalty)
            updated_scores[idx] = score
        ranked = sorted(ranked, key=lambda i: updated_scores.get(i, 0.0), reverse=True)

    if enable_subsection_boost and expected_subsection and "subsection_title" in meta.columns:
        target_subsection = rerank_normalize_text(expected_subsection)
        for idx in ranked:
            subsection_title = rerank_normalize_text(str(meta.iloc[idx].get("subsection_title", "") or ""))
            if subsection_title == target_subsection:
                updated_scores[idx] = float(updated_scores.get(idx, 0.0)) + float(subsection_boost)
        ranked = sorted(ranked, key=lambda i: updated_scores.get(i, 0.0), reverse=True)

    return ranked, updated_scores
