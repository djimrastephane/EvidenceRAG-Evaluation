from __future__ import annotations

import re

import pandas as pd

from rag_pdf.chunking import count_tokens
from rag_pdf.config import DEFAULT_CONFIG
from rag_pdf.table_markdown import (
    compact_table_headers,
    extract_table_row_records,
    render_row_block_markdown,
    select_summary_rows,
)

TABLE_EXTRACT_CFG = DEFAULT_CONFIG.TABLE_EXTRACT


def _norm_ws(s: object) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _trim_words(text: str, hard_max_words: int) -> str:
    words = str(text or "").split()
    if hard_max_words <= 0 or len(words) <= hard_max_words:
        return str(text or "").strip()
    return " ".join(words[:hard_max_words]).strip()


def _estimate_words(lines: list[str]) -> int:
    return len(" ".join(str(ln or "") for ln in lines).split())


def _table_caption_from_summary(table_summary: str) -> str:
    lines = str(table_summary or "").splitlines()
    line = _norm_ws(lines[0] if lines else "")
    if not line:
        return "Unknown"
    return line[:120]


def _table_headers_from_df(df: pd.DataFrame) -> list[str]:
    headers = compact_table_headers(df)
    if any(headers):
        return [h if h else "-" for h in headers]
    return [f"col_{i+1}" for i in range(len(df.columns))]


def _table_rows_from_df(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for _, row in df.iterrows():
        vals = [_norm_ws(v) for v in row.tolist()]
        vals = [v if v else "-" for v in vals]
        out.append(" | ".join(vals))
    return out


def _pack_lines_by_token_budget(
    lines: list[str],
    *,
    prefix_lines: list[str],
    chunk_size_tokens: int,
    enc,
) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    prefix_text = "\n".join([ln for ln in prefix_lines if _norm_ws(ln)])
    prefix_tokens = count_tokens(prefix_text, enc) if prefix_text else 0
    budget = max(40, int(chunk_size_tokens))

    for ln in lines:
        candidate = "\n".join(cur + [ln]).strip()
        candidate_tokens = count_tokens(candidate, enc) + prefix_tokens
        if cur and candidate_tokens > budget:
            body = "\n".join(cur).strip()
            if body:
                chunk = f"{prefix_text}\n{body}".strip() if prefix_text else body
                chunks.append(chunk)
            cur = [ln]
        else:
            cur.append(ln)

    if cur:
        body = "\n".join(cur).strip()
        if body:
            chunk = f"{prefix_text}\n{body}".strip() if prefix_text else body
            chunks.append(chunk)
    return chunks


def _build_row_group_lines(
    *,
    row_group: list[dict],
    headers: list[str],
    page_no: int,
    table_type: str | None,
    caption: str,
    local_facts_max: int,
) -> list[str]:
    categories = []
    seen_categories: set[str] = set()
    for row in row_group:
        category = str(row.get("category_context") or "").strip()
        if not category or category.lower() == "general" or category in seen_categories:
            continue
        seen_categories.add(category)
        categories.append(category)
    lines = [
        f"Table: {caption}",
        f"Page: {page_no}",
    ]
    if table_type:
        lines.append(f"Type: {str(table_type).replace('_', ' ').title()}")
    if len(categories) == 1:
        lines.append(f"Active category: {categories[0]}")
    elif len(categories) > 1:
        lines.append("Categories in block: " + " | ".join(categories[:4]))

    header_line = " | ".join(headers) if headers else ""
    if header_line:
        lines.append(f"Columns: {header_line}")

    row_md = render_row_block_markdown(row_group, headers)
    if row_md:
        lines.append("Rows:")
        lines.extend(row_md.splitlines())

    local_facts: list[str] = []
    for row in row_group:
        local_facts.extend([str(f).strip() for f in row.get("fact_lines", []) if str(f).strip()])
        if len(local_facts) >= local_facts_max:
            break
    local_facts = local_facts[:local_facts_max]
    if local_facts:
        lines.append("Facts:")
        lines.extend(local_facts)
    return [ln for ln in lines if _norm_ws(ln)]


def _pack_table_rows_into_groups(
    *,
    row_records: list[dict],
    headers: list[str],
    page_no: int,
    table_type: str | None,
    caption: str,
    word_target: int,
    hard_max_words: int,
    max_rows: int,
    local_facts_max: int,
) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []

    for row in row_records:
        candidate = current + [row]
        candidate_lines = _build_row_group_lines(
            row_group=candidate,
            headers=headers,
            page_no=page_no,
            table_type=table_type,
            caption=caption,
            local_facts_max=local_facts_max,
        )
        candidate_words = _estimate_words(candidate_lines)
        exceeds_rows = len(current) >= max_rows
        exceeds_target = current and candidate_words > word_target
        if exceeds_rows or exceeds_target:
            groups.append(current)
            current = [row]
            continue
        current = candidate

    if current:
        groups.append(current)

    normalized: list[list[dict]] = []
    for group in groups:
        if len(group) <= 1:
            normalized.append(group)
            continue
        group_lines = _build_row_group_lines(
            row_group=group,
            headers=headers,
            page_no=page_no,
            table_type=table_type,
            caption=caption,
            local_facts_max=local_facts_max,
        )
        if _estimate_words(group_lines) <= hard_max_words:
            normalized.append(group)
            continue
        for row in group:
            normalized.append([row])
    return normalized


def _build_table_summary_chunk_text(
    *,
    page_no: int,
    table_type: str | None,
    table_summary: str,
    raw_table: pd.DataFrame,
    word_target: int,
    key_rows_max: int,
) -> str:
    caption = _table_caption_from_summary(table_summary)
    headers = _table_headers_from_df(raw_table)
    summary_rows = select_summary_rows(raw_table, max_rows=key_rows_max)
    lines = [
        f"Table: {caption}",
        f"Page: {page_no}",
    ]
    if table_type:
        lines.append(f"Type: {str(table_type).replace('_', ' ').title()}")
    if table_summary:
        lines.append(str(table_summary).strip())
    if headers:
        lines.append("Columns: " + " | ".join(headers[: min(len(headers), 8)]))
    if summary_rows:
        lines.append("Key rows:")
        lines.extend(f"- {row}" for row in summary_rows)
    return _trim_words("\n".join(lines).strip(), word_target)


def _build_row_block_payloads(
    *,
    page_no: int,
    table_type: str | None,
    table_summary: str,
    raw_table: pd.DataFrame,
) -> list[dict]:
    caption = _table_caption_from_summary(table_summary)
    headers = _table_headers_from_df(raw_table)
    row_records = extract_table_row_records(raw_table)
    if not row_records:
        return []

    payloads: list[dict] = []
    summary_text = _build_table_summary_chunk_text(
        page_no=page_no,
        table_type=table_type,
        table_summary=table_summary,
        raw_table=raw_table,
        word_target=int(TABLE_EXTRACT_CFG.TABLE_SUMMARY_WORD_TARGET),
        key_rows_max=int(TABLE_EXTRACT_CFG.TABLE_SUMMARY_KEY_ROWS_MAX),
    )
    if summary_text:
        payloads.append(
            {
                "table_chunk_kind": "summary",
                "row_start_idx": None,
                "row_end_idx": None,
                "table_word_budget_target": int(TABLE_EXTRACT_CFG.TABLE_SUMMARY_WORD_TARGET),
                "chunk_text": summary_text,
            }
        )

    row_groups = _pack_table_rows_into_groups(
        row_records=row_records,
        headers=headers,
        page_no=page_no,
        table_type=table_type,
        caption=caption,
        word_target=int(TABLE_EXTRACT_CFG.TABLE_ROW_CHUNK_WORD_TARGET),
        hard_max_words=int(TABLE_EXTRACT_CFG.TABLE_ROW_CHUNK_WORD_HARD_MAX),
        max_rows=int(TABLE_EXTRACT_CFG.TABLE_ROW_CHUNK_MAX_ROWS),
        local_facts_max=int(TABLE_EXTRACT_CFG.TABLE_LOCAL_FACTS_MAX),
    )

    for group in row_groups:
        lines = _build_row_group_lines(
            row_group=group,
            headers=headers,
            page_no=page_no,
            table_type=table_type,
            caption=caption,
            local_facts_max=int(TABLE_EXTRACT_CFG.TABLE_LOCAL_FACTS_MAX),
        )
        chunk_text = _trim_words(
            "\n".join(lines).strip(),
            int(TABLE_EXTRACT_CFG.TABLE_ROW_CHUNK_WORD_HARD_MAX),
        )
        if not chunk_text:
            continue
        payloads.append(
            {
                "table_chunk_kind": "row_block",
                "row_start_idx": int(group[0].get("row_idx", 0)),
                "row_end_idx": int(group[-1].get("row_idx", 0)),
                "table_word_budget_target": int(TABLE_EXTRACT_CFG.TABLE_ROW_CHUNK_WORD_TARGET),
                "chunk_text": chunk_text,
            }
        )
    return payloads


def build_table_chunk_payloads(
    *,
    strategy: str,
    page_no: int,
    table_type: str | None,
    table_summary: str,
    raw_table: pd.DataFrame,
    header_injected_facts: str,
    table_markdown: str,
    chunk_size_tokens: int,
    enc,
) -> list[dict]:
    """Build the list of chunk payload dicts for a table using the configured chunking strategy (row_blocks or full_markdown)."""
    if strategy == "row_blocks":
        return _build_row_block_payloads(
            page_no=page_no,
            table_type=table_type,
            table_summary=table_summary,
            raw_table=raw_table,
        )

    if strategy == "baseline":
        parts = [table_summary]
        if header_injected_facts:
            parts.append("Table (header-injected facts):")
            parts.append(header_injected_facts)
        if table_markdown:
            parts.append("Table (markdown):")
            parts.append(table_markdown)
        return [
            {
                "table_chunk_kind": "full_table",
                "row_start_idx": None,
                "row_end_idx": None,
                "table_word_budget_target": None,
                "chunk_text": "\n\n".join(parts).strip(),
            }
        ]

    caption = _table_caption_from_summary(table_summary)
    table_prefix = f"TABLE | page={page_no} | {caption}"
    headers = _table_headers_from_df(raw_table)
    header_line = " | ".join(headers)
    row_lines = _table_rows_from_df(raw_table)

    if strategy == "row_preserving":
        prefix_lines = [table_prefix, f"COLUMNS: {header_line}"]
        return [
            {
                "table_chunk_kind": "row_preserving",
                "row_start_idx": None,
                "row_end_idx": None,
                "table_word_budget_target": None,
                "chunk_text": text,
            }
            for text in _pack_lines_by_token_budget(
                row_lines,
                prefix_lines=prefix_lines,
                chunk_size_tokens=chunk_size_tokens,
                enc=enc,
            )
        ]

    first_rows = row_lines[: min(5, len(row_lines))]
    units_line = ""
    year_line = ""
    for ln in first_rows:
        if not units_line and any(tok in ln for tok in ("£", "%", "000", "million", "m ")):
            units_line = ln
        if not year_line and re.search(r"\b(19|20)\d{2}(?:/\d{2,4})?\b", ln):
            year_line = ln
    header_chunk_lines = [table_prefix, f"COLUMNS: {header_line}"]
    if units_line:
        header_chunk_lines.append(f"UNITS: {units_line}")
    if year_line:
        header_chunk_lines.append(f"YEAR_LABELS: {year_line}")
    payloads = [
        {
            "table_chunk_kind": "header",
            "row_start_idx": None,
            "row_end_idx": None,
            "table_word_budget_target": None,
            "chunk_text": "\n".join(header_chunk_lines).strip(),
        }
    ]

    body_prefix = [table_prefix, f"COLUMNS: {header_line}"]
    for text in _pack_lines_by_token_budget(
        row_lines,
        prefix_lines=body_prefix,
        chunk_size_tokens=chunk_size_tokens,
        enc=enc,
    ):
        payloads.append(
            {
                "table_chunk_kind": "body",
                "row_start_idx": None,
                "row_end_idx": None,
                "table_word_budget_target": None,
                "chunk_text": text,
            }
        )
    return payloads


def build_table_chunk_texts(
    *,
    strategy: str,
    page_no: int,
    table_type: str | None,
    table_summary: str,
    raw_table: pd.DataFrame,
    header_injected_facts: str,
    table_markdown: str,
    chunk_size_tokens: int,
    enc,
) -> list[str]:
    """Return a list of plain text strings for table chunks (convenience wrapper around build_table_chunk_payloads)."""
    payloads = build_table_chunk_payloads(
        strategy=strategy,
        page_no=page_no,
        table_type=table_type,
        table_summary=table_summary,
        raw_table=raw_table,
        header_injected_facts=header_injected_facts,
        table_markdown=table_markdown,
        chunk_size_tokens=chunk_size_tokens,
        enc=enc,
    )
    return [str(p.get("chunk_text") or "").strip() for p in payloads if str(p.get("chunk_text") or "").strip()]
