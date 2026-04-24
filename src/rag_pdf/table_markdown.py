from __future__ import annotations

import re

import pandas as pd

from rag_pdf.config import DEFAULT_CONFIG

TABLE_EXTRACT_CFG = DEFAULT_CONFIG.TABLE_EXTRACT


def _sanitize_md_cell(val: str) -> str:
    cleaned = str(val or "").replace("\n", " ").replace("\r", " ").strip()
    return cleaned.replace("|", "\\|")


def _normalize_text(val: object) -> str:
    return re.sub(r"\s+", " ", str(val or "")).strip()


def _is_blankish(text: object) -> bool:
    s = _normalize_text(text).lower()
    return s in {"", "-", "—", "n/a", "na", "none"}


def table_to_markdown(
    df: pd.DataFrame,
    max_rows: int = TABLE_EXTRACT_CFG.TABLE_MARKDOWN_MAX_ROWS,
    max_cols: int = TABLE_EXTRACT_CFG.TABLE_MARKDOWN_MAX_COLS,
) -> str:
    """
    Render a dataframe as a pipe table to preserve structure for retrieval.
    """
    if df is None or len(df) == 0:
        return ""

    view = df.copy()
    if len(view.columns) > max_cols:
        view = view.iloc[:, :max_cols]
    if len(view) > max_rows:
        view = view.iloc[:max_rows]

    headers = []
    for i, col in enumerate(view.columns, start=1):
        name = str(col).strip()
        headers.append(name if name else f"col_{i}")

    header_row = "| " + " | ".join(_sanitize_md_cell(h) for h in headers) + " |"
    sep_row = "| " + " | ".join("---" for _ in headers) + " |"

    body_rows = []
    for _, row in view.iterrows():
        cells = [_sanitize_md_cell(v) for v in row.tolist()]
        body_rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header_row, sep_row] + body_rows)


def _split_md_row(line: str) -> list[str]:
    return [p.strip() for p in line.strip().strip("|").split("|")]


def _is_sep_row(line: str) -> bool:
    if "|" not in line:
        return False
    cells = _split_md_row(line)
    if not cells:
        return False
    return all(c and all(ch in "-: " for ch in c) and "-" in c for c in cells)


def _is_numeric_like(text: str) -> bool:
    s = str(text or "").strip().lower()
    if not s or s in {"n/a", "na", "-", "—"}:
        return False
    return bool(re.search(r"\d", s))


def _is_unit_fragment(text: str) -> bool:
    s = str(text or "").strip()
    if not s or " " in s:
        return False
    if any(sym in s for sym in ("£", "$", "€", "%")):
        return True
    return bool(re.fullmatch(r"[A-Za-z/().,-]{1,8}", s))


def _normalize_rows(rows: list[list[str]], width: int) -> list[list[str]]:
    out = []
    for row in rows:
        if len(row) < width:
            row = row + [""] * (width - len(row))
        elif len(row) > width:
            row = row[:width]
        out.append(row)
    return out


def _merge_headers(header_rows: list[list[str]], width: int) -> list[str]:
    merged = []
    for col in range(width):
        fragments = []
        units = []
        for row in header_rows:
            if col >= len(row):
                continue
            frag = str(row[col]).strip()
            if not frag:
                continue
            if _is_unit_fragment(frag):
                units.append(frag)
            else:
                fragments.append(frag)
        base = " ".join(fragments).strip()
        if units:
            unit_str = " ".join(units).strip()
            base = f"{base} {unit_str}".strip()
        merged.append(base)
    for i, name in enumerate(merged):
        if not name:
            merged[i] = f"col_{i}"
    merged[0] = "Row label"
    return merged


def compact_table_headers(df: pd.DataFrame) -> list[str]:
    """Return normalised column header strings for a DataFrame, filling blanks with positional fallbacks."""
    if df is None:
        return []
    headers = []
    for i, col in enumerate(df.columns, start=1):
        name = _normalize_text(col)
        headers.append(name if name else f"col_{i}")
    return headers


def build_local_fact_lines(row_record: dict, headers: list[str]) -> list[str]:
    """Convert a row record into a list of 'row > column : value' fact strings for chunk text."""
    row_label = str(row_record.get("row_label") or "").strip()
    if not row_label:
        row_label = f"row_{int(row_record.get('row_idx', 0)) + 1}"
    category_context = str(row_record.get("category_context") or "").strip()
    row_subject = row_label
    if category_context and category_context.lower() != "general":
        row_subject = f"[Category: {category_context}] {row_label}"
    cells = [str(c).strip() for c in row_record.get("cells", [])]
    facts: list[str] = []
    for j, value in enumerate(cells, start=1):
        if not value:
            continue
        col_path = headers[j] if j < len(headers) and headers[j] else f"col_{j}"
        facts.append(f"- {row_subject} > {col_path} : {value}")
    return facts


def render_row_markdown(row_record: dict, *, include_category: bool = False) -> str:
    """Render a single table row as a pipe-separated markdown line."""
    row_label = str(row_record.get("row_label") or "").strip() or "-"
    if include_category:
        category_context = str(row_record.get("category_context") or "").strip()
        if category_context and category_context.lower() != "general":
            row_label = f"{category_context} :: {row_label}"
    cells = [str(c).strip() or "-" for c in row_record.get("cells", [])]
    return " | ".join([row_label] + cells)


def _is_category_row(row_label: str, cells: list[str]) -> bool:
    label = _normalize_text(row_label)
    if len(label) < 4:
        return False
    if _is_numeric_like(label):
        return False
    non_blank_cells = [c for c in cells if not _is_blankish(c)]
    if not non_blank_cells:
        return True
    if any(_is_numeric_like(c) for c in non_blank_cells):
        return False
    return len(non_blank_cells) <= 1


def extract_table_row_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame into a list of row record dicts, tracking active category context and skipping header-only rows."""
    if df is None or len(df) == 0:
        return []
    headers = compact_table_headers(df)
    rows: list[dict] = []
    active_category = "General"
    for row_idx, (_, row) in enumerate(df.iterrows()):
        values = [_normalize_text(v) for v in row.tolist()]
        if not any(values):
            continue
        row_label = values[0] if values else ""
        cells = values[1:] if len(values) > 1 else []
        if _is_category_row(row_label, cells):
            active_category = row_label
            continue
        row_record = {
            "row_idx": row_idx,
            "row_label": row_label,
            "cells": cells,
            "category_context": active_category,
        }
        row_record["markdown_line"] = render_row_markdown(row_record)
        row_record["fact_lines"] = build_local_fact_lines(row_record, headers)
        rows.append(row_record)
    return rows


def render_row_block_markdown(row_group: list[dict], headers: list[str]) -> str:
    """Render a group of row records as a compact markdown block with a header line."""
    header_line = " | ".join(headers) if headers else ""
    lines: list[str] = []
    if header_line:
        lines.append(header_line)
    categories = {
        str(row.get("category_context") or "").strip()
        for row in row_group
        if str(row.get("category_context") or "").strip()
        and str(row.get("category_context") or "").strip().lower() != "general"
    }
    include_category = len(categories) > 1
    for row in row_group:
        lines.append(render_row_markdown(row, include_category=include_category))
    return "\n".join(ln for ln in lines if ln)


def select_summary_rows(df: pd.DataFrame, max_rows: int) -> list[str]:
    """Select up to max_rows representative markdown lines from a table DataFrame for chunk summary text."""
    if df is None or len(df) == 0 or max_rows <= 0:
        return []
    selected: list[str] = []
    for _, row in df.iterrows():
        values = [_normalize_text(v) for v in row.tolist() if _normalize_text(v)]
        if len(values) < 2:
            continue
        selected.append(" | ".join(values[:4]))
        if len(selected) >= max_rows:
            break
    return selected


def enrich_table_markdown(table_md: str) -> str:
    """
    Append compact header and row-map signals to a markdown table string.
    """
    lines = [ln.rstrip() for ln in (table_md or "").splitlines()]
    if not lines:
        return table_md

    table_lines = [ln for ln in lines if "|" in ln]
    if not table_lines:
        return table_md

    sep_idx = None
    for i, line in enumerate(table_lines):
        if _is_sep_row(line):
            sep_idx = i
            break
    if sep_idx is None or sep_idx == 0:
        return table_md

    header_row = _split_md_row(table_lines[sep_idx - 1])
    body_rows = [_split_md_row(ln) for ln in table_lines[sep_idx + 1 :]]
    width = max([len(header_row)] + [len(r) for r in body_rows] + [1])
    header_row = _normalize_rows([header_row], width)[0]
    body_rows = _normalize_rows(body_rows, width)

    header_row_numeric = all((not c) or _is_numeric_like(c) for c in header_row)

    body_start = None
    for idx, row in enumerate(body_rows):
        first = str(row[0]).strip()
        numeric_cells = sum(1 for c in row if _is_numeric_like(c))
        if first and numeric_cells >= 2:
            body_start = idx
            break
    if body_start is None:
        body_start = len(body_rows)
    if header_row_numeric:
        body_start = max(body_start, 1)

    raw_header_rows = [header_row] + body_rows[:body_start]
    header_rows = []
    for row in raw_header_rows:
        row0 = str(row[0]).strip()
        non_empty = sum(1 for c in row if str(c).strip())
        if row0 in {"Remuneration of:", "Executive Members", "Non Executive Members"}:
            break
        if len(row0) > 60 and non_empty == 1:
            continue
        header_rows.append(row)
    data_rows = body_rows[body_start:]

    headers = _merge_headers(header_rows, width)
    header_line = "Column headers: " + " | ".join(headers)

    alias_headers = ["col_0"] * width
    for j, full in enumerate(headers):
        low = full.lower()
        if j == 0:
            alias_headers[j] = "Row label"
        elif "salary" in low:
            alias_headers[j] = "Salary"
        elif "bonus" in low:
            alias_headers[j] = "Bonus"
        elif "benefits in kind" in low:
            alias_headers[j] = "Benefits"
        elif "sub total" in low or "subtotal" in low:
            alias_headers[j] = "Subtotal"
        elif "pension" in low:
            alias_headers[j] = "Pension"
        elif "total" in low and "remuneration" in low:
            alias_headers[j] = "Total remuneration"
        else:
            alias_headers[j] = f"col_{j}"

    row_candidates = []
    for idx, row in enumerate(data_rows):
        row_label = str(row[0]).strip()
        other_cells = row[1:]
        if not row_label:
            continue
        if not any(str(c).strip() for c in other_cells):
            continue
        numeric_cells = [j for j, c in enumerate(row) if _is_numeric_like(c)]
        if len(numeric_cells) < 2:
            continue
        row_candidates.append((idx, len(numeric_cells), row_label, row))

    top_rows = sorted(row_candidates, key=lambda x: (-x[1], x[0]))[:25]
    top_indices = {idx for idx, _, _, _ in top_rows}
    row_map_lines = []
    for idx, _, row_label, row in row_candidates:
        if idx not in top_indices:
            continue
        pairs = []
        for j, cell in enumerate(row[1:], start=1):
            cell_str = str(cell).strip()
            if not cell_str or not _is_numeric_like(cell_str):
                continue
            pairs.append(f"{alias_headers[j]}={cell_str}")
        if not pairs:
            continue
        row_map_lines.append(f"- {row_label} -> " + " ; ".join(pairs))
        if len(row_map_lines) >= 25:
            break

    if not row_map_lines:
        return table_md

    appended = "\n".join(["", header_line, "Row map:", *row_map_lines])
    return table_md.rstrip() + appended + "\n"


def build_header_injected_facts(table_md: str) -> str:
    """
    Build explicit cell-level facts with header paths for embedding.

    Output format:
      - <row label> > <column header path> : <value>
    """
    lines = [ln.rstrip() for ln in (table_md or "").splitlines()]
    if not lines:
        return ""

    table_lines = [ln for ln in lines if "|" in ln]
    if not table_lines:
        return ""

    sep_idx = None
    for i, line in enumerate(table_lines):
        if _is_sep_row(line):
            sep_idx = i
            break
    if sep_idx is None or sep_idx == 0:
        return ""

    header_row = _split_md_row(table_lines[sep_idx - 1])
    body_rows = [_split_md_row(ln) for ln in table_lines[sep_idx + 1 :]]
    width = max([len(header_row)] + [len(r) for r in body_rows] + [1])
    header_row = _normalize_rows([header_row], width)[0]
    body_rows = _normalize_rows(body_rows, width)

    header_row_numeric = all((not c) or _is_numeric_like(c) for c in header_row)

    body_start = None
    for idx, row in enumerate(body_rows):
        first = str(row[0]).strip()
        numeric_cells = sum(1 for c in row if _is_numeric_like(c))
        if first and numeric_cells >= 1:
            body_start = idx
            break
    if body_start is None:
        body_start = len(body_rows)
    if header_row_numeric:
        body_start = max(body_start, 1)

    raw_header_rows = [header_row] + body_rows[:body_start]
    header_rows = []
    for row in raw_header_rows:
        row0 = str(row[0]).strip()
        non_empty = sum(1 for c in row if str(c).strip())
        if row0 in {"Remuneration of:", "Executive Members", "Non Executive Members"}:
            break
        if len(row0) > 60 and non_empty == 1:
            continue
        header_rows.append(row)
    data_rows = body_rows[body_start : body_start + TABLE_EXTRACT_CFG.TABLE_HEADER_INJECTION_MAX_ROWS]

    headers = _merge_headers(header_rows, width)
    facts: list[str] = []
    for row in data_rows:
        row_label = str(row[0]).strip()
        if not row_label:
            continue
        for j, cell in enumerate(row[1:], start=1):
            val = str(cell).strip()
            if not val:
                continue
            col_path = str(headers[j]).strip() if j < len(headers) else f"col_{j}"
            if not col_path:
                col_path = f"col_{j}"
            facts.append(f"- {row_label} > {col_path} : {val}")
            if len(facts) >= TABLE_EXTRACT_CFG.TABLE_HEADER_INJECTION_MAX_FACTS:
                break
        if len(facts) >= TABLE_EXTRACT_CFG.TABLE_HEADER_INJECTION_MAX_FACTS:
            break

    return "\n".join(facts)
