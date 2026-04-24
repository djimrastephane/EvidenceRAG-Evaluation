from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd

from rag_pdf.config import DEFAULT_CONFIG

try:
    import camelot  # type: ignore
except Exception:
    camelot = None
    print("WARNING: camelot-py not installed. Table extraction will use pdfplumber only.")

TABLE_EXTRACT_CFG = DEFAULT_CONFIG.TABLE_EXTRACT


@dataclass
class TableResult:
    page_no: int
    flavor: str
    dataframe: pd.DataFrame
    parsing_report: dict[str, Union[float, int, str]]
    logs: list[str]


def _to_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return default


def _extract_parsing_report(table_obj: object) -> dict[str, Union[float, int, str]]:
    report = {}
    raw = getattr(table_obj, "parsing_report", None)
    if isinstance(raw, dict):
        report = dict(raw)
    return {
        "accuracy": _to_float(report.get("accuracy"), 0.0),
        "whitespace": _to_float(report.get("whitespace"), 100.0),
        "order": int(_to_float(report.get("order"), 0.0)),
        "page": str(report.get("page", "")),
    }


def _best_camelot_table(tables) -> Optional[object]:
    if not tables:
        return None
    candidates = []
    for t in tables:
        try:
            df = t.df
            rows, cols = df.shape
            if rows == 0 or cols == 0:
                continue
            rep = _extract_parsing_report(t)
            candidates.append((rows * cols, cols, rep.get("accuracy", 0.0), t))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]


def _table_signature(df: pd.DataFrame) -> tuple[int, int, tuple[str, ...]]:
    if df is None or not isinstance(df, pd.DataFrame):
        return (0, 0, tuple())
    rows, cols = df.shape
    probes: list[str] = []
    for ridx in range(min(2, rows)):
        row = []
        for cidx in range(min(3, cols)):
            row.append(str(df.iat[ridx, cidx]).strip().lower())
        probes.append("|".join(row))
    return (rows, cols, tuple(probes))


def _clean_valid_tables_from_camelot_list(
    tables,
    *,
    cleaner: Callable[[pd.DataFrame], Optional[pd.DataFrame]],
) -> list[pd.DataFrame]:
    out: list[pd.DataFrame] = []
    seen: set[tuple[int, int, tuple[str, ...]]] = set()
    if not tables:
        return out
    for t in tables:
        try:
            df = getattr(t, "df", None)
            if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
                continue
            cleaned = cleaner(df)
            if cleaned is None or len(cleaned) == 0:
                continue
            sig = _table_signature(cleaned)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(cleaned)
        except Exception:
            continue
    return out


def extract_tables_for_page(
    pdf_path: Path,
    page_no: int,
    config: Optional[dict] = None,
    *,
    cleaner: Callable[[pd.DataFrame], Optional[pd.DataFrame]],
) -> list[TableResult]:
    """
    Extract table(s) for one page with Camelot passes:
    1) lattice with strict gate
    2) hybrid with moderate gate
    3) stream fallback (table existence only)
    """
    policy = TABLE_EXTRACT_CFG
    cfg = dict(config or {})
    lattice_acc = int(cfg.get("lattice_accuracy_threshold", policy.CAMELOT_LATTICE_ACCURACY_THRESHOLD))
    lattice_ws = int(cfg.get("lattice_whitespace_max", policy.CAMELOT_LATTICE_WHITESPACE_MAX))
    hybrid_acc = int(cfg.get("hybrid_accuracy_threshold", policy.CAMELOT_HYBRID_ACCURACY_THRESHOLD))
    hybrid_ws = int(cfg.get("hybrid_whitespace_max", policy.CAMELOT_HYBRID_WHITESPACE_MAX))
    line_scale = int(cfg.get("line_scale", policy.CAMELOT_LINE_SCALE))
    resolution = int(cfg.get("resolution", policy.CAMELOT_RESOLUTION))
    row_tol = int(cfg.get("row_tol", policy.CAMELOT_STREAM_ROW_TOL))
    edge_tol = int(cfg.get("edge_tol", policy.CAMELOT_STREAM_EDGE_TOL))
    return_all_tables = bool(cfg.get("return_all_tables", False))
    secondary_bottom_area = cfg.get("secondary_bottom_area")
    logs: list[str] = []

    if camelot is None:
        logs.append(f"page {page_no}: camelot unavailable")
        print(logs[-1])
        return []

    passes = [
        {
            "name": "lattice",
            "kwargs": {
                "flavor": "lattice",
                "strip_text": " .\n",
                "split_text": True,
                "copy_text": ["v"],
                "line_scale": line_scale,
                "resolution": resolution,
            },
            "gate": (lattice_acc, lattice_ws),
            "strict_gate": True,
        },
        {
            "name": "hybrid",
            "kwargs": {
                "flavor": "hybrid",
                "strip_text": " .\n",
                "split_text": True,
                "copy_text": ["v"],
                "line_scale": line_scale,
                "resolution": resolution,
                "row_tol": row_tol,
                "edge_tol": edge_tol,
            },
            "gate": (hybrid_acc, hybrid_ws),
            "strict_gate": True,
        },
        {
            "name": "stream",
            "kwargs": {
                "flavor": "stream",
                "strip_text": " .\n",
                "split_text": True,
                "row_tol": row_tol,
                "edge_tol": edge_tol,
            },
            "gate": None,
            "strict_gate": False,
        },
    ]

    for p in passes:
        pname = str(p["name"])
        try:
            tables = camelot.read_pdf(
                str(pdf_path),
                pages=str(page_no),
                **p["kwargs"],  # type: ignore[arg-type]
            )
        except Exception as e:
            logs.append(f"page {page_no}: {pname} exception: {type(e).__name__}: {e}")
            print(logs[-1])
            continue

        if not tables or len(tables) == 0:
            logs.append(f"page {page_no}: {pname} failed (no tables)")
            print(logs[-1])
            continue

        best_table = _best_camelot_table(tables)
        if best_table is None:
            logs.append(f"page {page_no}: {pname} failed (empty/invalid parsed tables)")
            print(logs[-1])
            continue

        report = _extract_parsing_report(best_table)
        acc = float(report.get("accuracy", 0.0))
        ws = float(report.get("whitespace", 100.0))

        if p["strict_gate"]:
            gate_acc, gate_ws = p["gate"]  # type: ignore[misc]
            if acc < float(gate_acc):
                logs.append(f"page {page_no}: {pname} failed (accuracy {acc:.2f} < {gate_acc})")
                print(logs[-1])
                continue
            if ws > float(gate_ws):
                logs.append(f"page {page_no}: {pname} failed (whitespace {ws:.2f} > {gate_ws})")
                print(logs[-1])
                continue
        else:
            logs.append(f"page {page_no}: {pname} accepted (accuracy={acc:.2f}, whitespace={ws:.2f})")
            print(logs[-1])

        if return_all_tables:
            cleaned_tables = _clean_valid_tables_from_camelot_list(tables, cleaner=cleaner)
            if not cleaned_tables:
                logs.append(f"page {page_no}: {pname} failed (no valid cleaned tables)")
                print(logs[-1])
                continue
            logs.append(
                f"page {page_no}: {pname} succeeded (accuracy={acc:.2f}, whitespace={ws:.2f}, tables={len(cleaned_tables)})"
            )
            print(logs[-1])
            out = [
                TableResult(
                    page_no=page_no,
                    flavor=pname,
                    dataframe=tbl,
                    parsing_report=report,
                    logs=logs.copy(),
                )
                for tbl in cleaned_tables
            ]
            if secondary_bottom_area and pname == "stream":
                try:
                    extra = camelot.read_pdf(
                        str(pdf_path),
                        pages=str(page_no),
                        flavor="stream",
                        strip_text=" .\n",
                        split_text=True,
                        row_tol=row_tol,
                        edge_tol=edge_tol,
                        table_areas=[str(secondary_bottom_area)],
                    )
                    extra_tables = _clean_valid_tables_from_camelot_list(extra, cleaner=cleaner)
                    seen = {_table_signature(r.dataframe) for r in out}
                    add_n = 0
                    for tbl in extra_tables:
                        sig = _table_signature(tbl)
                        if sig in seen:
                            continue
                        seen.add(sig)
                        out.append(
                            TableResult(
                                page_no=page_no,
                                flavor="stream_bottom",
                                dataframe=tbl,
                                parsing_report={"accuracy": 0.0, "whitespace": 0.0, "order": 0, "page": str(page_no)},
                                logs=logs.copy(),
                            )
                        )
                        add_n += 1
                    if add_n:
                        logs.append(f"page {page_no}: stream_bottom added {add_n} table(s)")
                        print(logs[-1])
                except Exception as e:
                    logs.append(f"page {page_no}: stream_bottom exception: {type(e).__name__}: {e}")
                    print(logs[-1])
            return out

        df = getattr(best_table, "df", None)
        if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
            logs.append(f"page {page_no}: {pname} failed (best table has no dataframe)")
            print(logs[-1])
            continue
        cleaned = cleaner(df)
        if cleaned is None or len(cleaned) == 0:
            logs.append(f"page {page_no}: {pname} failed (cleaned dataframe empty)")
            print(logs[-1])
            continue

        logs.append(f"page {page_no}: {pname} succeeded (accuracy={acc:.2f}, whitespace={ws:.2f})")
        print(logs[-1])
        return [
            TableResult(
                page_no=page_no,
                flavor=pname,
                dataframe=cleaned,
                parsing_report=report,
                logs=logs.copy(),
            )
        ]

    logs.append(f"page {page_no}: all camelot passes failed")
    print(logs[-1])
    return []


def extract_table_camelot(pdf_path: Path, page_no: int) -> pd.Optional[pd.DataFrame]:
    """Extract the best table DataFrame from a single PDF page using Camelot.

    Tries lattice mode first (ruled-line tables); falls back to stream mode (whitespace-aligned
    tables) if lattice accuracy is below threshold or no tables are found.  Returns None when
    Camelot is unavailable or both passes fail.
    """
    if camelot is None:
        return None

    def _best_camelot_df(tables) -> pd.Optional[pd.DataFrame]:
        if not tables:
            return None
        candidates = []
        for t in tables:
            try:
                df = t.df
                rows, cols = df.shape
                if rows == 0 or cols == 0:
                    continue
                candidates.append((rows * cols, cols, t.accuracy, df))
            except Exception:
                continue
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][3]

    try:
        tables = camelot.read_pdf(str(pdf_path), pages=str(page_no), flavor="lattice", strip_text="\n")
        if len(tables) > 0 and tables[0].accuracy >= TABLE_EXTRACT_CFG.CAMELOT_LATTICE_ACCURACY_THRESHOLD:
            best = _best_camelot_df(tables)
            if best is not None:
                return best
    except Exception:
        pass

    try:
        tables = camelot.read_pdf(str(pdf_path), pages=str(page_no), flavor="stream")
        if len(tables) > 0:
            best = _best_camelot_df(tables)
            if best is not None:
                return best
    except Exception:
        pass

    return None
