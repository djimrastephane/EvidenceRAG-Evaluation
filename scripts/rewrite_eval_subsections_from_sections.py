from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rag_pdf.sections import find_section_for_page


DEFAULT_DOCS = [
    "Grampian-2020-2021",
    "Grampian-2021-2022",
    "Grampian-2022-2023",
    "Grampian-2023-2024",
    "Grampian-2024-2025",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite eval_set expected_subsection values from current section metadata."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_processed"))
    parser.add_argument("--docs", nargs="*", default=DEFAULT_DOCS)
    parser.add_argument(
        "--out-report",
        type=Path,
        default=Path("results/eval_subsection_rewrite/eval_subsection_rewrite_report.csv"),
    )
    return parser.parse_args()


def load_eval_queries(path: Path) -> tuple[dict | None, list[dict]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        return obj, list(obj.get("queries", []))
    return None, list(obj)


def save_eval_queries(path: Path, original_obj: dict | None, queries: list[dict]) -> None:
    if original_obj is None:
        payload = queries
    else:
        payload = dict(original_obj)
        payload["queries"] = queries
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_rows: list[dict[str, object]] = []

    for doc in args.docs:
        base = args.data_root / doc
        eval_path = base / "eval_set.json"
        sections_path = base / "sections.parquet"
        original_obj, queries = load_eval_queries(eval_path)
        sections_df = pd.read_parquet(sections_path)

        for q in queries:
            qid = str(q.get("query_id") or q.get("id") or "")
            pages = q.get("expected_pages") or []
            first_page = int(pages[0]) if pages else None
            old_sub = str(q.get("expected_subsection") or "").strip()
            new_sub = old_sub
            if first_page is not None:
                _, _, inferred_sub = find_section_for_page(sections_df, first_page)
                new_sub = str(inferred_sub or "").strip()
            q["expected_subsection"] = new_sub
            report_rows.append(
                {
                    "doc_id": doc,
                    "query_id": qid,
                    "expected_pages": ",".join(str(p) for p in pages),
                    "old_expected_subsection": old_sub,
                    "new_expected_subsection": new_sub,
                    "changed": old_sub != new_sub,
                }
            )

        save_eval_queries(eval_path, original_obj, queries)

    report_df = pd.DataFrame(report_rows)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(args.out_report, index=False)
    print(f"Wrote: {args.out_report}")
    print(
        report_df.groupby("doc_id")["changed"].agg(["sum", "count"]).rename(columns={"sum": "changed_queries"})
    )


if __name__ == "__main__":
    main()
