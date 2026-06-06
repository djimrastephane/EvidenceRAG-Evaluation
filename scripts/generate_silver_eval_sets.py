"""Generate silver-label eval sets for NHS documents that lack one.

Runs a fixed set of question templates through the pipeline (retrieval + Ollama
generation) and saves high-confidence answers as eval_set.json alongside the
existing gold sets.  Silver sets are clearly marked in _meta so they can be
excluded from gold-only benchmarking.

Usage:
    python scripts/generate_silver_eval_sets.py [--doc-id DOC_ID] [--force]

Options:
    --doc-id DOC_ID   Process a single doc (default: all docs missing eval_set.json)
    --force           Overwrite existing eval sets (even gold ones — use with care)
    --dry-run         Print which docs would be processed without running
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

DATA_ROOT = REPO_ROOT / "data_processed"
MODEL_PATH = REPO_ROOT / "models" / "all-MiniLM-L6-v2"

# ── Question templates ────────────────────────────────────────────────────────
# {trust}      → "NHS Grampian" or "NHS Shetland"
# {year_slash} → "2024/25"
# {year_dash}  → "2024-25"
# {year_end}   → "2025"

TEMPLATES: list[dict] = [
    # Financial
    {
        "suffix": "FIN_01",
        "question": "What was the total net expenditure for {trust} in {year_slash}?",
        "answer_type": "number",
        "difficulty": "LEX",
        "category": "FIN",
    },
    {
        "suffix": "FIN_02",
        "question": "What were the total staff costs for {trust} in {year_slash}?",
        "answer_type": "number",
        "difficulty": "LEX",
        "category": "FIN",
    },
    {
        "suffix": "FIN_03",
        "question": "What was the Core Revenue Resource Limit for {trust} in {year_slash}?",
        "answer_type": "number",
        "difficulty": "LEX",
        "category": "FIN",
    },
    {
        "suffix": "FIN_04",
        "question": "What was the Capital Resource Limit for {trust} in {year_slash}?",
        "answer_type": "number",
        "difficulty": "LEX",
        "category": "FIN",
    },
    {
        "suffix": "FIN_05",
        "question": "Did {trust} report a surplus or deficit against its resource budget in {year_slash}?",
        "answer_type": "text",
        "difficulty": "MOD",
        "category": "FIN",
    },
    {
        "suffix": "FIN_06",
        "question": "What was the Cash Requirement limit for {trust} in {year_slash}?",
        "answer_type": "number",
        "difficulty": "LEX",
        "category": "FIN",
    },
    {
        "suffix": "FIN_07",
        "question": "What was the total operating expenditure for {trust} in {year_slash}?",
        "answer_type": "number",
        "difficulty": "MOD",
        "category": "FIN",
    },
    # Staff
    {
        "suffix": "STF_01",
        "question": "How many whole time equivalent staff did {trust} employ in {year_slash}?",
        "answer_type": "number",
        "difficulty": "LEX",
        "category": "STF",
    },
    {
        "suffix": "STF_02",
        "question": "What were the agency staff costs for {trust} in {year_slash}?",
        "answer_type": "number",
        "difficulty": "LEX",
        "category": "STF",
    },
    {
        "suffix": "STF_03",
        "question": "What was the sickness absence rate for {trust} in {year_slash}?",
        "answer_type": "number",
        "difficulty": "MOD",
        "category": "STF",
    },
    # Operational
    {
        "suffix": "OPS_01",
        "question": "What percentage of A&E patients were seen within 4 hours at {trust} in {year_slash}?",
        "answer_type": "number",
        "difficulty": "LEX",
        "category": "OPS",
    },
    {
        "suffix": "OPS_02",
        "question": "What was the 18-week referral to treatment waiting time performance for {trust} in {year_slash}?",
        "answer_type": "number",
        "difficulty": "MOD",
        "category": "OPS",
    },
    # Governance
    {
        "suffix": "GOV_01",
        "question": "Who was the Chief Executive of {trust} in {year_end}?",
        "answer_type": "text",
        "difficulty": "LEX",
        "category": "GOV",
    },
    {
        "suffix": "GOV_02",
        "question": "Who was the Chair of {trust} in {year_end}?",
        "answer_type": "text",
        "difficulty": "LEX",
        "category": "GOV",
    },
    # Efficiency
    {
        "suffix": "SAV_01",
        "question": "What was {trust}'s efficiency savings or cost improvement target for {year_slash}?",
        "answer_type": "number",
        "difficulty": "MOD",
        "category": "SAV",
    },
]

TRUST_NAMES = {
    "Grampian": "NHS Grampian",
    "Shetland": "NHS Shetland",
}


def _parse_doc_id(doc_id: str) -> dict:
    """Extract trust, year_start, year_end from a doc_id like 'Grampian-2024-2025'.

    Raises ValueError for doc_ids that don't match the expected format.
    """
    parts = doc_id.split("-")
    if len(parts) < 3 or not parts[-1].isdigit() or not parts[-2].isdigit():
        raise ValueError(f"Cannot parse doc_id '{doc_id}': expected format 'Trust-YYYY-YYYY'")
    trust_key = parts[0]
    year_start = int(parts[-2])
    year_end = int(parts[-1])
    year_start_short = str(year_start)[2:]
    year_end_short = str(year_end)[2:]
    return {
        "trust_key": trust_key,
        "trust": TRUST_NAMES.get(trust_key, f"NHS {trust_key}"),
        "year_start": year_start,
        "year_end": year_end,
        "year_slash": f"{year_start}/{year_end_short}",
        "year_dash": f"{year_start}-{year_end_short}",
        "year_end_str": str(year_end),
    }


def _fill_template(template: str, ctx: dict) -> str:
    return (
        template
        .replace("{trust}", ctx["trust"])
        .replace("{year_slash}", ctx["year_slash"])
        .replace("{year_dash}", ctx["year_dash"])
        .replace("{year_end}", ctx["year_end_str"])
    )


def _docs_missing_eval(force: bool) -> list[str]:
    docs = []
    for d in sorted(DATA_ROOT.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "faiss.index").exists():
            continue
        eval_path = d / "eval_set.json"
        if force or not eval_path.exists():
            docs.append(d.name)
    return docs


def generate_for_doc(doc_id: str, search_svc, verbose: bool = True) -> dict:
    try:
        ctx = _parse_doc_id(doc_id)
    except ValueError as e:
        print(f"  Skipping {doc_id}: {e}")
        return {"_meta": {"label_type": "silver", "doc_id": doc_id, "skipped": True}, "queries": []}

    data_dir = DATA_ROOT / doc_id
    queries = []
    year_code = str(ctx["year_end"])[2:]

    for tmpl in TEMPLATES:
        question = _fill_template(tmpl["question"], ctx)
        query_id = f"Q_{year_code}_{tmpl['suffix']}"

        if verbose:
            print(f"  {query_id}: {question[:65]}...", end=" ", flush=True)

        try:
            result = search_svc.search(
                data_dir=data_dir,
                question=question,
                k=5,
                include_generated_answer=True,
            )
        except Exception as e:
            if verbose:
                print(f"ERROR ({e})")
            continue

        gen_status = result.get("generation_status", "")
        gen_answer = result.get("generated_answer")
        gen_confidence = result.get("generation_confidence", 0.0) or 0.0
        results = result.get("results", [])

        # Only keep high-confidence generated answers
        if gen_status != "ok" or not gen_answer or gen_confidence < 1.0:
            if verbose:
                print(f"skipped (status={gen_status}, conf={gen_confidence})")
            continue

        expected_pages = results[0]["pages"] if results else []

        queries.append({
            "query_id": query_id,
            "question": question,
            "expected_pages": expected_pages,
            "expected_answer": gen_answer,
            "answer_type": tmpl["answer_type"],
            "difficulty": tmpl["difficulty"],
            "doc_id": doc_id,
            "year": ctx["year_end"],
            "category": tmpl["category"],
        })

        if verbose:
            print(f"OK → {gen_answer[:50]}")

        time.sleep(0.2)  # avoid hammering Ollama

    return {
        "_meta": {
            "dataset_name": f"{doc_id} Retrieval Silver Set",
            "description": (
                f"Auto-generated silver evaluation set for {ctx['trust']} "
                f"Annual Report {ctx['year_slash']}. Answers produced by the "
                "pipeline (retrieval + Ollama generation) and filtered to "
                "generation_confidence=1.0. NOT human-verified."
            ),
            "label_type": "silver",
            "doc_id": doc_id,
            "generated_by": "scripts/generate_silver_eval_sets.py",
        },
        "queries": queries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.doc_id:
        docs = [args.doc_id]
    else:
        docs = _docs_missing_eval(force=args.force)
        # Filter to known NHS trust docs only
        docs = [d for d in docs if d.startswith(("Grampian-", "Shetland-", "scottish_"))]

    if not docs:
        print("No docs to process.")
        return

    print(f"Docs to process ({len(docs)}): {docs}")
    if args.dry_run:
        return

    # Lazy import — heavy deps only needed at runtime
    from rag_pdf.services.search_service import SearchService

    search_svc = SearchService(repo_root=REPO_ROOT, model_path=MODEL_PATH)

    for doc_id in docs:
        eval_path = DATA_ROOT / doc_id / "eval_set.json"
        print(f"\n── {doc_id} ──")
        result = generate_for_doc(doc_id, search_svc, verbose=True)
        n = len(result["queries"])
        print(f"  → {n}/{len(TEMPLATES)} queries retained")
        if n == 0:
            print("  Skipping (no high-confidence answers).")
            continue
        eval_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"  Saved: {eval_path}")


if __name__ == "__main__":
    main()
