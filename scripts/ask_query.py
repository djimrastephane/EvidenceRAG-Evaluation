from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag_pdf.services.search_service import SearchService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question against one processed document using the current SearchService."
    )
    parser.add_argument(
        "--doc",
        required=True,
        help="Document id, e.g. Grampian-2024-2025",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask.",
    )
    parser.add_argument(
        "--data-root",
        default="data_processed",
        help="Root directory containing processed document folders.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return.",
    )
    parser.add_argument(
        "--query-id",
        default=None,
        help="Optional eval_set query_id.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Enable grounded answer generation.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON response.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = REPO_ROOT / "models" / "all-MiniLM-L6-v2"
    data_dir = REPO_ROOT / args.data_root / args.doc
    if not data_dir.exists():
        raise SystemExit(f"Processed document directory not found: {data_dir}")

    service = SearchService(repo_root=REPO_ROOT, model_path=model_path)
    out = service.search(
        data_dir=data_dir,
        question=args.question,
        k=int(args.k),
        query_id=args.query_id,
        include_generated_answer=bool(args.generate),
    )

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print(f"Question: {args.question}")
    print(f"Document: {args.doc}")
    print(f"Predicted answer: {out.get('predicted_answer')}")
    results = out.get("results", [])
    results_by_chunk_id = {
        str(row.get("chunk_id") or "").strip(): row
        for row in results
        if str(row.get("chunk_id") or "").strip()
    }
    if args.generate:
        print(f"Generated answer: {out.get('generated_answer')}")
        print(f"Generation status: {out.get('generation_status')}")
        citations = out.get("generated_citations") or []
        if citations:
            print("Generated citations:")
            for c in citations:
                print(f"  - chunk_id={c.get('chunk_id')} page={c.get('page')}")
            print("\nCited evidence:")
            for i, c in enumerate(citations, start=1):
                chunk_id = str(c.get("chunk_id") or "").strip()
                row = results_by_chunk_id.get(chunk_id)
                if row is None:
                    print(f"\n{i}. chunk_id={chunk_id}")
                    print("   text=<not present in returned top-k results>")
                    continue
                text = str(row.get("chunk_text") or "").replace("\n", " ").strip()
                print(f"\n{i}. chunk_id={chunk_id}")
                print(f"   pages={row.get('pages')}")
                print(f"   section={row.get('section')}")
                print(f"   subsection={row.get('subsection')}")
                print(f"   text={text}")
        else:
            print("Generated citations: none")

    print("\nTop results:")
    for i, row in enumerate(results[: int(args.k)], start=1):
        text = str(row.get("chunk_text") or "").replace("\n", " ").strip()
        print(f"\n{i}. chunk_id={row.get('chunk_id')}")
        print(f"   pages={row.get('pages')}")
        print(f"   score={row.get('score')}")
        print(f"   section={row.get('section')}")
        print(f"   subsection={row.get('subsection')}")
        print(f"   text={text[:300]}...")


if __name__ == "__main__":
    main()
