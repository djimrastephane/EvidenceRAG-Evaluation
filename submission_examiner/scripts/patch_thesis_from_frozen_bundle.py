"""Copy frozen-bundle figures into the thesis template and patch LaTeX source to reference them.

Copies the chunk ablation table (.tex), bootstrap CI panel (.png), retrieval FP1-FP7 heatmap
(.png), and side-by-side heatmap (.png) from the frozen bundle into the thesis figures/
directory, then rewrites the relevant \\includegraphics and \\input references in
methodology.tex and results.tex to use stable figures/ paths. Runs as a dry run by default;
pass --write to apply changes.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Copy selected frozen-bundle assets into the thesis figures directory and patch LaTeX to use stable figures/ paths."
    )
    p.add_argument("--bundle-dir", required=True, help="Frozen bundle directory under results/thesis_rebuild_freeze/<run-name>.")
    p.add_argument(
        "--thesis-root",
        default="/Users/djimra/MSc Data Science Jan 2025/Thesis documents/Thesis/University_of_Aberdeen_thesis_template",
        help="Root of the thesis template.",
    )
    p.add_argument("--write", action="store_true", help="Apply changes in place. Default is dry-run.")
    return p.parse_args()


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"Expected block not found:\n{old}")
    return text.replace(old, new, 1)


def asset_map(bundle_dir: Path, figures_dir: Path) -> list[tuple[Path, Path]]:
    """Return the list of (source, destination) path pairs to copy from the bundle to the thesis figures dir."""
    return [
        (
            bundle_dir / "tables" / "chunk_ablation_table.tex",
            figures_dir / "chunk_ablation_table.tex",
        ),
        (
            bundle_dir / "bootstrap" / "paired_bootstrap_ci_panel_Grampian_2020_2025_hybrid_vs_dense.png",
            figures_dir / "paired_bootstrap_ci_22456.png",
        ),
        (
            bundle_dir
            / "failure_analysis"
            / "retrieval_only_normalized"
            / "current_pipeline_fp1_fp7_heatmap_labeled.png",
            figures_dir / "current_pipeline_fp1_fp7_heatmap.png",
        ),
        (
            bundle_dir
            / "failure_analysis"
            / "comparison"
            / "fp1_fp7_heatmaps_side_by_side_norm_labeled.png",
            figures_dir / "fp1_fp7_heatmaps_side_by_side.png",
        ),
    ]


def patch_methodology(text: str) -> str:
    """Replace the inline chunk ablation tabular block in methodology.tex with an \\input reference."""
    new = "\\input{figures/chunk_ablation_table.tex}"
    if new in text:
        return text
    patched = re.sub(
        r"\\input\{[^}]*chunk_ablation_table\.tex\}",
        lambda _: new,
        text,
        count=1,
    )
    if patched != text:
        return patched

    old = """\\begin{tabular}{lcccccc}
\\toprule
Configuration & Page Hit@1 & $\\Delta$Hit@1 & MRR@10 & $\\Delta$MRR@10 & Queries & Chunks Indexed \\\\
\\midrule
224 / 56 & 0.708 & 0.000  & 0.792 & 0.000  & 250 & 1730 \\\\
256 / 64 & 0.664 & -0.044 & 0.771 & -0.022 & 250 & 1541 \\\\
280 / 90 & 0.676 & -0.032 & 0.778 & -0.015 & 250 & 1481 \\\\
400 / 100 & 0.652 & -0.056 & 0.753 & -0.039 & 250 & 1110 \\\\
\\bottomrule
\\end{tabular}"""
    return _replace_once(text, old, new)


def patch_results(text: str) -> str:
    """Rewrite \\includegraphics paths in results.tex to point at stable figures/ locations."""
    replacements = {
        r"\\includegraphics\[width=0\.8\\textwidth\]\{[^}]*paired_bootstrap[^}]*\.png\}":
            r"\includegraphics[width=0.8\textwidth]{figures/paired_bootstrap_ci_22456.png}",
        r"\\includegraphics\[width=0\.9\\textwidth\]\{[^}]*current_pipeline_fp1_fp7_heatmap[^}]*\.png\}":
            r"\includegraphics[width=0.9\textwidth]{figures/current_pipeline_fp1_fp7_heatmap.png}",
        r"\\includegraphics\[width=\\textwidth\]\{[^}]*fp1_fp7_heatmaps_side_by_side[^}]*\.png\}":
            r"\includegraphics[width=\textwidth]{figures/fp1_fp7_heatmaps_side_by_side.png}",
    }
    out = text
    for old, new in replacements.items():
        if new in out:
            continue
        patched = re.sub(old, lambda _m, repl=new: repl, out, count=1)
        if patched == out:
            raise ValueError(f"Expected figure reference not found for pattern:\n{old}")
        out = patched
    return out


def main() -> None:
    """Copy assets and patch LaTeX source, or print a dry-run summary if --write is not set."""
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).resolve()
    thesis_root = Path(args.thesis_root).resolve()
    figures_dir = thesis_root / "figures"
    methodology_path = thesis_root / "chapters" / "methodology.tex"
    results_path = thesis_root / "chapters" / "results.tex"

    methodology_text = methodology_path.read_text(encoding="utf-8")
    results_text = results_path.read_text(encoding="utf-8")

    patched_methodology = patch_methodology(methodology_text)
    patched_results = patch_results(results_text)
    assets = asset_map(bundle_dir=bundle_dir, figures_dir=figures_dir)

    if args.write:
        for src, dst in assets:
            if not src.exists():
                raise FileNotFoundError(f"Missing frozen-bundle asset: {src}")
            shutil.copy2(src, dst)
        methodology_path.write_text(patched_methodology, encoding="utf-8")
        results_path.write_text(patched_results, encoding="utf-8")
        for _src, dst in assets:
            print(f"Copied {dst}")
        print(f"Patched {methodology_path}")
        print(f"Patched {results_path}")
    else:
        print("Dry run only. Re-run with --write to apply.")
        for src, dst in assets:
            print(f"Would copy {src} -> {dst}")
        if patched_methodology != methodology_text:
            print(f"Would patch {methodology_path}")
        if patched_results != results_text:
            print(f"Would patch {results_path}")


if __name__ == "__main__":
    main()
