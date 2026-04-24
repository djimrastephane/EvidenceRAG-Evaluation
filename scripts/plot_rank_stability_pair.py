from __future__ import annotations

import argparse
from pathlib import Path

import sys

repo_root = Path(__file__).resolve().parents[1]
ui_path = repo_root / "app" / "ui"
if ui_path.exists() and str(ui_path) not in sys.path:
    sys.path.insert(0, str(ui_path))

import _matplotlib_env  # noqa: F401
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a paired success vs FP2 rank-stability figure from two PNG charts.")
    parser.add_argument(
        "--success-image",
        default="results/rank_stability/Grampian-2020-2021/Q_2021_FIN_01_rank_stability_rrf.png",
        help="Path to the success-case chart image.",
    )
    parser.add_argument(
        "--failure-image",
        default="results/rank_stability/Grampian-2020-2021/Q_2021_COV_01_rank_stability_rrf.png",
        help="Path to the FP2 failure-case chart image.",
    )
    parser.add_argument(
        "--output-path",
        default="results/rank_stability/Grampian-2020-2021/paired_success_vs_fp2.png",
        help="Output PNG path for the paired figure.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    success_path = Path(args.success_image).resolve()
    failure_path = Path(args.failure_image).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success_img = mpimg.imread(success_path)
    failure_img = mpimg.imread(failure_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9.6))
    captions = [
        (
            "Correct page ranked 1st.\n"
            "Competing pages remain close in score."
        ),
        (
            "Correct page present but ranked below 5.\n"
            "Small margin causes an FP2 near-miss."
        ),
    ]

    for ax, img, title, caption in zip(
        axes,
        [success_img, failure_img],
        ["Success: Correct Page Ranked 1st", "FP2 Near-Miss: Correct Page Ranked Below 5"],
        captions,
    ):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=13, pad=10)
        bbox = ax.get_position()
        fig.text(
            bbox.x0 + bbox.width / 2.0,
            0.115,
            caption,
            ha="center",
            va="top",
            fontsize=10.5,
            wrap=True,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#d7dde8"},
        )

    fig.suptitle("Rank Stability Comparison: Success vs FP2 Near-Miss", fontsize=16, y=0.98)
    fig.text(
        0.5,
        0.94,
        "Small score differences (< 0.02) determine whether the correct page is ranked first or missed (FP2).",
        ha="center",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.918,
        "Chunks are sorted by final fused rank. The dashed red floor marks weak candidates unlikely to be correct.",
        ha="center",
        fontsize=10.0,
    )
    fig.subplots_adjust(top=0.90, bottom=0.20, wspace=0.08)
    fig.savefig(output_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote paired figure to {output_path}")


if __name__ == "__main__":
    main()
