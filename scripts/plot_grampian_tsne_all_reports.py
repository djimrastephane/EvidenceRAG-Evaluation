#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull


DOC_RE = re.compile(r"^Grampian-(?P<start>\d{4})-(?P<end>\d{4})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a t-SNE projection for all 21 NHS Grampian report embeddings."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_variants/tiktoken_all_docs_224_56"),
        help="Directory containing per-report embeddings.npy files.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("results/grampian_tsne_all_reports.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/grampian_tsne_all_reports_coords.csv"),
        help="Output CSV path for projected coordinates.",
    )
    parser.add_argument(
        "--out-era-centroids-png",
        type=Path,
        default=Path("results/grampian_tsne_all_reports_era_centroids.png"),
        help="Output PNG path for the separate era-centroid plot.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--perplexity",
        type=float,
        default=35.0,
        help="t-SNE perplexity.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=8.0,
        help="Scatter point size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.22,
        help="Scatter alpha.",
    )
    return parser.parse_args()


def load_embeddings(data_root: Path) -> tuple[np.ndarray, pd.DataFrame]:
    vectors: list[np.ndarray] = []
    rows: list[dict[str, object]] = []

    for doc_dir in sorted(p for p in data_root.glob("Grampian-*") if p.is_dir()):
        match = DOC_RE.match(doc_dir.name)
        if not match:
            continue
        emb_path = doc_dir / "embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
        arr = np.load(emb_path)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array in {emb_path}, got shape {arr.shape}")
        start_year = int(match.group("start"))
        end_year = int(match.group("end"))
        vectors.append(arr.astype(np.float32, copy=False))
        for idx in range(arr.shape[0]):
            rows.append(
                {
                    "doc_id": doc_dir.name,
                    "start_year": start_year,
                    "end_year": end_year,
                    "chunk_index": idx,
                }
            )

    if not vectors:
        raise ValueError(f"No Grampian embeddings found under {data_root}")

    return np.vstack(vectors), pd.DataFrame(rows)


def assign_era(start_year: int) -> str:
    if 2004 <= start_year <= 2009:
        return "2004-2009"
    if 2010 <= start_year <= 2018:
        return "2010-2018"
    if 2019 <= start_year <= 2021:
        return "2019-2021"
    if 2022 <= start_year <= 2025:
        return "2022-2025"
    raise ValueError(f"Unhandled year: {start_year}")


def compute_tsne(vectors: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    # PCA initialization stabilizes the projection and trims noise before t-SNE.
    reduced = PCA(n_components=min(50, vectors.shape[1]), random_state=seed).fit_transform(vectors)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        max_iter=1000,
        verbose=0,
    )
    return tsne.fit_transform(reduced)


def compute_era_medoids(vectors: np.ndarray, meta_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for era, era_idx in meta_df.groupby("era").groups.items():
        idx = np.asarray(list(era_idx), dtype=int)
        era_vectors = vectors[idx]
        center = era_vectors.mean(axis=0, keepdims=True)
        dists = np.sum((era_vectors - center) ** 2, axis=1)
        best_local = int(np.argmin(dists))
        best_global = int(idx[best_local])
        rows.append({"era": era, "medoid_index": best_global})
    return pd.DataFrame(rows)


def plot_projection(
    coords_df: pd.DataFrame,
    era_medoids_df: pd.DataFrame,
    out_path: Path,
    point_size: float,
    alpha: float,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    era_order = ["2004-2009", "2010-2018", "2019-2021", "2022-2025"]
    era_colors = {
        "2004-2009": "#355C7D",
        "2010-2018": "#6C8EAD",
        "2019-2021": "#C06C84",
        "2022-2025": "#F2A65A",
    }
    present_eras = [era for era in era_order if era in set(coords_df["era"])]

    for era in present_eras:
        era_df = coords_df.loc[coords_df["era"] == era]
        ax.scatter(
            era_df["tsne_x"],
            era_df["tsne_y"],
            s=point_size,
            alpha=alpha,
            linewidths=0.0,
            color=era_colors[era],
            label=era,
        )

        if len(era_df) >= 3:
            pts = era_df[["tsne_x", "tsne_y"]].to_numpy()
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            ax.fill(
                hull_pts[:, 0],
                hull_pts[:, 1],
                color=era_colors[era],
                alpha=0.06,
                zorder=0,
            )
            ax.plot(
                np.append(hull_pts[:, 0], hull_pts[0, 0]),
                np.append(hull_pts[:, 1], hull_pts[0, 1]),
                color=era_colors[era],
                alpha=0.45,
                linewidth=1.0,
                zorder=1,
            )

    era_centroids = coords_df.loc[era_medoids_df["medoid_index"]].copy()
    ax.scatter(
        era_centroids["tsne_x"],
        era_centroids["tsne_y"],
        s=165,
        marker="D",
        c=era_centroids["era"].map(era_colors),
        edgecolors="black",
        linewidths=0.9,
        zorder=4,
    )

    for _, row in era_centroids.iterrows():
        ax.text(
            float(row["tsne_x"]) + 0.9,
            float(row["tsne_y"]) + 0.9,
            str(row["era"]),
            fontsize=9,
            ha="left",
            va="bottom",
        )

    num_docs = int(coords_df["doc_id"].nunique())
    ax.set_title(f"t-SNE of Chunk Embeddings Across {num_docs} NHS Grampian Reports", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.15, linewidth=0.5)
    ax.legend(title="Narrative era", loc="upper right", frameon=True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_era_centroids(coords_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    era_order = ["2004-2009", "2010-2018", "2019-2021", "2022-2025"]
    era_colors = {
        "2004-2009": "#355C7D",
        "2010-2018": "#6C8EAD",
        "2019-2021": "#C06C84",
        "2022-2025": "#F2A65A",
    }
    present_eras = [era for era in era_order if era in set(coords_df["era"])]

    centroids = coords_df.groupby("era", as_index=False)[["tsne_x", "tsne_y"]].mean()
    for _, row in centroids.iterrows():
        era = row["era"]
        if era not in present_eras:
            continue
        ax.scatter(
            row["tsne_x"],
            row["tsne_y"],
            s=170,
            marker="D",
            color=era_colors[era],
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
            label=era,
        )
        ax.text(
            float(row["tsne_x"]) + 0.8,
            float(row["tsne_y"]) + 0.8,
            era,
            fontsize=10,
            ha="left",
            va="bottom",
        )

    num_docs = int(coords_df["doc_id"].nunique())
    ax.set_title(f"Era Centroids in t-SNE Space Across {num_docs} NHS Grampian Reports", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.18, linewidth=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Narrative era", loc="best", frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    vectors, meta_df = load_embeddings(args.data_root)
    meta_df = meta_df.assign(era=meta_df["start_year"].map(assign_era))
    coords = compute_tsne(vectors, seed=args.seed, perplexity=args.perplexity)
    era_medoids_df = compute_era_medoids(vectors, meta_df)
    coords_df = meta_df.assign(
        tsne_x=coords[:, 0],
        tsne_y=coords[:, 1],
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    coords_df.to_csv(args.out_csv, index=False)
    plot_projection(coords_df, era_medoids_df, args.out_png, point_size=args.point_size, alpha=args.alpha)
    plot_era_centroids(coords_df, args.out_era_centroids_png)

    print(f"Wrote {args.out_png}")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_era_centroids_png}")
    print(f"Points projected: {len(coords_df)}")


if __name__ == "__main__":
    main()
