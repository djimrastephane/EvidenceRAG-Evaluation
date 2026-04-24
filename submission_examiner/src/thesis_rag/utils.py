from __future__ import annotations

"""General utilities shared across thesis_rag stages.

These helpers cover deterministic runtime setup, logging, JSON persistence,
stable ordering, and dependency checks. They are kept small and explicit so the
pipeline's operational assumptions are easy to audit.
"""

import importlib
import json
import logging
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def now_utc_iso() -> str:
    """Return a filesystem-friendly UTC timestamp string without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def configure_logging(log_path: Path, level: str = "INFO") -> None:
    """Configure file and stderr logging for a single pipeline run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def set_global_determinism(seed: int, deterministic_torch: bool) -> None:
    """Set Python, NumPy, and Torch determinism controls for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def resolve_device(requested: str) -> str:
    """Resolve the requested runtime device to an available Torch device."""
    try:
        import torch
    except Exception:
        return "cpu"
    desired = requested.strip().lower()
    if desired == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if desired == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if desired == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return "cpu"
    return desired or "cpu"


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Apply row-wise L2 normalization to an embedding matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON to disk, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write newline-delimited JSON records to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    """Read a JSON file from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit_hash(project_root: Path) -> str | None:
    """Return the current git commit hash when the project is in a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def make_run_dir(runs_dir: Path, prefix: str = "run") -> Path:
    """Create a timestamped run directory inside the configured runs root."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = runs_dir / f"{timestamp}_{prefix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def dependency_check(required: dict[str, str]) -> None:
    """Validate that required Python modules are importable and version-aligned."""
    missing: list[str] = []
    mismatched: list[str] = []
    for module_name, expected_version in required.items():
        try:
            module = importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
            continue
        actual = getattr(module, "__version__", None)
        if expected_version and actual and actual != expected_version:
            mismatched.append(f"{module_name}=={actual} (expected {expected_version})")
    if missing or mismatched:
        parts: list[str] = []
        if missing:
            parts.append(f"missing: {', '.join(sorted(missing))}")
        if mismatched:
            parts.append(f"version mismatch: {', '.join(sorted(mismatched))}")
        raise RuntimeError("Dependency validation failed: " + "; ".join(parts))


def stable_unique(values: list[int]) -> list[int]:
    """Remove duplicates while preserving first-seen order."""
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered
