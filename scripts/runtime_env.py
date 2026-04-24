"""Runtime environment inspection utilities for the thesis pipeline.

Provides functions to probe installed Python packages, external binaries, and overall
runtime provenance. Consumed by check_examiner_path.py, check_environment.py, and the
setup_thesis_rebuild_freeze.py workflow to produce consistent environment snapshots. Not
intended to be run directly; import the functions from other scripts.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


DEFAULT_PACKAGES = [
    "camelot",
    "faiss",
    "ghostscript",
    "numpy",
    "pandas",
    "pdfplumber",
    "pyarrow",
    "pymupdf",
    "sentence_transformers",
    "tiktoken",
    "torch",
]

DEFAULT_COMMANDS = ["gs", "pdftoppm", "tesseract"]

DIST_NAME_OVERRIDES = {
    "camelot": "camelot-py",
    "faiss": "faiss-cpu",
    "pymupdf": "PyMuPDF",
    "sentence_transformers": "sentence-transformers",
}


def _distribution_name(module_name: str) -> str:
    return DIST_NAME_OVERRIDES.get(module_name, module_name)


def module_status(module_name: str) -> dict[str, Any]:
    """Return installation status and version for a single Python package."""
    spec = importlib.util.find_spec(module_name)
    installed = spec is not None
    version = None
    error = None
    if installed:
        dist_name = _distribution_name(module_name)
        try:
            version = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            version = None
        except Exception as exc:  # pragma: no cover - defensive metadata fallback
            error = str(exc)
    return {
        "installed": installed,
        "version": version,
        "module": module_name,
        "distribution": _distribution_name(module_name),
        "error": error,
    }


def command_status(command_name: str) -> dict[str, Any]:
    """Return availability and resolved path for a command-line binary."""
    path = shutil.which(command_name)
    return {
        "available": path is not None,
        "path": path,
    }


def dependency_report(
    packages: list[str] | None = None,
    commands: list[str] | None = None,
) -> dict[str, Any]:
    """Return a combined installation report for all specified packages and binaries."""
    package_names = packages or list(DEFAULT_PACKAGES)
    command_names = commands or list(DEFAULT_COMMANDS)
    modules = {name: module_status(name) for name in package_names}
    binaries = {name: command_status(name) for name in command_names}
    return {
        "modules": modules,
        "commands": binaries,
    }


def collect_runtime_provenance(
    packages: list[str] | None = None,
    commands: list[str] | None = None,
) -> dict[str, Any]:
    """Collect a full runtime snapshot including Python version, platform, conda env, and dependency report."""
    report = dependency_report(packages=packages, commands=commands)
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cwd": str(Path.cwd()),
        "conda_prefix": os.getenv("CONDA_PREFIX"),
        "conda_default_env": os.getenv("CONDA_DEFAULT_ENV"),
        "virtual_env": os.getenv("VIRTUAL_ENV"),
        "dependency_report": report,
    }


def critical_environment_checks() -> list[dict[str, Any]]:
    """Return a list of pass/fail checks for the packages and binaries that the pipeline cannot run without."""
    report = dependency_report()
    modules = report["modules"]
    commands = report["commands"]
    checks = [
        {
            "name": "tiktoken_available",
            "ok": bool(modules["tiktoken"]["installed"]),
            "detail": "Required for exact preprocessing token counts.",
        },
        {
            "name": "camelot_available",
            "ok": bool(modules["camelot"]["installed"]),
            "detail": "Required for the structured table extraction branch.",
        },
        {
            "name": "ghostscript_python_wrapper_available",
            "ok": bool(modules["ghostscript"]["installed"]),
            "detail": "Needed by Camelot on many environments.",
        },
        {
            "name": "ghostscript_binary_available",
            "ok": bool(commands["gs"]["available"]),
            "detail": "Ghostscript binary used by Camelot.",
        },
        {
            "name": "faiss_available",
            "ok": bool(modules["faiss"]["installed"]),
            "detail": "Required for dense index build and retrieval.",
        },
        {
            "name": "sentence_transformers_available",
            "ok": bool(modules["sentence_transformers"]["installed"]),
            "detail": "Required for MiniLM embeddings.",
        },
        {
            "name": "torch_available",
            "ok": bool(modules["torch"]["installed"]),
            "detail": "Required backend for sentence-transformers.",
        },
    ]
    return checks


def pinned_requirements_status(requirements_path: Path) -> list[dict[str, Any]]:
    """Compare installed package versions against the pinned versions in requirements.txt."""
    statuses: list[dict[str, Any]] = []
    if not requirements_path.exists():
        return statuses
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        dist_name, expected_version = [part.strip() for part in line.split("==", 1)]
        try:
            installed_version = metadata.version(dist_name)
            installed = True
        except metadata.PackageNotFoundError:
            installed_version = None
            installed = False
        statuses.append(
            {
                "distribution": dist_name,
                "expected_version": expected_version,
                "installed": installed,
                "installed_version": installed_version,
                "matches": installed and installed_version == expected_version,
            }
        )
    return statuses
