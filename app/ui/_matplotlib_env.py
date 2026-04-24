from __future__ import annotations

import os
from pathlib import Path


def configure_matplotlib_env() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cache_root = repo_root / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    if not os.getenv("XDG_CACHE_HOME"):
        os.environ["XDG_CACHE_HOME"] = str(cache_root)

    if not os.getenv("MPLCONFIGDIR"):
        mpl_cache_dir = cache_root / "matplotlib"
        mpl_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

    if not os.getenv("NUMBA_CACHE_DIR"):
        numba_cache_dir = cache_root / "numba"
        numba_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(numba_cache_dir)


configure_matplotlib_env()
