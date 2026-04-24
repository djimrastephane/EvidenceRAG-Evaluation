from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thesis_rag.config import load_config


def test_load_config_prefers_explicit_pipeline_normalization_flag(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  project_root: .",
                "embedding:",
                "  model_name: sentence-transformers/all-MiniLM-L6-v2",
                "  apply_l2_normalization: true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.embedding.apply_l2_normalization is True


def test_load_config_accepts_legacy_normalize_embeddings_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  project_root: .",
                "embedding:",
                "  model_name: sentence-transformers/all-MiniLM-L6-v2",
                "  normalize_embeddings: false",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.embedding.apply_l2_normalization is False
