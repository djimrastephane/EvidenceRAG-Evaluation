# Repository Layout

Current canonical layout:

- `src/`: importable project code
- `scripts/`: entrypoints and analysis scripts
- `tests/`: test suite
- `configs/`: tracked configs and examples
- `config/`: local untracked runtime config area
- `docs/`: architecture notes, slides, experiment notes, and documentation figures
- `Data/`: raw source PDFs
- `data_processed/`: canonical processed corpus outputs
- `data_variants/`: alternate processed corpora used for ablations, refreshes, or comparisons
- `runs/`: intermediate experiment/run outputs
- `results/`: promoted reporting outputs
- `results/ablations/`: promoted ablation summaries, selections, and comparison artifacts

Legacy areas still present:

- `archive/legacy_experiments/Experiment/`: historical notebooks and exploratory work
- `figures/`: local-only scratch figure output area, ignored by git

Recommended placement for new files:

- new code: `src/` or `scripts/`
- new docs/slides: `docs/`
- new experiment notes: `docs/experiments/`
- new machine-readable configs: `configs/`
- new one-off run outputs: `runs/`
- new tracked documentation figures: `docs/architecture/`
- new final charts/tables: `results/`
- new ablation summaries/selections: `results/ablations/`
