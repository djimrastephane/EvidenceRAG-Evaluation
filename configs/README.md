# Config Layout

Tracked configuration belongs in `configs/`.

- `configs/examples/`: sample JSON configs and templates
- `configs/*.yaml`: tracked experiment and tuning configs
- `configs/batch.json`: preferred location for a tracked batch-processing config when you want to version it

Legacy/local behavior:

- `config/batch.json` is still supported by `scripts/run_batch.py`
- `config/` is best treated as a local, machine-specific override area
- private or large local-only configs should stay out of git there
