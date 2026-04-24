# Evaluation Protocol

This note defines the intended thesis-facing evaluation protocol for the current repository.

## Core Rule

Use the same statements everywhere:

- tuning and ablations are exploratory
- frozen bundle outputs are thesis-facing
- demo and visualization tooling are not the basis of headline claims

## Tracked Config Roles

- `configs/retrieval_tuning_thesis_5docs_q50.yaml`
  Use for broader tuning-style exploration across the 5 Grampian evaluation documents.
- `configs/retrieval_tuning_minilm_cap_5docs.yaml`
  Use as the promoted final chunking-selection config that feeds the frozen thesis rebuild bundle.
- `configs/retrieval_tuning_224_56_5docs.yaml`
  Use as the focused 224/56 sanity comparison config.

## Corpus Scope

The thesis-facing evaluation set is the 5-document Grampian corpus with tracked `eval_set.json` files:

- `Grampian-2020-2021`
- `Grampian-2021-2022`
- `Grampian-2022-2023`
- `Grampian-2023-2024`
- `Grampian-2024-2025`

Older partial `data_processed/Grampian-*` folders are out of scope for thesis headline metrics unless explicitly stated otherwise.

## Methodological Caveat

The tracked configs in this repo reuse the same 5-document corpus for tuning-style exploration and final promoted runs.

That means the thesis should not imply a clean train/dev/test separation across distinct document sets unless such a split was created outside these tracked configs.

The defensible wording is:

- settings were explored on the 5-document evaluation corpus
- final numbers were taken from a frozen bundle produced from the promoted final config
- this reuse of the same document set is a limitation and should be stated explicitly

## Canonical Evidence

Treat these as the main thesis-facing evidence artifacts:

- `results/thesis_rebuild_freeze/thesis_rebuild_freeze_smoke_2026-03-18/`
- `results/reproducibility/grampian_5docs_repro.json`
- `results/reproducibility/retrieval_parity_batch_smoke.json`

## Audit Script

Regenerate the protocol audit with:

```bash
python scripts/audit_evaluation_protocol.py
```

This writes:

- `results/reproducibility/evaluation_protocol_audit.json`
- `results/reproducibility/evaluation_protocol_audit.md`

Use that audit to support the methodology section and to catch obvious protocol drift before submission.
