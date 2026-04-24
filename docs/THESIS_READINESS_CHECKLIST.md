# Thesis Readiness Checklist

Use this as a final pre-submission guardrail for the current repo.

## 1. Canonical Pipeline

- [ ] Confirm the thesis claims rely on the evaluated hybrid pipeline in `scripts/retrieval_eval_hybrid.py`.
- [ ] Confirm the live service in `src/rag_pdf/services/search_service.py` is aligned with that evaluated path or clearly described as demo-only.
- [ ] Re-check that recent API/UI changes did not reintroduce retrieval drift.

## 2. Environment

- [ ] Run thesis-facing work from `rag-pipeline`, not Anaconda `base`.
- [ ] Run `python scripts/check_environment.py --strict` before canonical reruns.
- [ ] Keep `environment.yml` and `requirements.txt` aligned.
- [ ] Pin any new package before using it for thesis outputs.

## 3. Frozen Artifacts

- [ ] Identify the exact frozen run or bundle that thesis numbers come from.
- [ ] Preserve commit hash, environment manifest, and runbook for that run.
- [ ] Avoid rebuilding canonical `data_processed` roots after fixing thesis numbers.
- [ ] Prefer frozen bundle artifacts over ad hoc local outputs for figures and tables.

Repo anchors:
- `docs/thesis_rebuild_freeze_workflow.md`
- `scripts/setup_thesis_rebuild_freeze.py`
- `scripts/export_thesis_rebuild_bundle.py`
- `scripts/check_thesis_bundle_drift.py`

## 4. Evaluation Hygiene

- [ ] State clearly which query sets were used for tuning.
- [ ] State clearly which query sets were used for final reporting.
- [ ] Check that final “best settings” were not selected on the same final test set without saying so.
- [ ] Keep the link between ablations, final selection, and frozen thesis numbers explicit.

## 5. Corpus Discipline

- [ ] Decide what to do with partial historical corpora under `data_processed/`.
- [ ] Ensure thesis-facing scripts only operate on evaluation-ready corpora.
- [ ] Prevent mixed partial and complete corpora from affecting comparisons.

Current known risk:
- Older `Grampian-*` folders are partial and not evaluation-complete.

## 6. Reproducibility

- [ ] Keep at least one reproducibility report for the final thesis pipeline.
- [ ] Confirm repeated runs over frozen inputs are identical or acceptably stable.
- [ ] Preserve reproducibility outputs in the examiner-facing bundle where possible.

Repo anchors:
- `scripts/check_pipeline_reproducibility.py`
- `results/reproducibility/...`

## 7. Failure Analysis

- [ ] Keep FP1-FP7 analysis tied to canonical outputs.
- [ ] Be able to state which failure classes remain dominant after final tuning.
- [ ] Include representative examples, not just counts.
- [ ] Present the taxonomy as evidence-backed analysis.

Repo anchors:
- `scripts/run_current_pipeline_fp1_fp7.py`
- `results/thesis_rebuild_freeze/.../failure_analysis/...`

## 8. Limitations

- [ ] State remaining top-1 retrieval miss behavior explicitly.
- [ ] State subsection or heading quality variability where relevant.
- [ ] State corpus scope limits, including incomplete historical folders if out of scope.
- [ ] State that demo and visualization tooling is not the basis of headline claims unless explicitly tied to frozen evaluated outputs.

## 9. Demo Separation

- [ ] Remove or clearly label any display-only or simulated UI control.
- [ ] Keep WIZMAP and embedding diagnostics separate from thesis-evaluated metrics.
- [ ] Make it obvious what is evaluated output versus demo output versus visualization overlay.

Repo anchors:
- `app/ui/streamlit_app.py`
- `scripts/export_wizmap_umap.py`
- `scripts/serve_wizmap_local.py`

## 10. Security Posture

- [ ] Keep localhost defaults for examiner or local demo use.
- [ ] Require `API_KEY` for widened-origin or non-local deployment.
- [ ] Keep uploads disabled unless explicitly needed.
- [ ] Avoid exposing filesystem paths through the API.

Repo anchors:
- `app/api/main.py`
- `UPLOAD_ENABLED`
- `DEMO_MODE`
- `API_KEY_POLICY`

## 11. Provenance

- [ ] Make every thesis figure and table traceable to one script or one frozen artifact.
- [ ] Avoid hand-edited intermediate CSVs for final reported results unless tracked and justified.
- [ ] Keep input CSV or JSON next to exported figures where practical.

Repo anchors:
- `scripts/plot_*`
- `scripts/make_*`
- `scripts/generate_*`
- `scripts/export_thesis_*`

## 12. Examiner Usability

- [ ] Have one short examiner run path that works without guesswork.
- [ ] Keep the startup sequence explicit: activate env, run preflight, run evaluation or demo.
- [ ] Make sure the examiner submission manifest matches what you actually intend to share.

Repo anchors:
- `docs/EXAMINER_SUBMISSION_MANIFEST.md`
- `scripts/build_examiner_submission.sh`
- `results/thesis_rebuild_freeze/.../RUNBOOK.md`

## Minimum Bar

- [ ] Pinned environment passes strict preflight.
- [ ] Final thesis numbers come from a frozen run or bundle.
- [ ] Reproducibility evidence is preserved.
- [ ] Tuning versus final reporting protocol is defensible and documented.
- [ ] Residual limitations are written up explicitly.
- [ ] Examiner-facing instructions are tested once from a clean shell.
