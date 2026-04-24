# FP2 Reduction Experiment Plan

Use this plan to reduce `FP2_MISSED_TOP_RANK` in a controlled way.

## Goal

Prioritize retrieval-stage improvements that reduce top-rank misses before changing generation behavior.

## Experiment 1: FP2 Triage

Question:

`Why is the top-ranked chunk wrong?`

Input:

- Current `FP2_MISSED_TOP_RANK` cases from the per-query failure analysis CSV.

Label each sampled FP2 with one subtype:

- `semantic_neighbor`
- `table_lost_to_prose`
- `wrong_chunk_same_page`
- `section_mismatch`
- `lexical_miss`
- `unit_or_entity_miss`
- `other`

Success criterion:

- At least 70-80% of sampled FP2 cases can be assigned to a subtype.
- One or two subtypes dominate the sample.

Output:

- `results/fp2_triage/fp2_triage_sample.csv`
- Counts by subtype

## Experiment 2: Numeric/Table Rerank Bias

Question:

`Can simple rerank bias improve top-1 for numeric and table-heavy questions?`

Change:

- Add a score bonus on top-N candidates when:
- `answer_type=number`
- Candidate `is_table=True`
- Candidate includes overlapping unit or currency cues
- Candidate includes strong financial/numeric terms relevant to the question

Evaluate:

- Overall `Hit@1`
- Numeric-query `Hit@1`
- Table-evidence-query `Hit@1`
- Total `FP2` count

Success criterion:

- Overall `Hit@1` improves by at least `+0.01`, or
- `FP2` decreases materially without harming other subsets

Output:

- Before/after metrics table
- Before/after `FP2` counts
- Changed-query audit CSV

## Experiment 3: Section/Subsection Rerank

Question:

`Are top-rank misses driven by local structural mismatch?`

Change:

- Add a small rerank bonus when `section_title` or `subsection_title` aligns with query cues.

Evaluate:

- Overall `Hit@1`
- Total `FP2` count
- Per-document `Hit@1`
- Improved-to-hit count
- Regressed-from-hit count

Success criterion:

- Net positive movement in hit rate
- Especially useful if `section_mismatch` or `wrong_chunk_same_page` was common in Experiment 1

Output:

- Per-document comparison CSV
- Changed-query audit CSV

## Experiment 4: Limited Window Expansion

Question:

`Can small adjacent-context expansion reduce FP3 after retrieval is mostly correct?`

Change:

- Expand top-1 or top-2 chunk with one adjacent chunk on either side
- Preferably only when retrieval margin is low or for selected query types

Evaluate:

- `FP3` count
- End-to-end `HIT` count
- Latency `p50/p95`
- Context size inflation

Success criterion:

- `FP3` decreases
- End-to-end `HIT` improves
- Latency increase is acceptable

Output:

- `FP3` before/after summary
- Latency before/after summary

## Core Metrics

Capture these for every experiment:

- `Hit@1`
- `Hit@3`
- `MRR@10`
- Total `HIT`
- `FP2` count
- `FP3` count
- `FP4` count
- Changed queries
- Improved-to-hit count
- Regressed-from-hit count
- Latency if the change affects runtime

## Decision Policy

Keep a change only if:

- It reduces the targeted failure mode.
- It does not create too many regressions elsewhere.
- It is explainable in thesis terms.
- It does not materially worsen latency unless the gain clearly justifies it.

## Recommended Order

1. FP2 triage
2. Numeric/table rerank bias
3. Section/subsection rerank
4. Limited window expansion
