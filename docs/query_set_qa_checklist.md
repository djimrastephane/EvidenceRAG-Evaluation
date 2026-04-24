# Query Set QA/QC Checklist

Use this checklist to review the evaluation gold set before final benchmarking.

## Per-Query Checks

- `question` asks one clear information need only.
- Question wording is unambiguous without relying on hidden context.
- `expected_pages` contains the evidence directly.
- `expected_answer` is fully supported by the cited page(s).
- `expected_answer` matches the scoring policy for the query.
- `answer_type` is correct (`number` or `text`).
- `expected_section` matches the evidence page.
- `expected_subsection` matches the evidence page.
- `evidence_layout` is correct.
- `difficulty` is consistent with the project definitions.
- `filter_hints` do not contradict the actual evidence location.

## Ambiguity Checks

Flag the query if any of the following are true:

- More than one plausible answer exists.
- The answer appears on multiple pages in different contexts.
- The question can be interpreted at different aggregation levels.
- The question uses vague references such as `this year`, `the total`, `it`, or `current`.
- The answer requires combining evidence across multiple places but is labeled as simple retrieval.

## Numeric Query Checks

Apply to all queries with `answer_type=number`.

- Units are explicit (`%`, `£`, `£000`, millions, etc.).
- Sign conventions are clear (deficit vs surplus, negative vs positive).
- Rounding and formatting are consistent with the annotation policy.
- The expected answer is free of transcription errors.
- The value is the one actually asked for, not a nearby related number.
- For table evidence, the row/column intersection is unambiguous.

## Paraphrase Checks

Apply to `_P1`, `_P2`, and similar variants.

- The paraphrase asks the same underlying information need as the base query.
- The paraphrase does not accidentally leak easier lexical cues.
- The paraphrase does not introduce a new concept or constraint.
- The expected pages remain appropriate.
- The expected answer remains appropriate.

## Dataset-Level Checks

- Category coverage is balanced enough for the claims made in the thesis.
- Difficulty labels are not concentrated in one category only.
- Evidence layouts are represented clearly.
- The set does not contain too many near-duplicate queries.
- The set does not overstate semantic difficulty with mostly lexical-match questions.

Summarize:

- Counts by category
- Counts by difficulty
- Counts by evidence layout
- Counts of `number` vs `text`
- Counts of paraphrase queries

## Recommended Review Process

1. Review all numeric queries.
2. Review all paraphrase queries.
3. Randomly sample 25-50 remaining queries.
4. Fix recurring issues across the full set.
5. Freeze the final query set before ablation comparison.

## Suggested Thesis Wording

Use or adapt:

`The evaluation set was subjected to manual QA/QC to verify question clarity, answer support, page-level evidence alignment, answer-type consistency, and paraphrase equivalence prior to final benchmarking.`
