---
name: Feedback preferences
description: Code style, communication, and workflow preferences for this project
type: feedback
---

No trailing summaries at end of responses — user can see the diff/output directly.

**Why:** User preference for terse, direct communication.
**How to apply:** End responses with one short sentence stating what changed and what's next.

When removing em dashes from LaTeX, scan the entire file — not just the paragraph being edited.

**Why:** Previous session missed em dashes in the future work section after removing them from limitations. User had to ask twice.
**How to apply:** After removing em dashes in one section, grep the full file before confirming.

Do not add buzzword/AI-sounding language to thesis text. Words to avoid: leverage, nuanced, pivotal, tapestry, beacon, foster, elevate, robust (when meaning "effective"), paramount, underscore, delve, crucial, vital, realm, landscape.

**Why:** User explicitly asked for a scan-and-replace of these words in both discussion.tex and appendix.tex.
**How to apply:** When writing or editing thesis prose, avoid these words entirely.

All references to appendices in main thesis chapters should be bold: `\textbf{Appendix~\ref{app:...}}` and `\textbf{(Supplementary Material)}`.

**Why:** User requested bold appendix references for visual emphasis.
**How to apply:** When writing new thesis text with appendix references, always bold them.
