# How One Question Is Answered

## What happens when a user asks a question?

**1. Turn the question into a numeric fingerprint**

The system converts the question into a mathematical representation that captures its meaning.

**2. Compare it against all stored text chunks**

Each chunk from the reports has already been converted into the same kind of fingerprint in advance.

**3. Find the most similar chunks**

FAISS is the search engine that quickly finds the chunks whose meaning is closest to the question.

**4. Recover the original report context**

The system maps those matches back to the real report text, page numbers, and section titles.

**5. Combine semantic search with keyword search**

It does not rely on meaning alone. It also checks direct word overlap to improve accuracy.

**6. Rank the best evidence**

The strongest candidate chunks are ordered so the most useful evidence appears first.

**7. Generate the answer from the top evidence**

The answer is produced only from the highest-ranked report passages, with page-linked citations.

## Plain-English Summary

`The system turns the question into a meaning-based fingerprint, finds the most similar report passages, checks them against keyword matches, ranks the best evidence, and then answers using only that evidence.`
