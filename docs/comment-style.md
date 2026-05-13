# Comment Style Guide

Use comments to explain contracts a reader cannot infer safely from names alone:
ownership, lifetime, invariants, determinism, thread-safety, numerical domains,
and failure behavior.

## Public Headers

- Start each public header with `@file` and `@brief`.
- Add a short module overview when the header introduces a major concept.
- Document every public function with `@brief`, parameters, return value, and
  ownership/lifetime notes where relevant.
- Mark experimental APIs with a visible `@warning Experimental API.` paragraph.
- For public structs with raw pointers, say whether each pointer is borrowed,
  caller-owned, callee-owned, optional, or invalidated by an update call.

## Source Files

- Prefer comments before complex blocks, not before every statement.
- Explain numerical shortcuts, branch thresholds, fallback paths, and why a
  fast path is valid.
- Keep implementation notes close to the code they justify.

## Roadmap Items

- Public headers should describe current behavior.
- Move future work to `/docs/roadmap.md` or issue tracking.
- If an implementation TODO must remain in source, include the reason and the
  condition that will let someone remove it.
