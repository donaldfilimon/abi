# Task Plan - Refactor docs/

## Objective
Refactor `docs/index.js` to reduce duplication and improve maintainability without changing search behavior.

## Checklist
- [x] Identify repetitive patterns in docs search result construction.
- [x] Introduce helper(s) to normalize result object creation.
- [x] Keep ranking/filtering behavior unchanged.
- [x] Run a quick syntax/behavioral sanity check for `docs/index.js`.
- [x] Add a short review summary with verification notes.

## Review
- Added `addResult(results, query, score, payload)` to centralize result append conditions and score assignment.
- Replaced repeated `if (!q || score > 0) results.push(...)` blocks for modules, symbols, commands, guides, plans, and roadmap entries.
- Verified syntax correctness with `node --check docs/index.js`.
