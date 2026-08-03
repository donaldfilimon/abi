# Contributing Documentation

This file describes conventions for writing and contributing docs for the ABI project.

## Doc structure

- Published Mintlify docs live in `docs/` (see [`docs/README.md`](README.md)).
- Tutorials and research notes live in `docs/tutorials/` and `docs/research/`.

## Rustdoc style

- Start with a one-sentence summary.
- Add details only when the contract or invariants are not obvious from the
  signature.
- Use `# Errors`, `# Panics`, and `# Safety` sections when they apply.
- Keep examples executable where practical; use `no_run` only when the example
  requires an external service or operator-owned path.

Example:

```rust
/// Computes a deterministic embedding for local retrieval.
///
/// # Errors
///
/// Returns [`EmbeddingError::EmptyInput`] when `input` is empty.
pub fn embed(input: &str) -> Result<Vec<f32>, EmbeddingError> {
    // implementation
}
```

## How to preview docs

- Mintlify site: `cd docs && npx mint@latest dev`
- CI validation: `.agents/skills/docs-validate/validate.sh`

## Validation

- Format Rust: `./tools/cargo.sh fmt --all`.
- Run the primary repository gate: `./tools/check.sh`.
- After changing `docs/`, run `.agents/skills/docs-validate/validate.sh` with an
  LTS Node runtime.

## Submitting docs

- Create a branch and open a PR referencing the relevant issue.
- Include examples and steps to validate any code snippets.
- Follow claim boundaries in [`contracts/external-claims-audit.mdx`](contracts/external-claims-audit.mdx).
