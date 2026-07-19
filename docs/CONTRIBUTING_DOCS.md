# Contributing Documentation

This file describes conventions for writing and contributing docs for the ABI project.

## Doc structure

- Published Mintlify docs live in `docs/` (see [`docs/README.md`](README.md)).
- Tutorials and research notes live in `docs/tutorials/` and `docs/research/`.

## Doc comment style for Zig

- Summary line (1 sentence)
- Longer description paragraph(s)
- `Parameters:` list for function inputs
- `Returns:` description
- `Errors:` list (if applicable)

Example:

```zig
/// Compute the answer to life.
///
/// Parameters:
///   - allocator: allocator to use
///   - input: the input string
///
/// Returns: Zig string containing the answer
```

## How to preview docs

- Mintlify site: `cd docs && npx mint@latest dev`
- CI validation: `.agents/skills/docs-validate/validate.sh`

## Formatting

- Run project formatter: `./build.sh fix`

## Submitting docs

- Create a branch and open a PR referencing the relevant issue.
- Include examples and steps to validate any code snippets.
- Follow claim boundaries in [`contracts/external-claims-audit.mdx`](contracts/external-claims-audit.mdx).
