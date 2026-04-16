# Contributing Documentation

This file describes the conventions for writing and contributing docs for the ABI project.

## Doc structure
- Top-level docs live in `docs/`.
- Tutorials live in `docs/tutorials/`.

## Doc comment style for Zig
- Summary line (1 sentence)
- Longer description paragraph(s)
- `Parameters:` list for function inputs
- `Returns:` description
- `Errors:` list (if applicable)

Example:

// Compute the answer to life.
//
// Parameters:
//   - allocator: allocator to use
//   - input: the input string
//
// Returns: Zig string containing the answer

## How to preview docs
- Docs are maintained as markdown. There is no automated generator in this scaffold.
- To preview locally: open the markdown file in your editor or use a static site generator of your choice.

## Formatting
- Run project formatter: `./build.sh fix`

## Submitting docs
- Create a branch and open a PR referencing the relevant issue.
- Include examples and steps to validate any code snippets.
