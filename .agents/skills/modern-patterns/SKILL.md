---
name: modern-patterns
description: This skill should be used when the user asks what a modern idiom looks like for error handling, types, or modularity in Zig — e.g. 'show me the modern way to do this', 'what's the idiomatic Zig 0.17 pattern here' — while implementing a refactor.
---

# Modern Patterns

Concrete before/after modern idioms for errors, types, modularity, and other constructs.

## Error Handling

**Legacy**
Broad catches, silent ignores.

**Modern**
Explicit error unions, log or propagate. No silent empty catches in data/inference/persistence paths.

## Types

**Legacy**
String unions or magic values.

**Modern**
Enums, tagged unions, strong types.

## Modularity

Extract pure core, push effects to edges. Small focused units.

See references for catalog of before/after in Zig context.

## Additional Resources

- `references/patterns-catalog.md`
- `examples/before-after-zig.md`

Base directory for this skill: /Users/donaldfilimon/abi/.agents/skills/modern-patterns
Relative paths in this skill (e.g., references/) are relative to this base directory.
