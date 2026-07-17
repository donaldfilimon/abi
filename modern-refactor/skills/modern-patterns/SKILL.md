---
name: modern-patterns
description: Concrete before/after modern idioms for errors, types, modularity, and Zig 0.17 patterns in ABI. Use to apply clean modern patterns during refactors.
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

Base directory for this skill: /Users/donaldfilimon/abi/modern-refactor/skills/modern-patterns
Relative paths in this skill (e.g., references/) are relative to this base directory.
