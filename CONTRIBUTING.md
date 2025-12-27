# Contributing

Thanks for helping improve ABI. Keep changes focused, documented, and tested.

## Development Setup
```bash
git clone <repo>
cd abi
zig build
zig build test
zig fmt .
```

## Workflow
1. Create a focused branch.
2. Make changes with clear scope.
3. Run `zig build` and `zig build test`.
4. Format with `zig fmt .`.
5. Update docs for public API changes.

## Style
- 4 spaces, no tabs, lines under 100 chars.
- PascalCase types, snake_case functions/variables.
- Explicit imports only (no `usingnamespace`).
- Use `!` return types and specific error enums.
- Prefer `defer`/`errdefer` for cleanup.

## Zig 0.16 Conventions

### Memory Management
- **Prefer `std.ArrayListUnmanaged` over `std.ArrayList`**
  - Unmanaged requires passing allocator explicitly to methods
  - Provides better control over memory ownership
  - Reduces hidden allocator dependencies

```zig
// Good
var list = std.ArrayListUnmanaged(u8).empty;
try list.append(allocator, item);
list.deinit(allocator);

// Avoid
var list = std.ArrayList(u8).init(allocator);
try list.append(item);
list.deinit();
```

### Formatting and I/O
- **Use modern format specifiers** instead of manual conversions:
  - `{t}` for enum and error values (replaces `@tagName()`)
  - `{B}` / `{Bi}` for byte sizes
  - `{D}` for durations
  - `{b64}` for base64 encoding

```zig
// Good
std.debug.print("Status: {t}\n", .{status});

// Avoid
std.debug.print("Status: {s}\n", .{@tagName(status)});
```

### Error Handling
- Use specific error sets instead of `anyerror`
- Always document when errors can occur
- Use `errdefer` for cleanup on error paths

### Testing
- Unit coverage lives in library tests and `tests/mod.zig`.
- New features must include tests or clear justification.
- Run `zig build test --summary all` to see detailed results.
