# Example: parallel extract before cutover

**Context:** A large handler file mixes dispatch, formatting, and validation.

## Before (conceptual)

```text
handlers/foo.zig  (~800 LOC)
  parseArgs → validate → run → formatOutput → write
```

## After (strangler / extract)

```text
handlers/foo.zig          # thin dispatch only
handlers/foo_validate.zig # pure validation (unit-tested)
handlers/foo_format.zig   # pure formatting (unit-tested)
```

## Steps

1. Add characterizing tests on public CLI/MCP path.
2. Move pure validation into a sibling module; call from old path.
3. Run focused tests + `./build.sh check`.
4. Move formatting next; re-run gates.
5. Only after parity, shrink the original file further.

Do **not** change frozen command/tool names while extracting.
