# zig-abi-plugin

OpenCode plugin for the ABI Zig framework.

## Quick Start

```bash
git clone https://github.com/donaldfilimon/abi.git && cd abi
tools/zigly --bootstrap    # Install zig + zls, symlink to ~/.local/bin, verify
```

Add to PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"  # Add to your shell profile
```

## Verify Setup

```bash
# macOS 26.4+ (Darwin 25.x) — use ./build.sh, NOT zig build
./build.sh test --summary all

# Linux or older macOS
zig build test --summary all
```

**Why `./build.sh` on macOS 26.4+?** Zig's LLD can't resolve Apple's `.tbd` files. `build.sh` relinks with Apple's `/usr/bin/ld`. Required on Darwin 25+.

## Editor

`tools/zigly --link` installs ZLS. Configure your editor to use `~/.local/bin/zls`.

## Structure

- `agents/` — Domain-specific agents (build, parity, pipeline, feature scaffolding)
- `skills/` — Skill definitions for common tasks
- `hooks/` — Pre-commit validation hooks
- `.mcp.json` — MCP server config (points to `zig-out/bin/abi-mcp`)