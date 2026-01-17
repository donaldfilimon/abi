# zls-lsp

Zig Language Server (ZLS) integration for Claude Code, providing code intelligence, diagnostics, and navigation.

## Supported Extensions
`.zig`, `.zon`

## Installation

### Via Scoop (Windows - Recommended for dev builds)
```powershell
# Add versions bucket for dev builds
scoop bucket add versions

# Install latest dev ZLS (matches zig-dev)
scoop install zls-dev

# Or install stable
scoop install zls
```

### Via ZVM (Zig Version Manager)
```bash
# Install zvm first
scoop install zvm

# Use zvm to manage zig versions
zvm install master
zvm use master
```

### Build from source
```bash
git clone https://github.com/zigtools/zls
cd zls
zig build -Doptimize=ReleaseSafe
```

## More Information
- [ZLS GitHub](https://github.com/zigtools/zls)
- [Zig Language](https://ziglang.org/)
