---
title: Installation
description: System requirements, toolchain setup, and building ABI from source
section: Start
order: 2
---

# Installation

This page covers everything you need to install, build, and verify ABI on your system.

## System Requirements

| Requirement | Details |
|-------------|---------|
| **Zig** | `0.16.0-dev.2611+f996d2866` or newer (pinned in `.zigversion`) |
| **Git** | Any recent version (2.30+) |
| **OS** | macOS (Apple Silicon or Intel), Linux (x86_64, aarch64), Windows (x86_64) |
| **Shell** | Bash, Zsh, or Fish (Linux / macOS); PowerShell (Windows) |
| **Optional** | GPU drivers (CUDA, Vulkan, or Metal) for hardware-accelerated compute |
| **Optional** | Docker for containerized deployment |

ABI requires an exact Zig nightly version. The pinned version is recorded in the
`.zigversion` file at the repository root. Using a different Zig version will likely
produce compilation errors.

## Installing Zig

### Recommended: zvm (Zig Version Manager)

[zvm](https://github.com/marler182/zvm) lets you install and switch between multiple
Zig versions. This is the recommended approach because ABI tracks the Zig master branch.

```bash
# If you don't have zvm yet, follow https://github.com/marler182/zvm#installation

# Update zvm itself to the latest release
zvm upgrade

# Install the latest Zig master nightly
zvm install master

# Activate the master toolchain
zvm use master
```

After activation, verify the version matches the pinned requirement:

```bash
which zig
# Expected: ~/.zvm/bin/zig

zig version
# Expected: 0.16.0-dev.2611+f996d2866 (or newer)

cat .zigversion
# Should match your zig version output
```

**Shell PATH precedence.** If `which zig` points to a different location (for example
`~/.local/bin/zig` or `/usr/local/bin/zig`), you need to ensure `~/.zvm/bin` comes
first in your `PATH`. Add this to your shell profile (`.bashrc`, `.zshrc`, etc.):

```bash
export PATH="$HOME/.zvm/bin:$PATH"
```

Then restart your shell or run `source ~/.bashrc`.

### Manual Install from ziglang.org

If you prefer not to use zvm, download a matching nightly build directly:

1. Go to [ziglang.org/download](https://ziglang.org/download/)
2. Find the **master** section
3. Download the archive for your platform
4. Extract it and add the `zig` binary to your `PATH`

Make sure the version string matches what is in `.zigversion`.

## Verifying Your Toolchain

ABI includes a toolchain doctor script that checks for common setup issues:

```bash
zig run tools/scripts/toolchain_doctor.zig
```

This script verifies:

- `zig` is on your `PATH` and executable
- The active Zig version matches `.zigversion`
- There are no conflicting Zig installations shadowing `~/.zvm/bin/zig`
- `build.zig` and documentation reference the same version

You can also run the toolchain doctor as a build step:

```bash
zig build toolchain-doctor
```

### Version Consistency Check

To verify that `.zigversion`, `build.zig`, and documentation all reference the same
Zig version:

```bash
zig run tools/scripts/check_zig_version_consistency.zig
```

## Cloning the Repository

```bash
git clone https://github.com/your-org/abi.git
cd abi
```

## Building

Build ABI with default flags. All features are enabled except `mobile`:

```bash
zig build
```

The first build compiles the framework, CLI, and all enabled feature modules. Subsequent
builds are incremental and much faster.

### Verify the Build

```bash
# Print the framework version
zig build run -- version

# Show system info and which features are compiled in
zig build run -- system-info
```

## Running Tests

ABI maintains two test suites with enforced baselines.

### Main Test Suite

```bash
zig build test --summary all
```

Expected: **1270 pass, 5 skip** (1275 total).

### Feature Inline Tests

```bash
zig build feature-tests --summary all
```

Expected: **1534 pass** (1534 total).

Both baselines must be maintained when contributing changes.

## Full Validation (Optional)

For a comprehensive check before committing:

```bash
# Format + tests + feature tests + flag validation + CLI smoke tests
zig build full-check

# Or the extended version (adds examples, benchmarks, WASM check, Ralph gate)
zig build verify-all
```

## Troubleshooting

### Wrong Zig version

If you see compilation errors mentioning unknown builtins or changed standard library
APIs, your Zig version likely does not match `.zigversion`. Run:

```bash
zig version
cat .zigversion
```

If they differ, install the correct version with `zvm install master && zvm use master`.

### PATH conflicts

Multiple Zig installations can shadow each other. Check for conflicts:

```bash
which -a zig
```

If more than one path appears, ensure `~/.zvm/bin` is first in your `PATH`.

### macOS Gatekeeper

On macOS, the first run of a downloaded Zig binary may be blocked by Gatekeeper. Allow
it in System Settings > Privacy & Security, or run:

```bash
xattr -d com.apple.quarantine $(which zig)
```

## Next Steps

- [Getting Started](getting-started.html) -- first build, first test, first example
- [Architecture](architecture.html) -- understand the module hierarchy
- [Configuration](configuration.html) -- all build flags and environment variables

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
