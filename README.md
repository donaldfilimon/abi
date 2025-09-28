# ABI Framework
> A Zig runtime that focuses on feature toggles, plugin discovery, and a
> lightweight bootstrap executable.

[![Zig Version](https://img.shields.io/badge/Zig-0.16.0--dev-orange.svg)](https://ziglang.org/builds/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

ABI provides a small but structured runtime for orchestrating feature flags and
plugins. The library is written entirely in Zig and is designed to be embedded
into other applications or exercised through the bundled CLI. Rather than
marketing a large feature set, the goal is to offer well-tested building blocks
that demonstrate how to wire together allocators, plugin registries, and the new
Zig streaming writer APIs.

---

## Features

- **Runtime orchestration** – Enable or disable feature groups at runtime and
  iterate over the active set for diagnostics.
- **Plugin registry** – Track search paths, discover shared objects, and lazily
  load them into the running process.
- **Bootstrap CLI** – A tiny executable that initialises the framework and
  prints a summary using the streaming writer introduced in Zig 0.16.
- **Documentation-ready build** – `zig build docs` emits compiler generated
  documentation that mirrors the source layout.

These pieces intentionally stay modest and heavily commented so they can serve
as reference material for other Zig projects.

---

## Getting Started

### Tooling
- Install the Zig version listed in [`.zigversion`](.zigversion) (currently
  `0.16.0-dev.427+86077fe6b`).
- A recent LLVM toolchain is required when building on Windows.

### Clone & Build
```bash
git clone https://github.com/donaldfilimon/abi.git
cd abi
zig build
```

### Helpful Targets
```bash
zig build test        # run the unit test suite
zig build run         # execute the bootstrap binary
zig build docs        # write compiler docs to zig-out/docs
zig build fmt         # format source files in place
```

The resulting executable lives at `zig-out/bin/abi`.

---

## Usage Overview

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), .{ .auto_discover_plugins = false });
    defer framework.deinit();

    try framework.writeSummary(std.io.getStdOut().writer());
}
```

The runtime exposes helpers for managing plugin search paths, loading
discovered artefacts, and toggling features:

```zig
const feature = abi.framework.config.Feature.distributed_tracing;
if (!framework.isFeatureEnabled(feature)) {
    _ = framework.enableFeature(feature);
}

try framework.addPluginPath("./plugins");
try framework.refreshPlugins();
```

---

## Project Layout

```text
abi/
├── src/            # Library code and CLI entrypoint
├── tests/          # Unit tests
├── docs/           # Static documentation site (Jekyll compatible)
├── tools/          # Developer utilities
└── zig-out/        # Build artefacts and generated docs
```

---

## Contributing

Issues and pull requests are welcome. Please run `zig build test` and
`zig fmt src tests build.zig` before submitting changes so CI stays green. For
documentation tweaks, `zig build docs` regenerates the compiler output inside
`zig-out/docs` which can be previewed locally with a static file server.

The project is released under the [MIT license](LICENSE).

