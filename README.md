# Abi Framework
> Experimental Zig framework that provides a bootstrap runtime and a curated set of feature modules for AI experiments.

[![Zig Version](https://img.shields.io/badge/Zig-0.16.0--dev-orange.svg)](https://ziglang.org/builds/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Version-0.1.0a-purple.svg)](CHANGELOG.md)

## Project status

`abi` is not a full-stack product yet. The current executable initialises the framework, emits a textual summary of the configured
modules, and exits. The value of the repository lies in the reusable modules under `src/` that you can import from your own
applications.

The `0.1.0a` prerelease focuses on:

- providing consistent imports such as `@import("abi").ai` and `@import("abi").database`
- documenting the bootstrap CLI accurately
- establishing a truthful changelog for the initial prerelease

## Getting started

### Prerequisites

- **Zig** `0.16.0-dev.457+f90510b08` (see `.zigversion` for the authoritative toolchain)
- A C++ compiler for Zig's build dependencies

### Clone and build

```bash
git clone https://github.com/donaldfilimon/abi.git
cd abi
zig build
zig build test
```

The default build produces `zig-out/bin/abi`. Running the executable prints a summary of enabled features:

```bash
./zig-out/bin/abi
```

Sample output:

```
ABI Framework bootstrap complete
• Features: ai, database, gpu, monitoring, web, connectors
• Plugins: discovery disabled (configure via abi.framework)
```

### Using the library from Zig

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), .{});
    defer abi.shutdown(&framework);

    // Load the lightweight agent prototype.
    const Agent = abi.ai.agent.Agent;
    var agent = try Agent.init(gpa.allocator(), .{ .name = "QuickStart" });
    defer agent.deinit();

    const reply = try agent.process("Hello", gpa.allocator());
    defer gpa.allocator().free(@constCast(reply));
}
```

The top-level module now re-exports the major feature namespaces for convenience:

- `abi.ai` – experimental agents and model helpers
- `abi.database` – WDBX vector database components and HTTP/CLI front-ends
- `abi.gpu` – GPU utilities (currently CPU-backed stubs)
- `abi.web` – minimal HTTP scaffolding used by the WDBX demo
- `abi.monitoring` – logging and metrics helpers shared across modules
- `abi.connectors` – placeholder for third-party integrations
- `abi.wdbx` – compatibility namespace exposing the database module and helpers
- `abi.VectorOps` – SIMD helpers re-exported from `abi.simd`

Refer to the `docs/` directory for API references that are generated from the Zig sources.

## Development workflow

- Format code with `zig fmt .`
- Run the full test suite with `zig build test`
- Use `zig build run` to execute the bootstrap binary under the debug configuration

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting issues and proposing changes.

## License

MIT License – see [LICENSE](LICENSE).
