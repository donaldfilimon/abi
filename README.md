# Abi Framework
> Experimental Zig framework that provides a bootstrap runtime and a curated set of feature modules for AI experiments.

[![Zig Version](https://img.shields.io/badge/Zig-0.16.0-orange.svg)](https://ziglang.org/)
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
- capturing the broader modernization roadmap documented in [`docs/MODERNIZATION_BLUEPRINT.md`](docs/MODERNIZATION_BLUEPRINT.md)

## Getting started

### Prerequisites

- **Zig** `0.16.0` (see `.zigversion` for the authoritative toolchain)
- A C++ compiler for Zig's build dependencies

### Clone and build

```bash
git clone https://github.com/donaldfilimon/abi.git
cd abi
zig build
zig build test
zig build fmt     # format sources
zig build docs    # generate docs
zig build bench   # run benchmarks
zig build tools   # run developer tools entrypoint
zig build check   # format + tests aggregate
```

The default build produces `zig-out/bin/abi`. This executable now implements a full sub‑command based CLI. Use `abi --help` to view available commands and `abi <subcommand> --help` for detailed usage.

```bash
# Show help (lists all sub‑commands)
./zig-out/bin/abi --help

# Run the benchmark suite
zig build bench -- --format=markdown --output=results

# Run developer tools entrypoint
zig build tools -- --help

# Example: list enabled features in JSON mode
./zig-out/bin/abi features list --json
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

The top‑level module still re‑exports the major feature namespaces for convenience, and the new CLI provides runtime control of these features:

- `abi.ai` – experimental agents and model helpers
- `abi.database` – WDBX vector database components and HTTP/CLI front-ends
- `abi.gpu` – GPU utilities (currently CPU-backed stubs)
- `abi.web` – minimal HTTP scaffolding used by the WDBX demo
- `abi.monitoring` – logging and metrics helpers shared across modules
- `abi.connectors` – placeholder for third-party integrations
- `abi.wdbx` – compatibility namespace exposing the database module and helpers
- `abi.VectorOps` – SIMD helpers re-exported from `abi.simd`

- **CLI documentation** – see `docs/CLI_USAGE.md` for a comprehensive list of sub‑commands, flags, and examples.  
- **API references** – see `docs/MODULE_REFERENCE.md` (generated from the Zig sources).  
- **Project structure** – see `docs/PROJECT_STRUCTURE.md` for an overview of the repository layout.

## Development workflow

- Format code with `zig fmt .`
- Run the full test suite with `zig build test`
- Use `zig build run` to execute the bootstrap binary under the debug configuration

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting issues and proposing changes.

## License

MIT License – see [LICENSE](LICENSE).
