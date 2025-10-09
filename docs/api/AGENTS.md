# Repository Guidelines

This document provides a quick reference for contributors working on the **ABI** Zig framework. Follow these rules to keep the codebase clean, tests reliable, and releases predictable.

## Project Structure \u0026 Module Organization

```
src/          # Public Zig modules (abi.*)
│   mod.zig   # Re‑exports for the `abi` namespace
│   comprehensive_cli.zig  # Modern CLI implementation
│
framework/    # Core runtime, configuration, and state
features/     # Feature sub‑packages: ai, database, gpu, web, monitoring, connectors
shared/       # Common utilities, logging, platform helpers
core/         # Collections and low‑level helpers
examples/     # Runnable demos that compile with ``zig build``
benchmarks/   # Benchmarks that produce JSON metrics
tests/        # Mirrored feature tree with *_test.zig files
docs/         # Markdown reference, guides, and API docs
docker/       # Dockerfile, compose, and Helm charts
scripts/      # CI helpers and release tooling
deps/         # Exact pinning in build.zig.zon
```

## Build, Test, and Development Commands

```
zig build                 # Compile the CLI (abi) and tests
zig build test            # Run the unit test suite
zig build bench           # Execute benchmarks and output JSON
zig build docs            # Generate the Markdown API reference

# Docker
docker build -t abi-cli .   # Build the release image
docker run -it abi-cli --help   # Inspect CLI options
```

All commands run under the current workspace; no external package manager is required because Zig manages dependencies automatically.

## Coding Style \u0026 Naming Conventions

* **Indentation** – 4‑space tabs (``tab`` in VSCode). No trailing whitespace.
* **CamelCase** – Public functions and types use ``PascalCase`` (e.g., ``Agent``, ``DatabaseConfig``). 
* **snake_case** – Local variables, errors, and file names (``error{NotFound}``).
* **Zig fmt** – Run ``zig fmt --check .`` before committing.
* **Error handling** – Prefer Zig's ``error{}`` sets over sentinel values.

## Testing Guidelines

Zig's built‑in ``std.testing`` framework is used. Test files are named ``*_test.zig`` and live next to the module they exercise.

```
zig build test --filter \u003cmodule\u003e
```

Test files should exercise both success and failure paths. Coverage should be \u003e90 %; add new tests when a bug is fixed.

## Commit \u0026 Pull Request Guidelines

* **Commit messages** – Follow Conventional Commits: ``feat: add GPU backend``, ``fix: typo in docs``, ``chore: update deps``.
* **PR description** – Briefly describe the change, reference any relevant issue (``Closes #123``), and list new or updated tests.
* **Review** – Aim for two approvals. Keep diff size \u003c50 files per PR.
* **CI** – All CI checks must pass before merging.

## Security \u0026 Configuration Tips

* Keep all external dependencies pinned in ``build.zig.zon``.
* Use the ``-Denable-gpu``, ``-Denable-web``, and ``-Denable-monitoring`` flags to toggle experimental features.
* Configuration is supplied via command‑line options or a JSON file using the ``-c \u003cpath\u003e`` flag; the CLI validates JSON schema before use.

Feel free to open an issue if you encounter any ambiguities or want to propose a new feature.

