---
name: abi-plugin-system
description: Review ABI's 16 compiled-in plugin fixtures, manifest validation, manual bundled registry, and real CLI/MCP run dispatch without claiming dynamic loading.
---

# ABI plugin system

Use this skill for the compiled Rust plugin fixtures under
`crates/abi-plugins/plugins/`. ABI does not provide an `/abi-plugin-system`
command, dynamic marketplace installation, sandboxing, or hot reload.

## Real public paths

```bash
./tools/cargo.sh build -p abi-cli
./target/debug/abi plugin list
./target/debug/abi plugin run example-plugin "test input"
```

MCP exposes the frozen `plugin_list` and `plugin_run` tools. The CLI listing is
alphabetical while MCP preserves bundled declaration order; their renderings
and metadata shapes intentionally differ even though both dispatch through the
same compiled plugin substrate.

There are no public `validate`, `generate`, or `info` plugin subcommands.
Manifest/parity validation runs through tests and the repository driver:

```bash
.agents/skills/plugin-runtime-tester/plugins.sh
./tools/cargo.sh test -p abi-plugins
```

## Compiled registry

- `crates/abi-plugins/src/lib.rs` contains the manually maintained `BUNDLED`
  declarations and registry descriptors.
- Each plugin directory contains `abi-plugin.json`, `mod.rs`, and `stub.rs`.
- Tests check the declarations against manifests and module/stub parity.
- A manifest `target_feature` is descriptive metadata, not a dynamic Cargo
  gate enforced by `plugin run`.
- `entry_point` validation rejects unsafe paths and requires a relative `.rs`
  entry beneath the plugin directory.

The current fixtures are `accelerator-plugin`, `ai-plugin`, `example-plugin`,
`example-wdbx-plugin`, `foundationmodels-plugin`, `gpu-plugin`, `hash-plugin`,
`metrics-plugin`, `mlir-plugin`, `mobile-plugin`, `nn-plugin`,
`os-control-plugin`, `sea-plugin`, `shader-plugin`, `telemetry-exporter`, and
`tui-plugin`.

## Manifest commands and context providers

Manifest `commands[]` and `context_providers[]` are descriptors reachable
through internal `__cmd__:<name>` and `__context__:<name>` plugin-manager
dispatch. They are not automatically registered as slash commands or injected
contexts in `abi agent tui`.

## Validation boundary

- Registry presence does not prove execution; run all 16 fixtures plus an
  unknown-plugin negative through the runtime driver.
- Compiled-in Rust modules are not sandboxed third-party extensions.
- Do not claim generated registry state, symmetric CLI/MCP rendering, dynamic
  feature enforcement, or hot loading.
