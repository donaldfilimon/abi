---
title: "CLI Smoke Test Guide"
tags: [testing, cli, automation]
---
# CLI Smoke Test Guide
> **Codebase Status:** Synced with repository as of 2026-01-24.

<p align="center">
  <img src="https://img.shields.io/badge/Tests-CLI_Smoke-blue?style=for-the-badge" alt="CLI Smoke"/>
  <img src="https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white" alt="Windows"/>
  <img src="https://img.shields.io/badge/Status-Automated-success?style=for-the-badge" alt="Automated"/>
</p>

The ABI project provides a Windows batch script `scripts/run_cli_tests.bat` that executes **all** example
commands and the three nested LLM commands (`chat`, `generate`, `list`).

## How to Run

```batch
cmd /c scripts\\run_cli_tests.bat
```

The script:

- Builds and runs each example (`hello`, `database`, `agent`, `compute`, `network`, `discord`, `llm`, `training`, `ha`, `orchestration`).
- Executes the LLM sub‑commands. Missing dummy model files will emit a **warning** but will not cause the overall run to fail.
- Reports success or failure for each command and exits with `0` on success.

## Adding New Examples

If you add a new example target, simply add a `call :run "zig build run-<example>"` line to the script.

## Common Issues

* **Model not found** – The script now tolerates this case; you will see a `WARNING` line but the test continues.
* **Build failures** – Ensure the required build options are enabled (e.g. `-Denable-llm` for LLM commands).

Running this script is part of the `full-check` step, which also runs formatting, unit tests, and benchmarks.  For a dedicated benchmark run you can also execute `scripts\\run_benchmarks.bat`.

## Additional Manual Commands

The smoke test script currently covers the existing examples and LLM commands. For the new `convert` dataset tool, run it manually as needed:

```bash
zig build run -- convert dataset --input data.bin --output data.wdbx --format to-wdbx
zig build run -- convert dataset --input data.wdbx --output data.bin --format to-tokenbin
```
