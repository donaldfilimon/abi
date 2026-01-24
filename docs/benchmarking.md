---
title: "Benchmarking Guide"
tags: [performance, benchmarks, testing]
---
# ABI Framework Benchmarking Guide
> **Codebase Status:** Synced with repository as of 2026-01-24.

<p align="center">
  <img src="https://img.shields.io/badge/Performance-Benchmarks-orange?style=for-the-badge" alt="Benchmarks"/>
  <img src="https://img.shields.io/badge/WDBX-6.1M_ops%2Fsec-success?style=for-the-badge" alt="WDBX"/>
  <img src="https://img.shields.io/badge/LLM-2.8M_tokens%2Fsec-blue?style=for-the-badge" alt="LLM"/>
</p>

This guide provides quick reference for using the **benchmark** command suite.

## Running benchmark suites

```bash
# Run all benchmark suites (human‑readable output)
abi bench all

# Run a specific suite
abi bench simd
abi bench memory
abi bench concurrency
abi bench database
abi bench network
abi bench crypto
abi bench ai
abi bench quick
```

## JSON output and file export

All suites accept `--json` to emit machine‑readable results and `--output <file>` to write the JSON to disk:

```bash
# Full suite, JSON output to a file
abi bench all --json --output all.json

# SIMD suite only, JSON to file
abi bench simd --json --output simd.json
```

The generated JSON follows this schema (excerpt):

```json
{
  "duration_sec": 0.08,
  "benchmarks": [
    {"name": "dot_product_64", "category": "simd", "ops_per_sec": 9.4e6, "mean_ns": 106, "iterations": 10000},
    ...
  ]
}
```

## Micro‑benchmarks

Fine‑grained performance tests are available via `micro`:

```bash
# Hash benchmark (default 1000 iterations)
abi bench micro hash

# Custom iterations and JSON output
abi bench micro alloc --iterations 2000 --json
abi bench micro parse --iterations 1500 --output parse.json
```

Supported operations:

- `hash` – simple hash computation
- `alloc` – memory allocation pattern
- `parse` – basic parsing workload
- `noop` – empty baseline (useful for overhead measurement)

## Quick benchmarks (CI)

The `quick` suite runs a minimal set of benchmarks suitable for continuous‑integration pipelines:

```bash
abi bench quick
```

## Listing available suites

```bash
abi bench list
```

## Error handling

If an unknown suite is supplied, the command prints an error and suggests `abi bench list`.

```bash
abi bench unknown
# → error: Unknown benchmark suite: unknown
#   Use 'abi bench list' to see available suites.
```

---

Feel free to explore the generated JSON files for deeper performance analysis or integrate them into your CI dashboards.
