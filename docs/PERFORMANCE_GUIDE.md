# Performance Guide

This guide shows how to run and interpret the SIMD micro‑benchmarks and performance utilities in this repo.

## Quick Commands

- Run tests (optimized): `zig build test -Doptimize=ReleaseFast`
- Run CLI: `zig build run`
- Run SIMD micro-benchmark: `zig build bench-simd`

## SIMD Micro-Benchmark

The micro-benchmark (`benchmarks/simd_micro.zig`) measures hot path vector ops:

- Vector ops: add, multiply, scale, normalize
- Reductions: sum, mean, variance, stddev
- Distances: L1 (Manhattan), dot product
- Matrix multiply: (256x64) × (64x64)

Output is per-op elapsed time in nanoseconds. Use relative comparisons across commits/runs for regression detection.

### Interpreting Results

- Add/multiply/scale: Expect near memory bandwidth-bound performance; significant regressions usually indicate alignment or loop stride issues.
- Normalize: Sensitive to sqrt; micro-architectural changes can skew outcomes—watch for consistent deltas.
- Reductions (sum/mean/variance/stddev): Expect vector-width scaling; tail handling should be negligible.
- Dot/L1: Track end-to-end distance kernels; used heavily by DB searches.
- Matrix multiply: Small-block case indicates cache/locality; for large matrices use separate end-to-end benchmarks.

### Tips

- Always run in a quiet system; CPU frequency scaling affects results.
- Compare ReleaseFast builds only; Debug builds are not representative.
- Prefer trend analysis over single-run numbers.

## Database & End-to-End

Use `zig build perf-guard` (when configured in CI) to enforce performance thresholds. See `docs/OPTIMIZATION_SUMMARY.md` for broader optimization context.

