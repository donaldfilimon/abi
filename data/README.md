# Dashboard data

`sample_benchmarks.json` holds **synthetic placeholder data**. It exists so the
dashboard has something to render and so a benchmark exporter has a schema to
target. It is not a measurement of ABI, and nothing on the published page should
be read as a performance claim.

## Schema

An array of records, one per run:

| Field | Type | Meaning |
|-------|------|---------|
| `date` | `string` (`YYYY-MM-DD`) | Run date, used as the x-axis label |
| `p50` | `number` | Median latency, milliseconds |
| `p90` | `number` | 90th percentile latency, milliseconds |
| `p99` | `number` | 99th percentile latency, milliseconds |
| `throughput` | `number` | Operations per second |

Records are rendered in file order; the dashboard does not sort them.

## Replacing it with real numbers

The dashboard reads this file at page load with `fetch`, so publishing real
measurements only requires writing the same shape to the same path. The relevant
producers in this repo are:

- `abi wdbx benchmark [count]` — local in-process insert/search timing
  (`crates/abi-cli/src/wdbx/benchmark.rs`).
- `./tools/bench_regress.sh` — the same-system regression gate, with its frozen
  baseline in `tools/bench_baseline.json`.

Both are explicitly same-host development guards. If real numbers ever land
here, keep the page's provenance note accurate about which machine, which
toolchain, and which workload produced them — see
`docs/contracts/external-claims-audit.mdx`.
