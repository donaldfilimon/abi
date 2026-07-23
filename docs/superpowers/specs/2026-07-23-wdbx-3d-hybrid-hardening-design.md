# WDBX 3D-Spatial + Hybrid-Retrieval Hardening — Design

## Context

A request came in to build WDBX (ABI's vector+spatial store) from scratch as a
"production-quality" system with a first-class 3D index, a `ComputeBackend`
abstraction, a runnable 3D+embedding hybrid-retrieval example, and documented
cross-compilation. Before writing any code, `src/features/wdbx/` was audited
against those four priorities: it already has ~11,700 lines covering WAL,
segments, HNSW, a 3D spatial module, a compute-backend abstraction backed by
real Metal/CPU kernels, clustering, compression, and REST — a from-scratch
rewrite would discard substantial working, tested code and contradicts the
repo's own no-mass-rewrite, no-unproven-claims constraints (`AGENTS.md`).

This spec instead targets the **three gaps the audit actually found**, leaves
everything else untouched, and keeps the existing reference-grade framing —
no "production-quality" language enters the docs.

## Audit findings (what's real vs. already done)

- **Already satisfies "ComputeBackend abstraction"**: `compute.zig`'s
  `Backend`/`Capability`/`Selection` + `src/features/gpu/` (Metal kernels +
  CPU fallback). No changes proposed.
- **Gap 1 — no real spatial index structure**: `SpatialIndex3D.radiusSearch`
  and `.nearestNeighbors` (`spatial_3d.zig:135-160`) linear-scan a flat
  `ArrayListUnmanaged(SpatialRecord3D)`. O(n) per query.
- **Gap 2 — no combined semantic+3D query**: `retrieval.zig`'s `hybridSearch`
  (line 31) blends semantic + temporal + causal signals but never touches
  `SpatialIndex3D`. There is no query that ranks by both cosine similarity
  and 3D proximity together.
- **Gap 3 — no runnable 3D+embedding example, no cross-compile docs**: no
  `examples/` entry demonstrates the 3D+embedding path; `README.md` has no
  cross-compilation section (verified: no `-Dtarget` mentions).

## Approach

Additive, backward-compatible hardening of the existing module. No new
top-level module, no CLI surface change (the frozen 13-command CLI is
untouched), no public API removal.

### Component 1 — Octree inside `SpatialIndex3D` (closes Gap 1)

`SpatialIndex3D`'s public surface (`init`, `initWithPool`, `deinit`, `insert`,
`count`, `radiusSearch`, `nearestNeighbors`) does not change — `store.zig`
and `stub.zig`, its only callers, need zero edits.

Internally:
- New `Octree` type in `spatial_3d.zig`: 8-way spatial subdivision.
- **Rebuild-on-write**: inserts append to the existing flat `records` list
  and mark the tree dirty. The next `radiusSearch`/`nearestNeighbors` call
  rebuilds the tree first if dirty — bounds are computed fresh from all
  current points (min/max per axis with a small epsilon pad so boundary
  points aren't excluded by floating-point edge cases).
- **Small-N fallback**: below a `OCTREE_MIN_POINTS = 64` threshold, both
  query methods use the existing linear scan directly — no tree build
  overhead for small point sets, and it doubles as the correctness oracle
  in tests.
- Distance math (`calculateDistance`, `euclideanDistance`, etc.) is reused
  unchanged — 3-component points don't benefit meaningfully from `@Vector`
  SIMD lanes, so no change there.

### Component 2 — `hybridSpatialSearch` (closes Gap 2)

New function in `retrieval.zig`, following the existing `hybridSearch` /
`hybridSearchScoped` pattern (same file, same store/allocator conventions):

```zig
pub const RankedSpatialResult = struct {
    id: u32,
    vector: ?[]const f32,
    point: spatial_3d.Point3D,
    score: f32,
    semantic_score: f32,
    spatial_score: f32,
};

pub fn hybridSpatialSearch(
    allocator: std.mem.Allocator,
    store: *const wdbx_mod.Store,
    query_vector: []const f32,
    center: spatial_3d.Point3D,
    limit: usize,
    weight_semantic: f32,
    weight_spatial: f32,
) ![]RankedSpatialResult
```

`score = weight_semantic * cosine_similarity(query_vector, record.vector) +
weight_spatial * (1 - normalized_distance)`, where `normalized_distance` is
each candidate's Euclidean distance from `center` divided by the max
distance among candidates (0 when max distance is 0, matching the existing
temporal-decay normalization style already in `temporal.zig`). Results are
sorted descending by `score`.

Weight `1.0/0.0` recovers pure-semantic ranking; `0.0/1.0` recovers pure-3D
ranking — this is the acceptance test for the blending math.

### Component 3 — Runnable example (closes Gap 3, first half)

`examples/wdbx_3d_hybrid/`, following the exact convention of
`examples/multiway/` (README + fixture/demo file, no new CLI subcommand):

- `demo.zig` — standalone program: creates a `Store`, inserts ~200
  synthetic 3D points each with an attached embedding, calls
  `hybridSpatialSearch`, prints ranked results with both score components.
- `README.md` — mirrors `examples/multiway/README.md`'s format: what the
  demo does, how to run it, sample output.
- Wired into `build.zig` as a `run-example-3d-hybrid` step (same shape as
  the existing `benchmarks` step) so `./build.sh full-check` can smoke-test
  it (exit-0 check only, same pattern as `dashboard-smoke`).

### Component 4 — Cross-compilation docs (closes Gap 3, second half)

A new README section listing `zig build -Dtarget=x86_64-linux-gnu`,
`aarch64-macos`, `x86_64-windows-gnu`. Each target is actually cross-compiled
once locally before the doc line is written — no command goes in undemonstrated.

## Testing strategy

- **Octree correctness**: property test — for N random points (below and
  above `OCTREE_MIN_POINTS`, both uniform and clustered distributions),
  `nearestNeighbors(k)` and `radiusSearch(r)` must return the **same result
  set** (by id) as the pre-existing linear-scan path, used as the oracle.
- **`hybridSpatialSearch`**: unit tests in the style of `retrieval.zig`'s
  existing `hybridSearch` tests (lines 185-275) — weight edge cases
  (1.0/0.0 and 0.0/1.0) plus a "spatially near beats semantically distant
  under high spatial weight" case.
- **Example**: `run-example-3d-hybrid` build step asserts exit 0.
- **Cross-compile docs**: each documented target is built once as part of
  implementing the task, not just claimed.

## Claims discipline

No doc language changes to "production-quality" or similar unproven framing
anywhere in this work. WDBX stays described as reference-grade, consistent
with `AGENTS.md`'s existing policy. Every capability claim added to docs
ties to a test or a build step that was actually run.

## Out of scope

- Any change to `compute.zig` / `src/features/gpu/` (already satisfies the
  ComputeBackend priority).
- Octree deletion/update support (WDBX's existing insert-only pattern for
  spatial records doesn't need it; adding it would be speculative).
- A new CLI subcommand for spatial/hybrid queries (frozen 13-command surface).
- Voxel grid or k-d tree alternatives (octree was the explicit choice for
  Component 1).
