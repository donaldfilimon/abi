# WDBX 3D-spatial + hybrid-retrieval example

A runnable demonstration of WDBX's 3D spatial index combined with semantic
search: `demo.zig` inserts 200 synthetic 3D points, each carrying a small
embedding, then runs `abi.features.wdbx.retrieval.hybridSpatialSearch` to
rank candidates by a blend of embedding-cosine-similarity and 3D proximity
to a query center.

This is a reference example over WDBX's existing store/index APIs, not a
new subsystem -- see `src/features/wdbx/spatial_3d.zig` (3D index) and
`src/features/wdbx/retrieval.zig` (`hybridSpatialSearch`) for the
underlying implementation.

## Running

```bash
zig build run-example-3d-hybrid
```

## Sample output

```
hybridSpatialSearch: top 10 of 200 points (semantic+spatial, 0.5/0.5 blend)
  1. id=59 score=0.8284 (semantic=0.9699 spatial=0.6869) point=(12.42,6.79,-1.77)
  2. id=197 score=0.8045 (semantic=0.8516 spatial=0.7575) point=(7.75,-0.36,7.87)
  3. id=187 score=0.7660 (semantic=0.7622 spatial=0.7698) point=(3.39,-6.35,7.63)
  ...
```

(Output is reproducible run-to-run -- the PRNG seed in `demo.zig` is fixed
at `42`; change it to explore other point clouds.)
