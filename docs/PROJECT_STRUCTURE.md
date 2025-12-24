# Project Structure

```
abi/
├── lib/                    # Core library sources
│   ├── core/               # Core utilities and diagnostics
│   ├── features/           # Feature modules
│   ├── framework/          # Orchestration runtime
│   └── shared/             # Shared helpers
├── tools/                  # CLI entrypoint
├── tests/                  # Smoke tests
├── docs/                   # Architecture + guides
├── deploy/                 # Deployment assets
└── python/                 # Python package bindings (experimental)
```

## Additional Details
- Examples and benchmarks are intentionally removed for production focus.
- Public entrypoints are exported from `lib/mod.zig`.
