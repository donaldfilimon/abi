# Documentation Layout

The ABI documentation tree is split into a few tracked buckets:

- [`docs/_docs/`](./_docs) for guide-style Markdown pages
- [`docs/api/`](./api) for generated API reference Markdown
- [`docs/plans/`](./plans) for project plans
- [`docs/data/`](./data) for docs-site data artifacts
- [`docs/index.html`](./index.html) for the local docs entrypoint

`zig build gendocs` is the canonical generator entrypoint, and `zig build check-docs` verifies deterministic output and policy compliance.
