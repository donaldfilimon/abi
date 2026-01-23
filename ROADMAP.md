# ABI Framework Roadmap

> Last updated: 2026-01-22 | Current version: 0.6.0

## Vision

ABI is a production-grade Zig 0.16 framework for AI systems, combining GPU acceleration,
vector database, and distributed compute. Our focus is developer experience, performance,
and correctness.

---

## Now (v0.7.0)

Active development. In progress or immediately queued.

### Code Quality
- [ ] GPU codegen consolidation - GLSL refactor ([plan](docs/plans/2026-01-22-gpu-codegen-consolidation.md))
- [ ] Observability module consolidation ([plan](docs/plans/2026-01-17-refactor-phase2.md))

### Features
- [ ] Task management system - CLI + persistence ([plan](docs/plans/2026-01-17-task-management-system.md))

### Documentation
- [ ] Archive completed plans to docs/plans/archive/

---

## Next (v0.8.0)

Scoped for upcoming release. Not yet started.

### Ecosystem
- [ ] Python bindings expansion (beyond foundation)
- [ ] npm package for WASM bindings

### Features
- [ ] Streaming inference API improvements
- [ ] Multi-model orchestration

### Tooling
- [ ] VS Code extension for ABI development
- [ ] Interactive benchmark dashboard

---

## Later (2026-2027)

Strategic direction. May require RFCs.

### Research
- [ ] Hardware acceleration (FPGA, ASIC exploration)
- [ ] Novel vector index structures
- [ ] Zig std.gpu native integration (when stable)

### Ecosystem
- [ ] Cloud function adapters (AWS Lambda, GCP, Azure)
- [ ] Kubernetes operator

### Community
- [ ] RFC process formalization
- [ ] Contributor certification program

---

## Not Planned

Items explicitly deprioritized:

- **GUI toolkit** - Out of scope; ABI is backend-focused
- **JavaScript runtime** - WASM bindings cover this use case
- **Legacy Zig support** - 0.16+ only

---

## Contributing

1. Check [issues](https://github.com/donaldfilimon/abi/issues) for `help-wanted`
2. Propose features in [discussions](https://github.com/donaldfilimon/abi/discussions)
3. See [CONTRIBUTING.md](CONTRIBUTING.md)

## History

See [CHANGELOG.md](CHANGELOG.md) for release history.
