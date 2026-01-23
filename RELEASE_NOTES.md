# Release Notes

## v0.1.1 – 2026-01-23

### Highlights
- Bumped framework version to **0.1.1** (package_version updated).
- Added comprehensive GPU backend guide (`docs/gpu-backends.md`).
- Implemented real CUDA VTable stub with loadable driver handling.
- Added OpenGL, OpenGL‑ES, WebGL2 stub support (mutually exclusive warning).
- Introduced dedicated benchmark runner (`scripts/run_benchmarks.bat`).
- Captured and documented benchmark results in the README.
- Updated security guidance to reference the built‑in Secrets Manager.
- Minor documentation improvements and CLI testing guide.

### Compatibility
No breaking API changes. Existing binaries remain compatible.

### Upgrade Instructions
Simply rebuild the project with the default flags (`zig build`). All new features are enabled via existing build options.

Previous versions can be found via git tags.

© 2026 ABI Framework contributors.

