# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/) while it remains in the `0.y` phase.

## [Unreleased]

### Fixed
- Ensure `SessionDatabase.insert` releases duplicated metadata when an append fails so temporary allocations are not leaked.
- Track metadata ownership for `SessionDatabase` entries so teardown skips freeing slices that were never allocated.

### Deprecated
- `shared/core/profiles.zig` is now a legacy shim around `shared/core/profile.zig` and emits a compile-time notice when imported. Downstream users should migrate to the new module before the next release.

## [0.1.0a] - 2025-09-21
### Added
- Re-exported feature modules at the root (`abi.ai`, `abi.database`, `abi.gpu`, etc.) for a consistent public API.
- Introduced an `abi.wdbx` compatibility namespace that surfaces the vector database helpers and HTTP/CLI front-ends.
- Documented the intended usage of the library module and the bootstrap executable in the README.

### Changed
- Updated all version strings (library, CLI, and WDBX metadata) to report `0.1.0a`.
- Rewrote the changelog to describe the 0.1.0a prerelease instead of fabricated 1.x milestones.

### Removed
- Prior claims about fully featured CLIs, REST services, and production benchmarks that are not present in this prerelease.
