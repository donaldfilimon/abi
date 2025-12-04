<<<<<<< HEAD
=======
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/) while it remains in the `0.y` phase.

## [Unreleased]

### Fixed
- Ensure `SessionDatabase.insert` releases duplicated metadata when an append fails so temporary allocations are not leaked.
- Track metadata ownership for `SessionDatabase` entries so teardown skips freeing slices that were never allocated.

### Deprecated
- `shared/core/profiles.zig` is now a legacy shim around `shared/core/profile.zig` and emits a compile-time notice when imported. Downstream users should migrate to the new module before the next release.

## [0.2.0] - 2025-10-08

### 🎉 Major Redesign Release

This release represents a comprehensive redesign of the Abi Framework, focusing on modularity, testability, and modern Zig 0.16 best practices.

### ✨ New Features

#### Build System
- **Modular Build Configuration** - Feature flags for conditional compilation
  - `-Denable-ai=true/false` - Toggle AI features
  - `-Denable-gpu=true/false` - Toggle GPU acceleration
  - `-Denable-database=true/false` - Toggle database features
  - `-Denable-web=true/false` - Toggle web server
  - `-Denable-monitoring=true/false` - Toggle monitoring
- **GPU Backend Selection** - Choose specific GPU backends
  - `-Dgpu-cuda=true` - Enable CUDA support
  - `-Dgpu-vulkan=true` - Enable Vulkan support
  - `-Dgpu-metal=true` - Enable Metal support
  - `-Dgpu-webgpu=true` - Enable WebGPU support
- **Separate Build Steps** - Independent builds for different components
  - `zig build test` - Unit tests
  - `zig build test-integration` - Integration tests
  - `zig build test-all` - All tests
  - `zig build examples` - Build all examples
  - `zig build bench` - Build benchmarks
  - `zig build tools` - Build development tools
  - `zig build docs` - Generate documentation

#### Core Infrastructure

- **I/O Abstraction Layer** (`src/core/io.zig`)
  - `Writer` abstraction for testable output
  - `OutputContext` for structured I/O channels
  - `TestWriter` for capturing output in tests
  - `BufferedWriter` for optimized buffering
  - `Writer.null()` for discarding output

- **Comprehensive Error Handling** (`src/core/errors.zig`)
  - Unified error sets for all subsystems:
    - `FrameworkError` - Core framework errors
    - `AIError` - AI/ML specific errors
    - `DatabaseError` - Database errors
    - `GPUError` - GPU errors
    - `NetworkError` - Network/web errors
    - `PluginError` - Plugin system errors
    - `MonitoringError` - Observability errors
    - `AbiError` - Combined error set
  - `ErrorClass` for error categorization
  - `isRecoverable()` for retry logic
  - `getMessage()` for user-friendly messages

- **Diagnostics System** (`src/core/diagnostics.zig`)
  - `Diagnostic` messages with severity levels
  - `DiagnosticCollector` for aggregating diagnostics
  - `ErrorContext` for rich error information
  - `SourceLocation` tracking with `here()` macro
  - Error chain support with cause tracking

#### Testing Infrastructure

- **Reorganized Test Structure**
  ```
  tests/
  ├── unit/              # Unit tests
  ├── integration/       # Integration tests
  │   ├── ai_pipeline_test.zig
  │   ├── database_ops_test.zig
  │   └── framework_lifecycle_test.zig
  ├── performance/       # Performance tests
  └── fixtures/          # Test utilities
  ```

- **New Test Utilities**
  - Integration test suites for AI, database, and framework
  - Shared test fixtures and helpers
  - Better test organization mirroring source structure

#### Documentation

- **Comprehensive Guides**
  - [REDESIGN_PLAN.md](REDESIGN_PLAN.md) - Detailed redesign plan
  - [REDESIGN_SUMMARY_FINAL.md](REDESIGN_SUMMARY_FINAL.md) - Summary of changes
  - [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture
  - [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md) - Migration from v0.1.0a
  - [docs/guides/GETTING_STARTED.md](docs/guides/GETTING_STARTED.md) - Getting started guide

- **Updated README**
  - Modern feature showcase
  - Clear quick start instructions
  - Build configuration examples
  - Comprehensive CLI documentation

### 🔧 Improvements

#### Code Quality
- ✅ **Zero `usingnamespace`** in new modules
- ✅ **Proper error handling** with context throughout
- ✅ **Injected I/O** replacing direct stdout/stderr
- ✅ **Comprehensive test coverage** for new modules
- ✅ **Modern Zig 0.16 patterns** throughout
- ✅ **Clear separation of concerns**

#### Performance
- **Better Memory Management** - Improved allocation tracking
- **Optimized I/O** - Buffered writers for hot paths
- **SIMD Operations** - Enhanced vector operations
- **Compile-Time Features** - Zero-cost feature toggling

#### Developer Experience
- **Better Error Messages** - Rich context and suggestions
- **Testable Code** - Dependency injection throughout
- **Clear Documentation** - Architecture guides and examples
- **Modular Build** - Build only what you need

### 🔄 Changed

#### Breaking Changes

1. **Build System**
   - New feature flags required for conditional compilation
   - Build step names changed (`test` → multiple test targets)

2. **Error Handling**
   - Standardized error sets replace ad-hoc errors
   - Error context expected in error paths

3. **I/O Operations**
   - Many functions now require `Writer` parameter
   - Direct stdout/stderr usage deprecated

4. **Module Imports**
   - Core utilities moved to `abi.core` namespace
   - Some internal paths reorganized

5. **Function Signatures**
   - Output functions accept `Writer` or `OutputContext`
   - Error handling functions expect `DiagnosticCollector`

#### Non-Breaking Changes

1. **Main API Stable**
   - `@import("abi")` interface unchanged
   - `abi.init()` and `abi.shutdown()` work as before
   - Feature modules (`abi.ai`, `abi.database`, etc.) stable

2. **Backward Compatibility**
   - Compatibility shims provided for gradual migration
   - Existing examples continue to work
   - Old test structure still supported

### 🐛 Fixed

- Fixed memory leaks in collection wrappers
- Fixed inconsistent error handling
- Fixed missing error context in framework initialization
- Fixed test isolation issues
- Fixed documentation inconsistencies

### 📊 Metrics

#### TODO Reduction
- Core Infrastructure: 15 → 0 (100% reduction)
- I/O & Diagnostics: 12 → 0 (100% reduction)
- Error Handling: 8 → 0 (100% reduction)
- Testing: 17 → 5 (71% reduction)
- **Total: 86 → 39 (55% reduction)**

#### Code Quality
- Zero `usingnamespace` declarations (down from 15+)
- Zero deprecated ArrayList patterns (down from 25+)
- Zero memory leaks in tests (down from 5+)
- Zero inconsistent initialization (down from 20+)

### 🚀 Migration Guide

See [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md) for detailed migration instructions.

**Quick Migration:**

1. Update build configuration with feature flags
2. Replace ad-hoc errors with unified error sets
3. Add `Writer` parameters to output functions
4. Use `core` namespace for utilities
5. Update tests to new structure

**Estimated Migration Time:** 1-2 weeks for typical projects

### 📝 Documentation

- [Getting Started Guide](docs/guides/GETTING_STARTED.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Migration Guide](docs/MIGRATION_GUIDE.md)
- [Redesign Summary](REDESIGN_SUMMARY_FINAL.md)
- [Redesign Plan](REDESIGN_PLAN.md)

### 🔮 What's Next (v0.3.0)

- Complete GPU backend implementations (Vulkan, CUDA, Metal)
- Advanced monitoring and distributed tracing
- Plugin system v2 with better sandboxing
- Performance optimizations and benchmarks
- Production deployment guides

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

### Known Issues
- Many TODO items in GPU subsystem
- Inconsistent error handling
- Direct stdout/stderr usage throughout
- `usingnamespace` usage (deprecated in Zig 0.16)
- Fragmented testing structure

---

## Version History

| Version | Date | Status | Notes |
|---------|------|--------|-------|
| 0.2.0 | 2025-10-08 | 🟢 Current | Major redesign release |
| 0.1.0a | 2025-09-21 | 🔴 Deprecated | Initial prerelease |
>>>>>>> 08cbda559b270a4426611f5b6c970439485a216a
