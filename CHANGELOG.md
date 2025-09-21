# Changelog

All notable changes to the Abi AI Framework are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[![Version](https://img.shields.io/badge/version-0.1.0a-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Table of Contents

- [Unreleased](#unreleased)
- [0.1.0a](#010a---2025-01-18)
- [0.1.0-alpha](#010-alpha---2024-12-12)
- [Roadmap](#roadmap)
- [Links](#links)
- [Changelog Guidelines](#changelog-guidelines)

---

## Unreleased

> Changes merged to `main` after the `0.1.0a` tag.

### Planned / In Progress
- Harden the plugin loader handshake with integration tests that cover mismatched ABI versions.
- Expand the feature toggle documentation with CLI examples and environment override guidance.
- Wire the docs generator into the CI workflow so that previews are produced on pull requests.

---

## 0.1.0a - 2025-01-18

> First structured preview release after the alpha milestone.

### Added
- Bootstrapped the Zig workspace with `build.zig`, dependency manifests, and a layered module layout for features, tools, and tests.
- Introduced compile-time feature toggles and configuration surfaces so experimental subsystems can be enabled or omitted per target.
- Delivered plugin system scaffolding, including a versioned ABI constant, loader stubs, and a template demonstrating how third-party plugins register capabilities.
- Brought up initial developer tooling: command-line entry points, documentation generator hooks, and smoke tests that exercise the bootstrap pipeline.

### Changed
- Normalized configuration defaults and environment overrides to keep the bootstrap experience consistent across platforms.
- Documented release packaging steps and contributor expectations for adding new toggles or plugins.

### Fixed
- Resolved early dependency resolution issues in the build graph when optional modules are disabled.
- Stabilized plugin teardown paths uncovered during bootstrap testing.

---

## 0.1.0-alpha - 2024-12-12

> Initial public alpha showcasing the core ideas that shaped the framework.

### Added
- Seeded the repository with the earliest AI agent experiments, a prototype vector store, and placeholder networking layers.
- Shipped the first draft of the CLI along with minimal examples demonstrating how to issue chat and embedding requests.
- Added skeletal documentation, contribution notes, and sanity tests to establish project conventions.

### Known Limitations
- GPU, WebAssembly, and advanced model backends were stubs only; they were disabled by default pending the bootstrap work in `0.1.0a`.
- Plugin lifecycle management required manual cleanup and lacked version negotiation.
- CI coverage focused on style checks and smoke tests, not exhaustive validation.

---

## Roadmap

### Near Term
- [ ] Publish sample plugins that exercise the ABI negotiation contract.
- [ ] Provide feature toggle matrices that map to packaging profiles (server, cli-only, embedded).
- [ ] Expand integration tests around configuration hot-reload.

### Longer Term
- [ ] Graduate the vector database prototype into a supported storage layer.
- [ ] Introduce distributed execution helpers for larger deployments.
- [ ] Ship polished web tooling and dashboard integrations.

---

## Links

- **[Unreleased]**: [Compare v0.1.0a...HEAD](https://github.com/yourusername/abi/compare/v0.1.0a...HEAD)
- **[0.1.0a]**: [Release v0.1.0a](https://github.com/yourusername/abi/releases/tag/v0.1.0a)
- **[0.1.0-alpha]**: [Release v0.1.0-alpha](https://github.com/yourusername/abi/releases/tag/v0.1.0-alpha)

---

## Changelog Guidelines

### Change Categories
- **Added** – new features and capabilities.
- **Changed** – updates to existing functionality.
- **Fixed** – bug fixes and reliability improvements.
- **Security** – security-related changes.
- **Deprecated** – features slated for removal.
- **Removed** – functionality that has been removed.

### Entry Expectations
Every entry should briefly describe the change, note its impact, and call out migration steps when behavior changes.

---

**This changelog is maintained by the Abi AI Framework team. For the latest updates, visit the [GitHub releases](https://github.com/yourusername/abi/releases) page.**
