# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Retargeted the build guard, metadata, and deployment docs to require Zig 0.16.0-dev (master).

## [0.1.0a] - 2025-09-20

### Added
- Framework runtime that derives feature toggles, manages plugin search paths, and coordinates plugin lifecycle operations.
- Cross-platform plugin loader and registry for discovering, loading, and managing plugins.
- Minimal AI agent implementation with persona management, bounded history, and validation helpers for safe message processing.
- File-backed WDBX vector database with WAL support, header validation, and HNSW indexing scaffolding for approximate search.
