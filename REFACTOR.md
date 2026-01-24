---
title: "REFACTOR"
tags: [architecture, refactoring]
---
# ABI Framework Modular Refactor Plan
> **Codebase Status:** Synced with repository as of 2026-01-24.

<p align="center">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" alt="Completed"/>
  <img src="https://img.shields.io/badge/Phases-7%2F7-green?style=for-the-badge" alt="Phases"/>
</p>

## Overview
This document outlines the phased refactor plan for the ABI framework to achieve clearer separation of concerns, stronger compile-time feature gating, and reduced coupling between AI and GPU paths, while preserving the public API in `abi.zig`.

## Phases and Tasks

### Phase 1: Interfaces and Contracts
- Define high-level interface boundaries for config, tasks, registry, and AI/GPU coupling.
- Document compile-time gating rules.

### Phase 2: Config Domain Split
- Extract nested structs to config/gpu.zig, config/ai.zig, etc.
- Maintain shim for backward compatibility.

### Phase 3: Tasks Functionality Split
- Split into tasks/persistence.zig, tasks/querying.zig, tasks/lifecycle.zig with facade.

### Phase 4: Registry Modularization
- Split lifecycle to registry/lifecycle.zig, plugins to registry/plugins/.

### Phase 5: AI/GPU Decoupling
- Introduce ai/gpu_interface.zig and shared/gpu_ai_utils.zig.

### Phase 6: Stub Parity Automation
- Implement build script for mod.zig/stub.zig sync.

### Phase 7: Performance Optimizations
- Apply guidelines for ArrayListUnmanaged, mem.copy, profiling.

### Phase 8: Testing & Quality Gates
- Expand tests for gating and cross-module validation.

### Phase 9: Documentation & Migration
- Update docs and create migration guide.

## Key Decisions
- Preserve ABI public API.
- Prioritize config split first.
- Automate stub parity.

## Metrics
- Compile time: Baseline ~X seconds.
- Test coverage: Maintain/exceed current levels.

For detailed tasks, see the JSON manifest in the repository.