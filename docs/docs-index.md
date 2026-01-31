---
title: "Documentation Index"
tags: [documentation, index, navigation]
---
# ABI Framework Documentation
> **Codebase Status:** Synced with repository as of 2026-01-30.
> **Docs:** [Documentation Map](README.md) · [Introduction](intro.md) · [Quickstart](../QUICKSTART.md)

<p align="center">
  <img src="https://img.shields.io/badge/Docs-Complete-success?style=for-the-badge" alt="Docs Complete"/>
  <img src="https://img.shields.io/badge/Last_Updated-2026.01.30-blue?style=for-the-badge" alt="Last Updated"/>
  <img src="https://img.shields.io/badge/Zig-0.16-F7A41D?style=for-the-badge&logo=zig&logoColor=white" alt="Zig 0.16"/>
</p>

<p align="center">
  <strong>Welcome to the ABI Framework documentation!</strong><br/>
  Quick navigation to all guides and references.
</p>

---

## Core Guides

| Guide | Description | Status |
|-------|-------------|--------|
| [Introduction](intro.md) | Architecture & design philosophy | ![Complete](https://img.shields.io/badge/-Complete-success) |
| [Framework](framework.md) | Lifecycle management & configuration | ![Complete](https://img.shields.io/badge/-Complete-success) |
| [Compute Engine](compute.md) | Work-stealing scheduler, task execution | ![Complete](https://img.shields.io/badge/-Complete-success) |
| [Database (WDBX)](database.md) | Vector database, search, backup | ![Complete](https://img.shields.io/badge/-Complete-success) |
| [GPU Acceleration](gpu.md) | Multi-backend GPU, unified API | ![Complete](https://img.shields.io/badge/-Complete-success) |
| [Network](network.md) | Distributed compute, Raft consensus | ![Complete](https://img.shields.io/badge/-Complete-success) |
| [AI & Agents](ai.md) | LLM, embeddings, training | ![Complete](https://img.shields.io/badge/-Complete-success) |
| [Observability](monitoring.md) | Metrics, tracing, profiling | ![Complete](https://img.shields.io/badge/-Complete-success) |
| [Exploration](explore.md) | AI-assisted code navigation | ![Complete](https://img.shields.io/badge/-Complete-success) |

## Technical References

| Document | Description | Status |
|----------|-------------|--------|
| [Zig 0.16 Migration](migration/zig-0.16-migration.md) | Migration patterns and best practices | ![Complete](https://img.shields.io/badge/-Complete-success) |
| [Performance Baseline](PERFORMANCE_BASELINE.md) | Benchmark measurements | ![Complete](https://img.shields.io/badge/-Complete-success) |
| [Benchmarking Guide](benchmarking.md) | Running benchmark suites and interpreting output | ![Complete](https://img.shields.io/badge/-Complete-success) |
| [GPU Backend Details](gpu-backend-improvements.md) | Implementation specifics | ![Complete](https://img.shields.io/badge/-Complete-success) |
| [Feature Flags](feature-flags.md) | Build configuration guide | ![Complete](https://img.shields.io/badge/-Complete-success) |
| [Troubleshooting](troubleshooting.md) | Common issues and solutions | ![Complete](https://img.shields.io/badge/-Complete-success) |
| [Multi‑Agent Coordination](api/ai-multi-agent.md) | Coordinator API for AI agents | ![Complete](https://img.shields.io/badge/-Complete-success) |

## Research & Architecture

| Document | Description | Status |
|----------|-------------|--------|
| [Abbey-Aviva Framework](research/abbey-aviva-abi-wdbx-framework.md) | Multi-persona AI architecture whitepaper | ![Research](https://img.shields.io/badge/-Research-blueviolet) |
| [FPGA/ASIC Analysis](research/hardware-acceleration-fpga-asic.md) | Hardware acceleration research | ![Research](https://img.shields.io/badge/-Research-blueviolet) |

## Implementation Plans

Current planning documents:

| Plan | Description | Status |
|------|-------------|--------|
| [C Bindings Reimplementation](plans/2026-01-30-c-bindings-reimplementation.md) | ABI C bindings rebuild | ![Active](https://img.shields.io/badge/-Active-yellow) |
| [Codebase Improvement](plans/2026-01-30-codebase-improvement.md) | Reliability, performance, cleanup | ![Active](https://img.shields.io/badge/-Active-yellow) |
| [Production Readiness](plans/2026-01-30-production-readiness.md) | Deployment and ops hardening | ![Active](https://img.shields.io/badge/-Active-yellow) |

## Developer Resources

| Resource | Description | Status |
|----------|-------------|--------|
| [Contributing](../CONTRIBUTING.md) | Development workflow & style | ![Ready](https://img.shields.io/badge/-Ready-blue) |
| [Claude Guide](../CLAUDE.md) | AI development guidance | ![Ready](https://img.shields.io/badge/-Ready-blue) |
| [TODO](../TODO.md) | Pending implementations (see [Claude‑Code Massive TODO](../TODO.md#claude-code-massive-todo)) | ![Active](https://img.shields.io/badge/-Active-yellow) |
| [Roadmap](../ROADMAP.md) | Upcoming milestones | ![Active](https://img.shields.io/badge/-Active-yellow) |

## Quick Links

<table>
<tr>
<td width="50%">

### Getting Started
- [README](../README.md) — Project overview
- [Quickstart](../QUICKSTART.md) — 5-minute setup
- [API Reference](../API_REFERENCE.md) — Public API
- [API Reference (Docs)](api/index.md) — Generated API pages

</td>
<td width="50%">

### By Topic
- **AI**: [AI Guide](ai.md), [Connectors](ai.md#connectors)
- **GPU**: [GPU Guide](gpu.md), [Backends](gpu-backend-improvements.md)
- **Data**: [Database](database.md), [Search](database.md#full-text-search)

</td>
</tr>
</table>

---

<p align="center">
  <a href="../README.md">← Back to README</a> •
  <a href="intro.md">Start with Introduction →</a>
</p>
