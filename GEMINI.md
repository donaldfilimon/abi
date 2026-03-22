---
title: Gemini Guide
purpose: Specific instructions for the Gemini CLI
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2934+47d2e5de9
---

# Gemini Guide

Gemini is a secondary migration and cross-check CLI in this repository. It is
not a separate workflow or Zig policy source.

## Canonical Sources

- Repo workflow: [AGENTS.md](AGENTS.md)
- Active execution tracker: [tasks/todo.md](tasks/todo.md)
- Correction log: [tasks/lessons.md](tasks/lessons.md)
- Zig validation: `zig build full-check` with a host-built or otherwise known-good Zig on Darwin 25+ / macOS 26+; if stock Zig is linker-blocked, record `zig fmt --check ...` plus `zig test <file> -fno-emit-bin` as fallback evidence

## Expectations

- Follow the same plan-first and verification rules documented in
  [AGENTS.md](AGENTS.md).
- Use the same Zig baseline and close-out gates documented in
  [CLAUDE.md](CLAUDE.md).
- Treat multi-CLI consensus as advisory input; final repository decisions still
  need local verification on this checkout.
