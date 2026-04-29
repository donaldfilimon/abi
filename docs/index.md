# ABI Framework

ABI is a Zig 0.17-dev framework for AI services, semantic vector storage, GPU acceleration, and distributed runtime.

## Start Here

- [New engineer onboarding](onboarding.md) covers local setup, build commands, and first-day orientation.
- [Abbey system specification](spec/ABBEY-SPEC.md) describes the multi-profile AI architecture and WDBX substrate.
- [Comprehensive review](review/2026-03-24-comprehensive-review.md) captures historical architecture findings and cleanup priorities.

## Current Operating Rules

- Use the pinned Zig toolchain from `.zigversion`.
- On macOS 26.4+ / Darwin 25.x, use `./build.sh` instead of calling `zig build` directly.
- Public feature APIs must keep `mod.zig` and `stub.zig` in parity, verified with `zig build check-parity`.
- Treat files under archived plans and specs as historical unless a current task explicitly reactivates them.

## Documentation Areas

| Area | Purpose |
| ---- | ------- |
| Onboarding | Setup and first-day workflow for contributors |
| Architecture | ABI, Abbey, and WDBX design references |
| Reviews | Historical codebase review snapshots |
| Archived Plans | Prior implementation plans kept for traceability |
| Archived Specs | Design artifacts that may still inform future work |
