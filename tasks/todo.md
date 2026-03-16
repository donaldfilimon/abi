# ABI Development Queue

Active task tracker. Use `git add -f tasks/todo.md` to stage.

## Active — Zig Version Migration (dev.1503 → dev.2905)

- [x] GPA→DebugAllocator rename (82 occurrences, 48 .zig files + README.md)
- [ ] Module system migration: rewrite `build/modules.zig` to wire ~30 named sub-modules (slash-path imports banned in dev.2905)
- [ ] Fix cross-directory `../` imports in sub-modules (config/mod.zig, etc.)
- [ ] Update version pin: `.zigversion`, `build.zig.zon`, `baseline.zig`, README, CI config
- [ ] Full test suite validation via `run_build.sh test --summary all`

## Active — Docs Surface Cleanup

- [ ] Rewrite top-level docs entrypoints (README.md, docs/README.md) for clarity
- [ ] Tighten agent/developer Markdown around current Zig workflow
- [ ] Remove stale `wdbx` named-module guidance from `zig-abi-plugin/` Markdown
- [ ] Improve supporting Markdown with clearer repo-specific guidance

## Next Phase — Release & Scale

- [ ] CI Restoration: Push to main and verify GitHub Actions pass on Linux
- [ ] WASM Optimization: Refine freestanding distance functions for browser-side inference
- [ ] API Expansion: Implement missing OpenAI-compatible streaming endpoints
- [ ] Darwin Validation: Keep compile-only and Linux CI guidance current
- [ ] Plugin Registry: Push `zig-abi-plugin` to official Claude Code registry

## Backlog

- [ ] Finalize automated doc generation for cross-language bindings
- [ ] Audit `tools/cli/commands/` for Windows compatibility
- [ ] Implement distributed WAL for WDBX clusters
- [ ] MCP server hardening (WDBX + ZLS integration)
- [ ] Comprehensive test suite run on Linux CI to verify all waves
