# simd

> SIMD operations (shorthand for `shared.simd`).

**Source:** [`src/services/shared/simd.zig`](../../src/services/shared/simd.zig)

**Availability:** Always enabled

---

Compatibility shim — redirects to simd/mod.zig

Many files import this path directly. The actual implementation has been
split into src/services/shared/simd/ submodules.

New code should import via the parent mod.zig chain instead of using
a direct file path.

---

## API

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
