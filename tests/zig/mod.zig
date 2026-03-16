//! Logical test-suite root for ABI.
//!
//! Phase 1 keeps the existing test modules under `src/services/tests/` while
//! the build graph migrates to the dedicated `tests/zig/` lane.

const legacy = @import("../../src/services/tests/mod.zig");

test {
    _ = legacy;
}
