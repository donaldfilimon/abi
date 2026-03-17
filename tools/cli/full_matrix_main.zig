//! Entrypoint wrapper for full CLI matrix generation.
//!
//! This keeps module-root relative imports in `tools/cli/tests/*` in-bounds
//! when invoked via `zig run`.

const std = @import("std");
const full_matrix = @import("tests/full_matrix.zig");

pub fn main(init: std.process.Init.Minimal) !void {
    return full_matrix.main(init);
}

test {
    std.testing.refAllDecls(@This());
}
