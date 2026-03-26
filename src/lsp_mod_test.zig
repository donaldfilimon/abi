//! Focused LSP unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const lsp = @import("protocols/lsp/mod.zig");

test {
    _ = lsp;
    std.testing.refAllDecls(lsp);
}
