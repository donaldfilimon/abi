//! Focused LSP unit-test root that keeps the module path anchored at `src/`.

const refAllDecls = @import("common/ref_all.zig").refAllDecls;
const lsp = @import("protocols/lsp/mod.zig");

test {
    refAllDecls(lsp);
}
