//! Focused LSP integration test root.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const lsp_tests = @import("integration/lsp_test.zig");
const lsp_protocol_tests = @import("integration/lsp_protocol_test.zig");

test {
    _ = abi;
    _ = build_options;
    std.testing.refAllDecls(@This());
}
