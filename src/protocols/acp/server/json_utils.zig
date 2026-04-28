//! JSON utility helpers for the ACP server.
//!
//! Delegates to `foundation.utils.json` for shared JSON escaping logic.

const std = @import("std");
const parity_gate = @import("../../../common/parity_gate.zig");
const foundation_json = @import("../../../foundation/utils/json.zig");

/// Escape a string for safe embedding in JSON output.
/// Delegates to the canonical foundation implementation which correctly
/// handles all control characters (the prior inline version silently
/// dropped characters on format errors).
pub const appendEscaped = foundation_json.appendJsonEscaped;

test "appendEscaped handles all special chars" {
    if (!parity_gate.canRunTest()) return;
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    try appendEscaped(allocator, &buf, "a\"b\\c\nd\re");
    try std.testing.expectEqualStrings("a\\\"b\\\\c\\nd\\re", buf.items);
}

test "appendEscaped handles control characters" {
    if (!parity_gate.canRunTest()) return;
    const allocator = std.testing.allocator;
    var buf = std.ArrayListUnmanaged(u8).empty;
    defer buf.deinit(allocator);

    // Test tab and control char below 0x20
    try appendEscaped(allocator, &buf, "a\tb\x01c");
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\\t") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\\u") != null);
}

test {
    if (!parity_gate.canRunTest()) return;
    std.testing.refAllDecls(@This());
}
