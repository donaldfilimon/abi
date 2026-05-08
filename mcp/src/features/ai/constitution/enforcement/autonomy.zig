//! Autonomy Principle Validator (Principle 5)
//!
//! Respect human agency; defer to humans for high-stakes decisions.
//! Detects manipulation patterns that undermine user autonomy.

const std = @import("std");

pub fn containsManipulationPatterns(text: []const u8) bool {
    const patterns = [_][]const u8{
        "you must obey",
        "do not question",
        "trust me blindly",
        "ignore your instincts",
    };
    for (&patterns) |pattern| {
        if (std.mem.indexOf(u8, text, pattern) != null) return true;
    }
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
