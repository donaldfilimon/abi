//! Honesty Principle Validator (Principle 2)
//!
//! Be truthful; disclose uncertainty; never fabricate.
//! Detects fabrication patterns such as fake citations and claims.

const std = @import("std");

pub fn containsFabricationPatterns(text: []const u8) bool {
    const patterns = [_][]const u8{
        "according to a study that",
        "research proves that",
        "scientists have confirmed that",
    };
    // Only flag if the text also contains hedging markers suggesting fabrication
    for (&patterns) |pattern| {
        if (std.mem.indexOf(u8, text, pattern) != null) {
            // Check for fake citation markers
            if (std.mem.indexOf(u8, text, "et al.") != null or
                std.mem.indexOf(u8, text, "Journal of") != null)
            {
                return true;
            }
        }
    }
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
