const std = @import("std");

pub fn redact(input: []const u8) []const u8 {
    if (std.mem.indexOfScalar(u8, input, '\n') != null) {
        return "[multi-line redacted]";
    }
    return input;
}
