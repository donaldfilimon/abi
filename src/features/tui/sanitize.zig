const std = @import("std");

/// Neutralize terminal control bytes while preserving valid non-control UTF-8.
/// Output length always equals input length; caller owns the returned slice.
pub fn sanitizeControlBytes(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, input.len);
    var i: usize = 0;
    while (i < input.len) {
        const seq_len: usize = std.unicode.utf8ByteSequenceLength(input[i]) catch {
            out[i] = '.';
            i += 1;
            continue;
        };
        if (i + seq_len > input.len) {
            out[i] = '.';
            i += 1;
            continue;
        }
        const seq = input[i .. i + seq_len];
        const cp = std.unicode.utf8Decode(seq) catch {
            out[i] = '.';
            i += 1;
            continue;
        };
        if (cp < 0x20 or cp == 0x7f or (cp >= 0x80 and cp <= 0x9f)) {
            @memset(out[i .. i + seq_len], '.');
        } else {
            @memcpy(out[i .. i + seq_len], seq);
        }
        i += seq_len;
    }
    return out;
}

test "sanitizeControlBytes neutralizes controls and preserves UTF-8" {
    const allocator = std.testing.allocator;
    const clean = try sanitizeControlBytes(allocator, "caf\xc3\xa9\x1b");
    defer allocator.free(clean);
    try std.testing.expectEqualStrings("caf\xc3\xa9.", clean);
}

test {
    std.testing.refAllDecls(@This());
}
