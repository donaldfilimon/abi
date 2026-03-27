//! Storage Validation Utilities
//!
//! Key validation (path traversal prevention) and MIME content type guessing.

const std = @import("std");

/// Validate a storage key for safety. Rejects empty keys, absolute paths,
/// path traversal (`..`), and overly long keys.
pub fn isValidKey(key: []const u8) bool {
    if (key.len == 0 or key.len > 4096) return false;
    // Reject absolute paths
    if (key[0] == '/') return false;
    // Reject path traversal
    var i: usize = 0;
    while (i < key.len) {
        if (key[i] == '.' and i + 1 < key.len and key[i + 1] == '.') {
            // ".." at start, end, or surrounded by slashes
            if ((i == 0 or key[i - 1] == '/') and
                (i + 2 >= key.len or key[i + 2] == '/'))
            {
                return false;
            }
        }
        i += 1;
    }
    return true;
}

/// Guess a MIME content type from the file extension of `key`.
pub fn guessContentType(key: []const u8) []const u8 {
    const ext = std.fs.path.extension(key); // includes leading dot
    if (ext.len == 0) return "application/octet-stream";
    const lookup = .{
        .{ ".txt", "text/plain" },
        .{ ".json", "application/json" },
        .{ ".html", "text/html" },
        .{ ".htm", "text/html" },
        .{ ".css", "text/css" },
        .{ ".js", "application/javascript" },
        .{ ".png", "image/png" },
        .{ ".jpg", "image/jpeg" },
        .{ ".jpeg", "image/jpeg" },
        .{ ".gif", "image/gif" },
        .{ ".svg", "image/svg+xml" },
        .{ ".xml", "application/xml" },
        .{ ".pdf", "application/pdf" },
        .{ ".zip", "application/zip" },
        .{ ".md", "text/markdown" },
        .{ ".csv", "text/csv" },
    };
    inline for (lookup) |pair| {
        if (std.mem.eql(u8, ext, pair[0])) return pair[1];
    }
    return "application/octet-stream";
}
