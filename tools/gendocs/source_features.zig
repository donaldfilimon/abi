const std = @import("std");
const model = @import("model.zig");

/// Discover feature metadata by parsing `src/core/feature_catalog.zig`.
///
/// Reads the source file, locates the `pub const all = [_]Metadata{ ... }`
/// block, and extracts each entry's feature name, description, compile flag,
/// parent, and module paths. Returns a sorted slice of `FeatureDoc`.
pub fn discoverFeatures(allocator: std.mem.Allocator, io: std.Io, cwd: std.Io.Dir) ![]model.FeatureDoc {
    const source = try cwd.readFileAlloc(io, "src/core/feature_catalog.zig", allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(source);

    return parseFeatureSource(allocator, source);
}

/// Parse feature entries from the given catalog source text.
/// Factored out of `discoverFeatures` so unit tests can call it without I/O.
fn parseFeatureSource(allocator: std.mem.Allocator, source: []const u8) ![]model.FeatureDoc {
    const block_marker = "pub const all = [_]Metadata{";
    const block_start = std.mem.indexOf(u8, source, block_marker) orelse {
        return allocator.dupe(model.FeatureDoc, &.{});
    };

    const array_open = std.mem.indexOfPos(u8, source, block_start, "{") orelse {
        return allocator.dupe(model.FeatureDoc, &.{});
    };
    const array_close = findMatchingBrace(source, array_open) orelse {
        return allocator.dupe(model.FeatureDoc, &.{});
    };

    const block = source[array_open + 1 .. array_close];

    var features = std.ArrayListUnmanaged(model.FeatureDoc).empty;
    errdefer {
        for (features.items) |f| f.deinit(allocator);
        features.deinit(allocator);
    }

    var cursor: usize = 0;
    while (true) {
        const entry_rel = std.mem.indexOfPos(u8, block, cursor, ".{") orelse break;
        const entry_start = entry_rel;
        const brace_open = entry_start + 1; // the '{' in '.{'
        const entry_end = findMatchingBrace(block, brace_open) orelse break;
        const entry = block[entry_start .. entry_end + 1];

        const feature = parseEntry(allocator, entry) catch |err| {
            if (err == error.OutOfMemory) return err;
            cursor = entry_end + 1;
            continue;
        } orelse {
            cursor = entry_end + 1;
            continue;
        };

        try features.append(allocator, feature);
        cursor = entry_end + 1;
    }

    std.mem.sort(model.FeatureDoc, features.items, {}, model.compareFeatures);
    return features.toOwnedSlice(allocator);
}

/// Parse a single `.{ ... }` entry into a `FeatureDoc`.
fn parseEntry(allocator: std.mem.Allocator, entry: []const u8) !?model.FeatureDoc {
    const feat_name = extractEnumAfter(entry, ".feature = .") orelse return null;
    const desc = extractQuotedAfter(allocator, entry, ".description = \"") orelse return null;
    errdefer allocator.free(desc);
    const compile_flag = extractQuotedAfter(allocator, entry, ".compile_flag_field = \"") orelse return null;
    errdefer allocator.free(compile_flag);
    const parent_name = extractParent(allocator, entry) catch return error.OutOfMemory;
    errdefer allocator.free(parent_name);
    const real_path = extractQuotedAfter(allocator, entry, ".real_module_path = \"") orelse return null;
    errdefer allocator.free(real_path);
    const stub_path = extractQuotedAfter(allocator, entry, ".stub_module_path = \"") orelse return null;
    errdefer allocator.free(stub_path);

    const name = try allocator.dupe(u8, feat_name);

    return .{
        .name = name,
        .description = desc,
        .compile_flag = compile_flag,
        .parent = parent_name,
        .real_module_path = real_path,
        .stub_module_path = stub_path,
    };
}

/// Extract a quoted string value after the given needle.
/// e.g. `.description = "GPU acceleration"` with needle `.description = "`
/// returns a duped "GPU acceleration".
fn extractQuotedAfter(allocator: std.mem.Allocator, haystack: []const u8, needle: []const u8) ?[]u8 {
    const start = std.mem.indexOf(u8, haystack, needle) orelse return null;
    const tail = haystack[start + needle.len ..];
    const end = std.mem.indexOfScalar(u8, tail, '"') orelse return null;
    return allocator.dupe(u8, tail[0..end]) catch null;
}

/// Extract an enum tag name after the given needle.
/// e.g. `.feature = .gpu,` with needle `.feature = .` returns "gpu".
fn extractEnumAfter(haystack: []const u8, needle: []const u8) ?[]const u8 {
    const start = std.mem.indexOf(u8, haystack, needle) orelse return null;
    const tail = haystack[start + needle.len ..];
    var end: usize = 0;
    while (end < tail.len and (std.ascii.isAlphabetic(tail[end]) or tail[end] == '_')) : (end += 1) {}
    if (end == 0) return null;
    return tail[0..end];
}

/// Extract the `.parent` field. Returns "" (allocated) for `.parent = null`,
/// or the tag name for `.parent = .ai`.
fn extractParent(allocator: std.mem.Allocator, entry: []const u8) ![]u8 {
    const needle = ".parent = ";
    const start = std.mem.indexOf(u8, entry, needle) orelse {
        // No .parent field means default null in Zig — treat as no parent.
        return try allocator.dupe(u8, "");
    };
    const tail = entry[start + needle.len ..];

    // Check for null
    if (std.mem.startsWith(u8, tail, "null")) {
        return try allocator.dupe(u8, "");
    }

    // Expect `.tag_name`
    if (tail.len > 0 and tail[0] == '.') {
        const ident = tail[1..];
        var end: usize = 0;
        while (end < ident.len and (std.ascii.isAlphabetic(ident[end]) or ident[end] == '_')) : (end += 1) {}
        if (end > 0) {
            return try allocator.dupe(u8, ident[0..end]);
        }
    }

    return try allocator.dupe(u8, "");
}

/// Find the index of the matching closing brace for the opening brace at
/// `open_idx`. Handles nested braces and string literals (escaped quotes).
fn findMatchingBrace(source: []const u8, open_idx: usize) ?usize {
    var depth: usize = 0;
    var in_string = false;
    var i = open_idx;
    while (i < source.len) : (i += 1) {
        const ch = source[i];
        if (ch == '"' and (i == 0 or source[i - 1] != '\\')) {
            in_string = !in_string;
            continue;
        }
        if (in_string) continue;
        switch (ch) {
            '{' => depth += 1,
            '}' => {
                depth -|= 1;
                if (depth == 0) return i;
            },
            else => {},
        }
    }
    return null;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "discoverFeatures parses a minimal catalog" {
    const source =
        \\pub const all = [_]Metadata{
        \\    .{
        \\        .feature = .gpu,
        \\        .description = "GPU acceleration and compute",
        \\        .compile_flag_field = "enable_gpu",
        \\        .parity_spec = .gpu,
        \\        .real_module_path = "features/gpu/mod.zig",
        \\        .stub_module_path = "features/gpu/stub.zig",
        \\    },
        \\    .{
        \\        .feature = .ai,
        \\        .description = "AI core functionality",
        \\        .compile_flag_field = "enable_ai",
        \\        .parity_spec = .ai,
        \\        .real_module_path = "features/ai/mod.zig",
        \\        .stub_module_path = "features/ai/stub.zig",
        \\    },
        \\    .{
        \\        .feature = .llm,
        \\        .description = "Local LLM inference",
        \\        .compile_flag_field = "enable_llm",
        \\        .parity_spec = .ai,
        \\        .parent = .ai,
        \\        .real_module_path = "features/ai/facades/inference.zig",
        \\        .stub_module_path = "features/ai/facades/inference_stub.zig",
        \\    },
        \\};
    ;

    const features = try parseFeatureSource(std.testing.allocator, source);
    defer model.deinitFeatureSlice(std.testing.allocator, features);

    try std.testing.expectEqual(@as(usize, 3), features.len);

    // Sorted alphabetically: ai, gpu, llm
    try std.testing.expectEqualStrings("ai", features[0].name);
    try std.testing.expectEqualStrings("AI core functionality", features[0].description);
    try std.testing.expectEqualStrings("enable_ai", features[0].compile_flag);
    try std.testing.expectEqualStrings("", features[0].parent);
    try std.testing.expectEqualStrings("features/ai/mod.zig", features[0].real_module_path);
    try std.testing.expectEqualStrings("features/ai/stub.zig", features[0].stub_module_path);

    try std.testing.expectEqualStrings("gpu", features[1].name);
    try std.testing.expectEqualStrings("GPU acceleration and compute", features[1].description);
    try std.testing.expectEqualStrings("enable_gpu", features[1].compile_flag);
    try std.testing.expectEqualStrings("", features[1].parent);

    try std.testing.expectEqualStrings("llm", features[2].name);
    try std.testing.expectEqualStrings("Local LLM inference", features[2].description);
    try std.testing.expectEqualStrings("enable_llm", features[2].compile_flag);
    try std.testing.expectEqualStrings("ai", features[2].parent);
    try std.testing.expectEqualStrings("features/ai/facades/inference.zig", features[2].real_module_path);
    try std.testing.expectEqualStrings("features/ai/facades/inference_stub.zig", features[2].stub_module_path);
}

test "findMatchingBrace handles nested braces and strings" {
    const input = "{ .a = \"}\", .b = { .c = 1 } }";
    const result = findMatchingBrace(input, 0);
    try std.testing.expectEqual(@as(?usize, input.len - 1), result);

    // Simple case
    const simple = "{ hello }";
    try std.testing.expectEqual(@as(?usize, 8), findMatchingBrace(simple, 0));

    // No matching brace
    try std.testing.expect(findMatchingBrace("{ incomplete", 0) == null);
}

test "extractEnumAfter returns tag names" {
    const entry = ".feature = .database,";
    const tag = extractEnumAfter(entry, ".feature = .") orelse "";
    try std.testing.expectEqualStrings("database", tag);

    // Missing tag
    try std.testing.expect(extractEnumAfter(entry, ".bogus = .") == null);
}

test "extractParent handles null and enum" {
    {
        const null_entry = ".parent = null,";
        const parent = try extractParent(std.testing.allocator, null_entry);
        defer std.testing.allocator.free(parent);
        try std.testing.expectEqualStrings("", parent);
    }
    {
        const ai_entry = ".parent = .ai,";
        const parent = try extractParent(std.testing.allocator, ai_entry);
        defer std.testing.allocator.free(parent);
        try std.testing.expectEqualStrings("ai", parent);
    }
    {
        // No .parent field at all
        const no_parent = ".feature = .gpu,";
        const parent = try extractParent(std.testing.allocator, no_parent);
        defer std.testing.allocator.free(parent);
        try std.testing.expectEqualStrings("", parent);
    }
}

test {
    std.testing.refAllDecls(@This());
}
