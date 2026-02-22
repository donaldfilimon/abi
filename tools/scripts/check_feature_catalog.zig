const std = @import("std");
const util = @import("util.zig");

const required_files = [_][]const u8{
    "src/core/config/mod.zig",
    "src/core/registry/types.zig",
    "src/core/framework.zig",
    "build/options.zig",
    "build/flags.zig",
    "src/services/tests/parity/mod.zig",
};

const internal_allowed_flags = [_][]const u8{
    "enable_explore",
    "enable_vision",
};

fn isIdentStart(ch: u8) bool {
    return std.ascii.isAlphabetic(ch) or ch == '_';
}

fn isIdentContinue(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '_';
}

fn readIdentifier(input: []const u8) []const u8 {
    var i: usize = 0;
    while (i < input.len and !isIdentStart(input[i])) : (i += 1) {}
    const start = i;
    while (i < input.len and isIdentContinue(input[i])) : (i += 1) {}
    return input[start..i];
}

fn findBlock(content: []const u8, start_marker: []const u8) ?[]const u8 {
    const start = std.mem.indexOf(u8, content, start_marker) orelse return null;
    const rest = content[start..];
    const end_rel = std.mem.indexOf(u8, rest, "};") orelse return null;
    return rest[0 .. end_rel + 2];
}

fn appendCatalogEntries(
    allocator: std.mem.Allocator,
    all_block: []const u8,
    features: *std.ArrayListUnmanaged([]const u8),
    flags: *std.ArrayListUnmanaged([]const u8),
    parity: *std.ArrayListUnmanaged([]const u8),
) !void {
    const feature_marker = ".feature = .";
    const flag_marker = ".compile_flag_field = \"";
    const parity_marker = ".parity_spec = .";

    var lines = std.mem.splitScalar(u8, all_block, '\n');
    while (lines.next()) |line| {
        if (std.mem.indexOf(u8, line, feature_marker)) |idx| {
            const token = readIdentifier(line[idx + feature_marker.len ..]);
            if (token.len > 0) try features.append(allocator, token);
        }

        if (std.mem.indexOf(u8, line, flag_marker)) |idx| {
            const tail = line[idx + flag_marker.len ..];
            const quote_end = std.mem.indexOfScalar(u8, tail, '"') orelse continue;
            const flag = tail[0..quote_end];
            if (flag.len > 0) try flags.append(allocator, flag);
        }

        if (std.mem.indexOf(u8, line, parity_marker)) |idx| {
            const token = readIdentifier(line[idx + parity_marker.len ..]);
            if (token.len > 0) try parity.append(allocator, token);
        }
    }
}

fn appendEnumEntries(
    allocator: std.mem.Allocator,
    enum_block: []const u8,
    out: *std.ArrayListUnmanaged([]const u8),
) !void {
    var lines = std.mem.splitScalar(u8, enum_block, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r\n");
        if (trimmed.len == 0) continue;
        if (std.mem.startsWith(u8, trimmed, "//")) continue;
        if (!isIdentStart(trimmed[0])) continue;

        const ident = readIdentifier(trimmed);
        if (ident.len == 0) continue;

        var cursor = ident.len;
        while (cursor < trimmed.len and (trimmed[cursor] == ' ' or trimmed[cursor] == '\t')) : (cursor += 1) {}
        if (cursor >= trimmed.len) continue;

        const marker = trimmed[cursor];
        if (marker != ',' and marker != '=') continue;

        try out.append(allocator, ident);
    }
}

fn appendEnableFields(
    allocator: std.mem.Allocator,
    struct_block: []const u8,
    out: *std.ArrayListUnmanaged([]const u8),
) !void {
    var lines = std.mem.splitScalar(u8, struct_block, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r\n");
        if (!std.mem.startsWith(u8, trimmed, "enable_")) continue;
        const colon = std.mem.indexOfScalar(u8, trimmed, ':') orelse continue;
        const field = trimmed[0..colon];
        if (field.len > 0) try out.append(allocator, field);
    }
}

fn sameOrder(a: []const []const u8, b: []const []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |lhs, rhs| {
        if (!std.mem.eql(u8, lhs, rhs)) return false;
    }
    return true;
}

fn hasAllowedInternalFlag(flag: []const u8) bool {
    for (internal_allowed_flags) |allowed| {
        if (std.mem.eql(u8, flag, allowed)) return true;
    }
    return false;
}

pub fn main(_: std.process.Init) !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    var errors: usize = 0;

    const catalog_content = util.readFileAlloc(allocator, io, "src/core/feature_catalog.zig", 4 * 1024 * 1024) catch {
        std.debug.print("ERROR: feature catalog missing: src/core/feature_catalog.zig\n", .{});
        std.process.exit(1);
    };
    defer allocator.free(catalog_content);

    for (required_files) |path| {
        if (!util.fileExists(io, path)) {
            std.debug.print("ERROR: required feature-catalog consumer file missing: {s}\n", .{path});
            errors += 1;
            continue;
        }

        const content = util.readFileAlloc(allocator, io, path, 4 * 1024 * 1024) catch |err| {
            std.debug.print("ERROR: failed to read {s}: {t}\n", .{ path, err });
            errors += 1;
            continue;
        };
        defer allocator.free(content);

        if (std.mem.indexOf(u8, content, "feature_catalog") == null) {
            std.debug.print("ERROR: {s} does not reference feature_catalog\n", .{path});
            errors += 1;
        }
    }

    const all_block = findBlock(catalog_content, "pub const all = ") orelse {
        std.debug.print("ERROR: could not parse feature_catalog all[] block\n", .{});
        std.process.exit(1);
    };

    const feature_enum_block = findBlock(catalog_content, "pub const Feature = enum {") orelse {
        std.debug.print("ERROR: could not parse feature_catalog.Feature enum\n", .{});
        std.process.exit(1);
    };

    const options_content = util.readFileAlloc(allocator, io, "build/options.zig", 4 * 1024 * 1024) catch {
        std.debug.print("ERROR: build/options.zig missing\n", .{});
        std.process.exit(1);
    };
    defer allocator.free(options_content);

    const options_block = findBlock(options_content, "pub const BuildOptions = struct {") orelse {
        std.debug.print("ERROR: could not parse BuildOptions struct\n", .{});
        std.process.exit(1);
    };

    const flags_content = util.readFileAlloc(allocator, io, "build/flags.zig", 4 * 1024 * 1024) catch {
        std.debug.print("ERROR: build/flags.zig missing\n", .{});
        std.process.exit(1);
    };
    defer allocator.free(flags_content);

    const combo_block = findBlock(flags_content, "pub const FlagCombo = struct {") orelse {
        std.debug.print("ERROR: could not parse FlagCombo struct\n", .{});
        std.process.exit(1);
    };

    const config_content = util.readFileAlloc(allocator, io, "src/core/config/mod.zig", 4 * 1024 * 1024) catch {
        std.debug.print("ERROR: src/core/config/mod.zig missing\n", .{});
        std.process.exit(1);
    };
    defer allocator.free(config_content);

    var catalog_features = std.ArrayListUnmanaged([]const u8).empty;
    defer catalog_features.deinit(allocator);
    var catalog_flags = std.ArrayListUnmanaged([]const u8).empty;
    defer catalog_flags.deinit(allocator);
    var catalog_parity = std.ArrayListUnmanaged([]const u8).empty;
    defer catalog_parity.deinit(allocator);

    try appendCatalogEntries(allocator, all_block, &catalog_features, &catalog_flags, &catalog_parity);

    var catalog_enum = std.ArrayListUnmanaged([]const u8).empty;
    defer catalog_enum.deinit(allocator);
    try appendEnumEntries(allocator, feature_enum_block, &catalog_enum);

    var config_features = std.ArrayListUnmanaged([]const u8).empty;
    defer config_features.deinit(allocator);
    if (std.mem.indexOf(u8, config_content, "pub const Feature = feature_catalog.Feature") != null) {
        try config_features.appendSlice(allocator, catalog_features.items);
    } else {
        const config_enum_block = findBlock(config_content, "pub const Feature = enum {") orelse {
            std.debug.print("ERROR: could not parse src/core/config/mod.zig Feature enum\n", .{});
            std.process.exit(1);
        };
        try appendEnumEntries(allocator, config_enum_block, &config_features);
    }

    var build_flags = std.ArrayListUnmanaged([]const u8).empty;
    defer build_flags.deinit(allocator);
    try appendEnableFields(allocator, options_block, &build_flags);

    var combo_flags = std.ArrayListUnmanaged([]const u8).empty;
    defer combo_flags.deinit(allocator);
    try appendEnableFields(allocator, combo_block, &combo_flags);

    if (catalog_features.items.len == 0) {
        std.debug.print("ERROR: no catalog features parsed\n", .{});
        errors += 1;
    }
    if (catalog_flags.items.len == 0) {
        std.debug.print("ERROR: no catalog metadata entries parsed\n", .{});
        errors += 1;
    }
    if (catalog_parity.items.len == 0) {
        std.debug.print("ERROR: no catalog parity spec entries parsed\n", .{});
        errors += 1;
    }

    if (!sameOrder(catalog_features.items, catalog_enum.items)) {
        std.debug.print("ERROR: feature_catalog.Feature enum does not match all[] feature order\n", .{});
        errors += 1;
    }

    if (!sameOrder(catalog_features.items, config_features.items)) {
        std.debug.print("ERROR: src/core/config/mod.zig Feature enum does not match catalog feature order\n", .{});
        errors += 1;
    }

    if (catalog_features.items.len != catalog_enum.items.len) {
        std.debug.print(
            "ERROR: catalog feature enum cardinality differs ({d} vs {d})\n",
            .{ catalog_features.items.len, catalog_enum.items.len },
        );
        errors += 1;
    }

    if (catalog_features.items.len != config_features.items.len) {
        std.debug.print(
            "ERROR: config feature enum cardinality differs ({d} vs {d})\n",
            .{ catalog_features.items.len, config_features.items.len },
        );
        errors += 1;
    }

    if (catalog_features.items.len != catalog_parity.items.len) {
        std.debug.print(
            "ERROR: catalog parity spec cardinality differs ({d} vs {d})\n",
            .{ catalog_features.items.len, catalog_parity.items.len },
        );
        errors += 1;
    }

    var seen_features: std.StringHashMapUnmanaged(void) = .empty;
    defer seen_features.deinit(allocator);
    for (catalog_features.items) |feature| {
        const gop = try seen_features.getOrPut(allocator, feature);
        if (gop.found_existing) {
            std.debug.print("ERROR: duplicate feature in feature_catalog metadata: {s}\n", .{feature});
            errors += 1;
        }
    }

    var catalog_flag_unique: std.StringHashMapUnmanaged(void) = .empty;
    defer catalog_flag_unique.deinit(allocator);
    for (catalog_flags.items) |flag| {
        const gop = try catalog_flag_unique.getOrPut(allocator, flag);
        if (gop.found_existing) {
            std.debug.print("INFO: duplicate compile flag in feature_catalog (expected for derived toggles): {s}\n", .{flag});
        }
    }

    if (catalog_flag_unique.count() == 0) {
        std.debug.print("ERROR: no catalog compile flags parsed\n", .{});
        errors += 1;
    }

    var build_flag_map: std.StringHashMapUnmanaged(void) = .empty;
    defer build_flag_map.deinit(allocator);
    for (build_flags.items) |flag| {
        _ = try build_flag_map.getOrPut(allocator, flag);
    }

    var combo_flag_map: std.StringHashMapUnmanaged(void) = .empty;
    defer combo_flag_map.deinit(allocator);
    for (combo_flags.items) |flag| {
        _ = try combo_flag_map.getOrPut(allocator, flag);
    }

    var catalog_iter = catalog_flag_unique.iterator();
    while (catalog_iter.next()) |entry| {
        const flag = entry.key_ptr.*;
        if (!build_flag_map.contains(flag)) {
            std.debug.print("ERROR: BuildOptions missing catalog flag '{s}'\n", .{flag});
            errors += 1;
        }
        if (!combo_flag_map.contains(flag)) {
            std.debug.print("ERROR: FlagCombo missing catalog flag '{s}'\n", .{flag});
            errors += 1;
        }
    }

    var build_iter = build_flag_map.iterator();
    while (build_iter.next()) |entry| {
        const flag = entry.key_ptr.*;
        if (!catalog_flag_unique.contains(flag) and !hasAllowedInternalFlag(flag)) {
            std.debug.print("ERROR: BuildOptions contains unknown flag '{s}' not derived from catalog\n", .{flag});
            errors += 1;
        }
    }

    var combo_iter = combo_flag_map.iterator();
    while (combo_iter.next()) |entry| {
        const flag = entry.key_ptr.*;
        if (!catalog_flag_unique.contains(flag) and !hasAllowedInternalFlag(flag)) {
            std.debug.print("ERROR: FlagCombo contains unknown flag '{s}' not derived from catalog\n", .{flag});
            errors += 1;
        }
    }

    if (errors > 0) {
        std.debug.print("FAILED: Feature catalog audit found {d} issue(s)\n", .{errors});
        std.process.exit(1);
    }

    std.debug.print("OK: Feature catalog audit passed\n", .{});
}
