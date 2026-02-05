//! ZON (Zig Object Notation) Format Support for WDBX Database
//!
//! Provides native Zig serialization format for WDBX vector databases.
//! ZON is Zig's built-in data interchange format, similar to JSON but
//! using Zig syntax for better tooling integration and type safety.
//!
//! ## Features
//! - Native Zig syntax - can be directly embedded in Zig source files
//! - Type-safe parsing with compile-time validation
//! - Streaming support for large datasets
//! - Metadata preservation
//! - Human-readable and editable
//!
//! ## File Format (.wdbx.zon)
//! ```zon
//! .{
//!     .version = 1,
//!     .name = "my_database",
//!     .dimension = 384,
//!     .distance_metric = .cosine,
//!     .records = .{
//!         .{ .id = 1, .vector = .{ 0.1, 0.2, 0.3 }, .metadata = "doc1" },
//!         .{ .id = 2, .vector = .{ 0.4, 0.5, 0.6 }, .metadata = "doc2" },
//!     },
//! }
//! ```
//!
//! ## Usage
//! ```zig
//! const zon_format = @import("formats/zon.zig");
//!
//! // Export database to ZON
//! const data = try zon_format.exportDatabase(allocator, database);
//! defer allocator.free(data);
//!
//! // Import database from ZON
//! var db = try zon_format.importDatabase(allocator, zon_data);
//! defer db.deinit();
//! ```

const std = @import("std");
const batch = @import("../batch.zig");
const time = @import("../../../services/shared/time.zig");

/// ZON format version for WDBX databases.
pub const ZON_FORMAT_VERSION: u32 = 1;

/// Distance metric options for serialization.
pub const DistanceMetric = enum {
    euclidean,
    cosine,
    dot_product,
    manhattan,
};

/// ZON database configuration structure.
pub const ZonDatabaseConfig = struct {
    version: u32 = ZON_FORMAT_VERSION,
    name: []const u8 = "wdbx_database",
    dimension: u32 = 0,
    distance_metric: DistanceMetric = .cosine,
    record_count: u64 = 0,
    created_at: ?i64 = null,
    modified_at: ?i64 = null,
    metadata: ?[]const u8 = null,
};

/// ZON record structure matching database vectors.
pub const ZonRecord = struct {
    id: u64 = 0,
    vector: []const f32 = &.{},
    metadata: ?[]const u8 = null,
    text: ?[]const u8 = null,
};

/// Complete ZON database structure.
pub const ZonDatabase = struct {
    version: u32 = ZON_FORMAT_VERSION,
    name: []const u8 = "wdbx_database",
    dimension: u32 = 0,
    distance_metric: DistanceMetric = .cosine,
    created_at: ?i64 = null,
    modified_at: ?i64 = null,
    db_metadata: ?[]const u8 = null,
    records: []const ZonRecord = &.{},
};

/// ZON format errors.
pub const ZonFormatError = error{
    InvalidVersion,
    InvalidDimension,
    DimensionMismatch,
    ParseError,
    SerializeError,
    OutOfMemory,
    InvalidFormat,
};

/// ZON format handler for WDBX databases.
pub const ZonFormat = struct {
    allocator: std.mem.Allocator,
    config: ZonDatabaseConfig,

    /// Initialize a new ZON format handler.
    pub fn init(allocator: std.mem.Allocator) ZonFormat {
        return .{
            .allocator = allocator,
            .config = .{},
        };
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(allocator: std.mem.Allocator, config: ZonDatabaseConfig) ZonFormat {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Parse ZON data into a database structure.
    pub fn parse(self: *ZonFormat, data: []const u8) !ZonDatabase {
        const parsed = std.zon.parseFromSlice(ZonDatabase, self.allocator, data, .{}) catch |err| {
            std.log.warn("ZON parse error: {t}", .{err});
            return ZonFormatError.ParseError;
        };
        defer parsed.deinit();

        // Validate version
        if (parsed.value.version > ZON_FORMAT_VERSION) {
            return ZonFormatError.InvalidVersion;
        }

        // Validate dimension consistency
        if (parsed.value.records.len > 0) {
            const expected_dim = parsed.value.dimension;
            for (parsed.value.records) |record| {
                if (expected_dim > 0 and record.vector.len != expected_dim) {
                    return ZonFormatError.DimensionMismatch;
                }
            }
        }

        // Deep copy the parsed data
        return self.deepCopyDatabase(parsed.value);
    }

    /// Deep copy a ZonDatabase structure.
    fn deepCopyDatabase(self: *ZonFormat, db: ZonDatabase) !ZonDatabase {
        const name_copy = try self.allocator.dupe(u8, db.name);
        errdefer self.allocator.free(name_copy);

        var db_metadata_copy: ?[]u8 = null;
        if (db.db_metadata) |m| {
            db_metadata_copy = try self.allocator.dupe(u8, m);
        }
        errdefer if (db_metadata_copy) |m| self.allocator.free(m);

        var records_copy = try self.allocator.alloc(ZonRecord, db.records.len);
        errdefer self.allocator.free(records_copy);

        var copied_count: usize = 0;
        errdefer {
            for (records_copy[0..copied_count]) |rec| {
                self.allocator.free(rec.vector);
                if (rec.metadata) |m| self.allocator.free(m);
                if (rec.text) |t| self.allocator.free(t);
            }
        }

        for (db.records, 0..) |record, i| {
            const vector_copy = try self.allocator.dupe(f32, record.vector);
            errdefer self.allocator.free(vector_copy);

            var meta_copy: ?[]u8 = null;
            if (record.metadata) |m| {
                meta_copy = try self.allocator.dupe(u8, m);
            }
            errdefer if (meta_copy) |m| self.allocator.free(m);

            var text_copy: ?[]u8 = null;
            if (record.text) |t| {
                text_copy = try self.allocator.dupe(u8, t);
            }

            records_copy[i] = .{
                .id = record.id,
                .vector = vector_copy,
                .metadata = meta_copy,
                .text = text_copy,
            };
            copied_count += 1;
        }

        return .{
            .version = db.version,
            .name = name_copy,
            .dimension = db.dimension,
            .distance_metric = db.distance_metric,
            .created_at = db.created_at,
            .modified_at = db.modified_at,
            .db_metadata = db_metadata_copy,
            .records = records_copy,
        };
    }

    /// Serialize a database structure to ZON format.
    pub fn serialize(self: *ZonFormat, db: ZonDatabase) ![]u8 {
        var aw = std.Io.Writer.Allocating.init(self.allocator);
        errdefer aw.deinit();
        const writer = &aw.writer;

        try writer.writeAll(".{\n");

        // Write header fields
        try writer.writeAll("    .version = ");
        try std.fmt.formatInt(db.version, 10, .lower, .{}, writer);
        try writer.writeAll(",\n");

        try writer.writeAll("    .name = ");
        try writeZonString(writer, db.name);
        try writer.writeAll(",\n");

        try writer.writeAll("    .dimension = ");
        try std.fmt.formatInt(db.dimension, 10, .lower, .{}, writer);
        try writer.writeAll(",\n");

        try writer.writeAll("    .distance_metric = .");
        // Use {t} format specifier instead of @tagName (Zig 0.16)
        try std.fmt.format(writer, "{t}", .{db.distance_metric});
        try writer.writeAll(",\n");

        // Optional timestamps
        if (db.created_at) |ts| {
            try writer.writeAll("    .created_at = ");
            try std.fmt.formatInt(ts, 10, .lower, .{}, writer);
            try writer.writeAll(",\n");
        }

        if (db.modified_at) |ts| {
            try writer.writeAll("    .modified_at = ");
            try std.fmt.formatInt(ts, 10, .lower, .{}, writer);
            try writer.writeAll(",\n");
        }

        // Optional database metadata
        if (db.db_metadata) |meta| {
            try writer.writeAll("    .db_metadata = ");
            try writeZonString(writer, meta);
            try writer.writeAll(",\n");
        }

        // Write records
        try writer.writeAll("    .records = .{\n");

        for (db.records, 0..) |record, idx| {
            try self.writeRecord(writer, record);
            if (idx < db.records.len - 1) {
                try writer.writeAll(",");
            }
            try writer.writeAll("\n");
        }

        try writer.writeAll("    },\n");
        try writer.writeAll("}\n");

        return aw.toOwnedSlice();
    }

    /// Write a single record in ZON format.
    fn writeRecord(self: *ZonFormat, writer: anytype, record: ZonRecord) !void {
        _ = self;
        try writer.writeAll("        .{ ");

        // Write ID
        try writer.writeAll(".id = ");
        try std.fmt.formatInt(record.id, 10, .lower, .{}, writer);

        // Write vector
        try writer.writeAll(", .vector = .{ ");
        for (record.vector, 0..) |v, i| {
            if (i > 0) try writer.writeAll(", ");
            try std.fmt.formatFloat(writer, v, .{});
        }
        try writer.writeAll(" }");

        // Optional metadata
        if (record.metadata) |meta| {
            try writer.writeAll(", .metadata = ");
            try writeZonString(writer, meta);
        }

        // Optional text
        if (record.text) |txt| {
            try writer.writeAll(", .text = ");
            try writeZonString(writer, txt);
        }

        try writer.writeAll(" }");
    }

    /// Import batch records from ZON data.
    pub fn importRecords(self: *ZonFormat, data: []const u8) ![]batch.BatchRecord {
        const db = try self.parse(data);
        defer self.freeDatabase(db);

        var records = try self.allocator.alloc(batch.BatchRecord, db.records.len);
        errdefer self.allocator.free(records);

        var copied: usize = 0;
        errdefer {
            for (records[0..copied]) |rec| {
                self.allocator.free(rec.vector);
                if (rec.metadata) |m| self.allocator.free(m);
                if (rec.text) |t| self.allocator.free(t);
            }
        }

        for (db.records, 0..) |zon_rec, i| {
            const vector_copy = try self.allocator.dupe(f32, zon_rec.vector);
            errdefer self.allocator.free(vector_copy);

            var meta_copy: ?[]u8 = null;
            if (zon_rec.metadata) |m| {
                meta_copy = try self.allocator.dupe(u8, m);
            }
            errdefer if (meta_copy) |m| self.allocator.free(m);

            var text_copy: ?[]u8 = null;
            if (zon_rec.text) |t| {
                text_copy = try self.allocator.dupe(u8, t);
            }

            records[i] = .{
                .id = zon_rec.id,
                .vector = vector_copy,
                .metadata = meta_copy,
                .text = text_copy,
            };
            copied += 1;
        }

        return records;
    }

    /// Export batch records to ZON data.
    pub fn exportRecords(self: *ZonFormat, records: []const batch.BatchRecord) ![]u8 {
        // Convert batch records to ZON records
        var zon_records = try self.allocator.alloc(ZonRecord, records.len);
        defer self.allocator.free(zon_records);

        // Infer dimension from first record
        const dimension: u32 = if (records.len > 0) @intCast(records[0].vector.len) else 0;

        for (records, 0..) |rec, i| {
            zon_records[i] = .{
                .id = rec.id,
                .vector = rec.vector,
                .metadata = rec.metadata,
                .text = rec.text,
            };
        }

        const db = ZonDatabase{
            .version = ZON_FORMAT_VERSION,
            .name = self.config.name,
            .dimension = dimension,
            .distance_metric = self.config.distance_metric,
            .created_at = time.unixSeconds(),
            .modified_at = time.unixSeconds(),
            .db_metadata = self.config.metadata,
            .records = zon_records,
        };

        return self.serialize(db);
    }

    /// Free a ZonDatabase structure.
    pub fn freeDatabase(self: *ZonFormat, db: ZonDatabase) void {
        self.allocator.free(db.name);
        if (db.db_metadata) |m| self.allocator.free(m);
        for (db.records) |record| {
            self.allocator.free(record.vector);
            if (record.metadata) |m| self.allocator.free(m);
            if (record.text) |t| self.allocator.free(t);
        }
        self.allocator.free(db.records);
    }
};

/// Write a ZON-escaped string literal.
fn writeZonString(writer: anytype, str: []const u8) !void {
    try writer.writeAll("\"");
    for (str) |c| {
        switch (c) {
            '\\' => try writer.writeAll("\\\\"),
            '"' => try writer.writeAll("\\\""),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => {
                if (c >= 0x20 and c < 0x7f) {
                    try writer.writeByte(c);
                } else {
                    try writer.writeAll("\\x");
                    const hex_chars = "0123456789abcdef";
                    try writer.writeByte(hex_chars[c >> 4]);
                    try writer.writeByte(hex_chars[c & 0x0f]);
                }
            },
        }
    }
    try writer.writeAll("\"");
}

/// Convenience function to export batch records to ZON.
pub fn exportToZon(allocator: std.mem.Allocator, records: []const batch.BatchRecord) ![]u8 {
    var format = ZonFormat.init(allocator);
    return format.exportRecords(records);
}

/// Convenience function to import batch records from ZON.
pub fn importFromZon(allocator: std.mem.Allocator, data: []const u8) ![]batch.BatchRecord {
    var format = ZonFormat.init(allocator);
    return format.importRecords(data);
}

// ============================================================================
// Tests
// ============================================================================

test "zon format basic serialize" {
    const allocator = std.testing.allocator;
    var format = ZonFormat.init(allocator);

    const vector1 = [_]f32{ 0.1, 0.2, 0.3 };
    const vector2 = [_]f32{ 0.4, 0.5, 0.6 };

    const records = [_]ZonRecord{
        .{ .id = 1, .vector = &vector1, .metadata = "doc1" },
        .{ .id = 2, .vector = &vector2, .metadata = "doc2" },
    };

    const db = ZonDatabase{
        .name = "test_db",
        .dimension = 3,
        .records = &records,
    };

    const serialized = try format.serialize(db);
    defer allocator.free(serialized);

    // Verify output contains expected fields
    try std.testing.expect(std.mem.indexOf(u8, serialized, ".version = 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, serialized, ".name = \"test_db\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, serialized, ".dimension = 3") != null);
    try std.testing.expect(std.mem.indexOf(u8, serialized, ".id = 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, serialized, ".id = 2") != null);
}

test "zon format export batch records" {
    const allocator = std.testing.allocator;
    var format = ZonFormat.init(allocator);

    const vector1 = [_]f32{ 1.0, 2.0, 3.0 };
    const vector2 = [_]f32{ 4.0, 5.0, 6.0 };

    const records = [_]batch.BatchRecord{
        .{ .id = 100, .vector = &vector1, .metadata = "test1" },
        .{ .id = 200, .vector = &vector2, .text = "hello" },
    };

    const exported = try format.exportRecords(&records);
    defer allocator.free(exported);

    // Verify it's valid ZON structure
    try std.testing.expect(std.mem.startsWith(u8, exported, ".{"));
    try std.testing.expect(std.mem.indexOf(u8, exported, ".records = .{") != null);
}

test "zon string escaping" {
    const allocator = std.testing.allocator;
    var aw = std.Io.Writer.Allocating.init(allocator);
    defer aw.deinit();

    try writeZonString(&aw.writer, "hello\nworld\t\"test\"\\path");
    const result = try aw.toOwnedSlice();
    defer allocator.free(result);

    try std.testing.expectEqualStrings("\"hello\\nworld\\t\\\"test\\\"\\\\path\"", result);
}

test "zon format version constant" {
    try std.testing.expectEqual(@as(u32, 1), ZON_FORMAT_VERSION);
}

test "distance metric enum" {
    // Use {t} format specifier instead of @tagName (Zig 0.16)
    var buf: [32]u8 = undefined;
    try std.testing.expectEqualStrings("euclidean", std.fmt.bufPrint(&buf, "{t}", .{DistanceMetric.euclidean}) catch "");
    try std.testing.expectEqualStrings("cosine", std.fmt.bufPrint(&buf, "{t}", .{DistanceMetric.cosine}) catch "");
    try std.testing.expectEqualStrings("dot_product", std.fmt.bufPrint(&buf, "{t}", .{DistanceMetric.dot_product}) catch "");
    try std.testing.expectEqualStrings("manhattan", std.fmt.bufPrint(&buf, "{t}", .{DistanceMetric.manhattan}) catch "");
}
