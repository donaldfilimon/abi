//! Batch Import/Export for Vector Database
//!
//! File import and export support for batch operations.
//! Supports JSON Lines, CSV, and ZON (Zig Object Notation) formats.

const std = @import("std");
const batch = @import("batch.zig");

const BatchRecord = batch.BatchRecord;
const BatchConfig = batch.BatchConfig;

/// Import/export formats.
pub const ImportFormat = enum {
    json,
    csv,
    parquet,
    npy,
    binary,
    zon,
};

/// Batch importer for file imports.
pub const BatchImporter = struct {
    allocator: std.mem.Allocator,
    config: BatchConfig,
    format: ImportFormat,

    pub fn init(allocator: std.mem.Allocator, format: ImportFormat, config: BatchConfig) BatchImporter {
        return .{
            .allocator = allocator,
            .config = config,
            .format = format,
        };
    }

    /// Import from JSON lines format (JSONL).
    /// Each line should be a JSON object with fields: id, vector, metadata (optional), text (optional)
    pub fn importJsonLines(self: *BatchImporter, data: []const u8) ![]BatchRecord {
        var records = std.ArrayListUnmanaged(BatchRecord).empty;
        errdefer {
            for (records.items) |record| {
                self.allocator.free(record.vector);
                if (record.metadata) |m| self.allocator.free(m);
                if (record.text) |t| self.allocator.free(t);
            }
            records.deinit(self.allocator);
        }

        var line_iter = std.mem.splitScalar(u8, data, '\n');
        while (line_iter.next()) |line| {
            // Skip empty lines
            const trimmed = std.mem.trim(u8, line, &std.ascii.whitespace);
            if (trimmed.len == 0) continue;

            // Parse JSON line
            const parsed = std.json.parseFromSlice(
                std.json.Value,
                self.allocator,
                trimmed,
                .{},
            ) catch |err| {
                std.log.warn("Failed to parse JSON line: {}", .{err});
                continue;
            };
            defer parsed.deinit();

            const obj = parsed.value.object;

            // Extract ID (required)
            const id_value = obj.get("id") orelse continue;
            const id: u64 = switch (id_value) {
                .integer => |i| @intCast(i),
                .number_string => |s| std.fmt.parseInt(u64, s, 10) catch continue,
                else => continue,
            };

            // Extract vector (required)
            const vector_value = obj.get("vector") orelse continue;
            if (vector_value != .array) continue;

            var vector_data = try self.allocator.alloc(f32, vector_value.array.items.len);
            errdefer self.allocator.free(vector_data);

            for (vector_value.array.items, 0..) |v, i| {
                vector_data[i] = switch (v) {
                    .float => |f| @floatCast(f),
                    .integer => |int| @floatFromInt(int),
                    .number_string => |s| std.fmt.parseFloat(f32, s) catch continue,
                    else => continue,
                };
            }

            // Extract metadata (optional)
            var metadata: ?[]u8 = null;
            if (obj.get("metadata")) |meta_value| {
                if (meta_value == .string) {
                    metadata = try self.allocator.dupe(u8, meta_value.string);
                }
            }
            errdefer if (metadata) |m| self.allocator.free(m);

            // Extract text (optional)
            var text: ?[]u8 = null;
            if (obj.get("text")) |text_value| {
                if (text_value == .string) {
                    text = try self.allocator.dupe(u8, text_value.string);
                }
            }
            errdefer if (text) |t| self.allocator.free(t);

            try records.append(self.allocator, .{
                .id = id,
                .vector = vector_data,
                .metadata = metadata,
                .text = text,
            });
        }

        return records.toOwnedSlice(self.allocator);
    }

    /// Import from CSV format.
    /// Format: id,vector[...],metadata,text
    /// Vector is comma-separated floats in square brackets or space-separated
    pub fn importCsv(self: *BatchImporter, data: []const u8) ![]BatchRecord {
        var records = std.ArrayListUnmanaged(BatchRecord).empty;
        errdefer {
            for (records.items) |record| {
                self.allocator.free(record.vector);
                if (record.metadata) |m| self.allocator.free(m);
                if (record.text) |t| self.allocator.free(t);
            }
            records.deinit(self.allocator);
        }

        var line_iter = std.mem.splitScalar(u8, data, '\n');
        var line_num: usize = 0;

        while (line_iter.next()) |line| {
            line_num += 1;

            // Skip empty lines and header
            const trimmed = std.mem.trim(u8, line, &std.ascii.whitespace);
            if (trimmed.len == 0) continue;
            if (line_num == 1 and std.mem.indexOf(u8, trimmed, "id") != null) continue; // Skip header

            // Split by comma (simple CSV parsing - doesn't handle quoted fields with commas)
            var field_iter = std.mem.splitScalar(u8, trimmed, ',');

            // Field 1: ID
            const id_str = field_iter.next() orelse continue;
            const id = std.fmt.parseInt(u64, std.mem.trim(u8, id_str, &std.ascii.whitespace), 10) catch |err| {
                std.log.warn("Line {d}: Invalid ID: {}", .{ line_num, err });
                continue;
            };

            // Field 2: Vector (could be in brackets or space-separated)
            const vector_str = field_iter.next() orelse continue;
            const vector_trimmed = std.mem.trim(u8, vector_str, &std.ascii.whitespace);

            // Parse vector
            var vector_list = std.ArrayListUnmanaged(f32).empty;
            defer vector_list.deinit(self.allocator);

            // Check if vector is in brackets [1.0,2.0,3.0] or just space/comma separated
            var vec_data = vector_trimmed;
            if (std.mem.startsWith(u8, vec_data, "[")) {
                vec_data = vec_data[1..];
            }
            if (std.mem.endsWith(u8, vec_data, "]")) {
                vec_data = vec_data[0 .. vec_data.len - 1];
            }

            var value_iter = std.mem.tokenizeAny(u8, vec_data, ", ");
            while (value_iter.next()) |val_str| {
                const val = std.fmt.parseFloat(f32, std.mem.trim(u8, val_str, &std.ascii.whitespace)) catch |err| {
                    std.log.warn("Line {d}: Invalid vector value '{s}': {}", .{ line_num, val_str, err });
                    continue;
                };
                try vector_list.append(self.allocator, val);
            }

            if (vector_list.items.len == 0) {
                std.log.warn("Line {d}: Empty vector", .{line_num});
                continue;
            }

            const vector_data = try vector_list.toOwnedSlice(self.allocator);
            errdefer self.allocator.free(vector_data);

            // Field 3: Metadata (optional)
            var metadata: ?[]u8 = null;
            if (field_iter.next()) |meta_str| {
                const meta_trimmed = std.mem.trim(u8, meta_str, &std.ascii.whitespace);
                if (meta_trimmed.len > 0) {
                    metadata = try self.allocator.dupe(u8, meta_trimmed);
                }
            }
            errdefer if (metadata) |m| self.allocator.free(m);

            // Field 4: Text (optional)
            var text: ?[]u8 = null;
            if (field_iter.next()) |text_str| {
                const text_trimmed = std.mem.trim(u8, text_str, &std.ascii.whitespace);
                if (text_trimmed.len > 0) {
                    text = try self.allocator.dupe(u8, text_trimmed);
                }
            }
            errdefer if (text) |t| self.allocator.free(t);

            try records.append(self.allocator, .{
                .id = id,
                .vector = vector_data,
                .metadata = metadata,
                .text = text,
            });
        }

        return records.toOwnedSlice(self.allocator);
    }

    /// Export records to JSON lines format.
    pub fn exportJsonLines(self: *BatchImporter, records: []const BatchRecord) ![]u8 {
        var aw = std.Io.Writer.Allocating.init(self.allocator);
        errdefer aw.deinit();
        const writer = &aw.writer;

        for (records) |record| {
            try writer.writeAll("{\"id\":");
            try std.fmt.formatInt(record.id, 10, .lower, .{}, writer);

            try writer.writeAll(",\"vector\":[");
            for (record.vector, 0..) |v, i| {
                if (i > 0) try writer.writeAll(",");
                try std.fmt.formatFloat(writer, v, .{});
            }
            try writer.writeAll("]");

            if (record.metadata) |meta| {
                try writer.writeAll(",\"metadata\":");
                try std.json.encodeJsonString(meta, .{}, writer);
            }

            if (record.text) |txt| {
                try writer.writeAll(",\"text\":");
                try std.json.encodeJsonString(txt, .{}, writer);
            }

            try writer.writeAll("}\n");
        }

        return aw.toOwnedSlice();
    }

    /// Export records to CSV format.
    pub fn exportCsv(self: *BatchImporter, records: []const BatchRecord) ![]u8 {
        var aw = std.Io.Writer.Allocating.init(self.allocator);
        errdefer aw.deinit();
        const writer = &aw.writer;

        // Write header
        try writer.writeAll("id,vector,metadata,text\n");

        for (records) |record| {
            // Write ID
            try std.fmt.formatInt(record.id, 10, .lower, .{}, writer);
            try writer.writeAll(",");

            // Write vector
            try writer.writeAll("[");
            for (record.vector, 0..) |v, i| {
                if (i > 0) try writer.writeAll(" ");
                try std.fmt.formatFloat(writer, v, .{});
            }
            try writer.writeAll("]");
            try writer.writeAll(",");

            // Write metadata (escaped if contains commas)
            if (record.metadata) |meta| {
                if (std.mem.indexOf(u8, meta, ",") != null) {
                    try writer.writeAll("\"");
                    try writer.writeAll(meta);
                    try writer.writeAll("\"");
                } else {
                    try writer.writeAll(meta);
                }
            }
            try writer.writeAll(",");

            // Write text (escaped if contains commas)
            if (record.text) |txt| {
                if (std.mem.indexOf(u8, txt, ",") != null) {
                    try writer.writeAll("\"");
                    try writer.writeAll(txt);
                    try writer.writeAll("\"");
                } else {
                    try writer.writeAll(txt);
                }
            }

            try writer.writeAll("\n");
        }

        return aw.toOwnedSlice();
    }

    /// Import from ZON (Zig Object Notation) format.
    /// Expects format:
    /// .{
    ///     .records = .{
    ///         .{ .id = 1, .vector = .{ 0.1, 0.2, 0.3 }, .metadata = "label" },
    ///         ...
    ///     },
    /// }
    /// Or simplified array format:
    /// .{
    ///     .{ .id = 1, .vector = .{ 0.1, 0.2, 0.3 } },
    ///     ...
    /// }
    pub fn importZon(self: *BatchImporter, data: []const u8) ![]BatchRecord {
        var records = std.ArrayListUnmanaged(BatchRecord).empty;
        errdefer {
            for (records.items) |record| {
                self.allocator.free(record.vector);
                if (record.metadata) |m| self.allocator.free(m);
                if (record.text) |t| self.allocator.free(t);
            }
            records.deinit(self.allocator);
        }

        // Parse using std.zon
        const ZonRecord = struct {
            id: u64 = 0,
            vector: []const f32 = &.{},
            metadata: ?[]const u8 = null,
            text: ?[]const u8 = null,
        };

        const ZonWrapper = struct {
            records: []const ZonRecord = &.{},
        };

        // ZON parser requires sentinel-terminated source
        const zon_source = std.fmt.allocPrintSentinel(self.allocator, "{s}", .{data}, 0) catch
            return error.OutOfMemory;
        defer self.allocator.free(zon_source);

        // Try parsing as wrapped format first (with .records field)
        const parsed_wrapped = std.zon.parse.fromSliceAlloc(ZonWrapper, self.allocator, zon_source, null, .{});
        if (parsed_wrapped) |parsed| {
            defer std.zon.parse.free(self.allocator, parsed);

            for (parsed.records) |zon_record| {
                const vector_data = try self.allocator.dupe(f32, zon_record.vector);
                errdefer self.allocator.free(vector_data);

                var metadata: ?[]u8 = null;
                if (zon_record.metadata) |m| {
                    metadata = try self.allocator.dupe(u8, m);
                }
                errdefer if (metadata) |m| self.allocator.free(m);

                var text: ?[]u8 = null;
                if (zon_record.text) |t| {
                    text = try self.allocator.dupe(u8, t);
                }
                errdefer if (text) |t| self.allocator.free(t);

                try records.append(self.allocator, .{
                    .id = zon_record.id,
                    .vector = vector_data,
                    .metadata = metadata,
                    .text = text,
                });
            }
            return records.toOwnedSlice(self.allocator);
        } else |_| {
            // Try parsing as direct array format
            const parsed_array = std.zon.parse.fromSliceAlloc([]const ZonRecord, self.allocator, zon_source, null, .{}) catch |err| {
                std.log.warn("Failed to parse ZON data: {t}", .{err});
                return err;
            };
            defer std.zon.parse.free(self.allocator, parsed_array);

            for (parsed_array) |zon_record| {
                const vector_data = try self.allocator.dupe(f32, zon_record.vector);
                errdefer self.allocator.free(vector_data);

                var metadata: ?[]u8 = null;
                if (zon_record.metadata) |m| {
                    metadata = try self.allocator.dupe(u8, m);
                }
                errdefer if (metadata) |m| self.allocator.free(m);

                var text: ?[]u8 = null;
                if (zon_record.text) |t| {
                    text = try self.allocator.dupe(u8, t);
                }
                errdefer if (text) |t| self.allocator.free(t);

                try records.append(self.allocator, .{
                    .id = zon_record.id,
                    .vector = vector_data,
                    .metadata = metadata,
                    .text = text,
                });
            }
            return records.toOwnedSlice(self.allocator);
        }
    }

    /// Export records to ZON (Zig Object Notation) format.
    /// Creates human-readable, Zig-native serialization.
    pub fn exportZon(self: *BatchImporter, records: []const BatchRecord) ![]u8 {
        var aw = std.Io.Writer.Allocating.init(self.allocator);
        errdefer aw.deinit();
        const writer = &aw.writer;

        try writer.writeAll(".{\n    .records = .{\n");

        for (records, 0..) |record, idx| {
            try writer.writeAll("        .{\n");

            // Write ID
            try writer.writeAll("            .id = ");
            try std.fmt.formatInt(record.id, 10, .lower, .{}, writer);
            try writer.writeAll(",\n");

            // Write vector
            try writer.writeAll("            .vector = .{ ");
            for (record.vector, 0..) |v, i| {
                if (i > 0) try writer.writeAll(", ");
                try std.fmt.formatFloat(writer, v, .{});
            }
            try writer.writeAll(" },\n");

            // Write metadata (optional)
            if (record.metadata) |meta| {
                try writer.writeAll("            .metadata = ");
                try writeZonString(writer, meta);
                try writer.writeAll(",\n");
            }

            // Write text (optional)
            if (record.text) |txt| {
                try writer.writeAll("            .text = ");
                try writeZonString(writer, txt);
                try writer.writeAll(",\n");
            }

            try writer.writeAll("        }");
            if (idx < records.len - 1) {
                try writer.writeAll(",");
            }
            try writer.writeAll("\n");
        }

        try writer.writeAll("    },\n}\n");

        return aw.toOwnedSlice();
    }

    /// Helper to write a ZON-escaped string literal.
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
                        // Write as hex escape
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
};
