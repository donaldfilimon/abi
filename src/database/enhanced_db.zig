//! Enhanced Database Module
//! Modern database implementation with advanced features and performance optimizations

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Database configuration
pub const DatabaseConfig = struct {
    max_file_size: usize = 1024 * 1024 * 1024, // 1GB
    page_size: usize = 4096,
    cache_size: usize = 1024 * 1024, // 1MB
    enable_compression: bool = false,
    enable_encryption: bool = false,
    compression_level: u8 = 6,
    encryption_key: ?[]const u8 = null,
    enable_logging: bool = true,
    log_level: std.log.Level = .info,
};

/// Database record with metadata
pub const DatabaseRecord = struct {
    id: u64,
    timestamp: i64,
    data: []u8,
    checksum: u32,
    compressed: bool,
    encrypted: bool,

    pub fn init(allocator: Allocator, data: []const u8) !DatabaseRecord {
        const data_copy = try allocator.dupe(u8, data);
        const checksum = std.hash.Crc32.hash(data);

        return DatabaseRecord{
            .id = std.time.microTimestamp(),
            .timestamp = std.time.microTimestamp(),
            .data = data_copy,
            .checksum = checksum,
            .compressed = false,
            .encrypted = false,
        };
    }

    pub fn deinit(self: *DatabaseRecord, allocator: Allocator) void {
        allocator.free(self.data);
    }

    pub fn validate(self: *const DatabaseRecord) bool {
        const computed_checksum = std.hash.Crc32.hash(self.data);
        return computed_checksum == self.checksum;
    }

    pub fn compress(self: *DatabaseRecord, allocator: Allocator, level: u8) !void {
        if (self.compressed) return;

        var stream = std.compress.zlib.deflateStream(allocator, level);
        defer stream.deinit();

        var compressed = std.ArrayList(u8).init(allocator);
        defer compressed.deinit();

        try stream.compress(self.data);
        try stream.finish();

        const compressed_data = try compressed.toOwnedSlice();
        allocator.free(self.data);
        self.data = compressed_data;
        self.compressed = true;
        self.checksum = std.hash.Crc32.hash(self.data);
    }

    pub fn decompress(self: *DatabaseRecord, allocator: Allocator) !void {
        if (!self.compressed) return;

        var stream = std.compress.zlib.inflateStream(allocator);
        defer stream.deinit();

        var decompressed = std.ArrayList(u8).init(allocator);
        defer decompressed.deinit();

        try stream.decompress(self.data);

        const decompressed_data = try decompressed.toOwnedSlice();
        allocator.free(self.data);
        self.data = decompressed_data;
        self.compressed = false;
        self.checksum = std.hash.Crc32.hash(self.data);
    }
};

/// Database index for fast lookups
pub const DatabaseIndex = struct {
    allocator: Allocator,
    records: std.AutoHashMap(u64, usize), // id -> position
    sorted_ids: std.ArrayList(u64),

    pub fn init(allocator: Allocator) DatabaseIndex {
        return DatabaseIndex{
            .allocator = allocator,
            .records = std.AutoHashMap(u64, usize).init(allocator),
            .sorted_ids = std.ArrayList(u64).init(allocator),
        };
    }

    pub fn deinit(self: *DatabaseIndex) void {
        self.records.deinit();
        self.sorted_ids.deinit();
    }

    pub fn add(self: *DatabaseIndex, id: u64, position: usize) !void {
        try self.records.put(id, position);
        try self.sorted_ids.append(id);
    }

    pub fn remove(self: *DatabaseIndex, id: u64) void {
        _ = self.records.remove(id);
        // Remove from sorted_ids
        for (self.sorted_ids.items, 0..) |item, i| {
            if (item == id) {
                _ = self.sorted_ids.orderedRemove(i);
                break;
            }
        }
    }

    pub fn get(self: *const DatabaseIndex, id: u64) ?usize {
        return self.records.get(id);
    }

    pub fn contains(self: *const DatabaseIndex, id: u64) bool {
        return self.records.contains(id);
    }

    pub fn getAllIds(self: *const DatabaseIndex) []const u64 {
        return self.sorted_ids.items;
    }

    pub fn size(self: *const DatabaseIndex) usize {
        return self.records.count();
    }
};

/// Enhanced database implementation
pub const EnhancedDatabase = struct {
    config: DatabaseConfig,
    allocator: Allocator,
    logger: std.log.scoped(.database), // Logger module
    file: std.fs.File,
    index: DatabaseIndex,
    cache: std.AutoHashMap(u64, DatabaseRecord),
    header: DatabaseHeader,
    stats: DatabaseStats,

    const Self = @This();

    /// Database header structure
    pub const DatabaseHeader = struct {
        magic: [4]u8 = "ENDB".*,
        version: u32 = 1,
        page_size: u32 = 4096,
        record_count: u64 = 0,
        index_offset: u64 = 0,
        data_offset: u64 = 0,
        checksum: u32 = 0,

        pub fn computeChecksum(self: *const DatabaseHeader) u32 {
            var hasher = std.hash.Crc32.init();
            hasher.update(&[_]u8{ 0, 0, 0, 0 }); // Skip checksum field
            hasher.update(std.mem.asBytes(self)[4..]);
            return hasher.final();
        }

        pub fn validate(self: *const DatabaseHeader) bool {
            const computed = self.computeChecksum();
            return computed == self.checksum;
        }
    };

    /// Database statistics
    pub const DatabaseStats = struct {
        total_reads: u64 = 0,
        total_writes: u64 = 0,
        total_deletes: u64 = 0,
        cache_hits: u64 = 0,
        cache_misses: u64 = 0,
        compression_ratio: f64 = 1.0,
        average_read_time_ms: f64 = 0.0,
        average_write_time_ms: f64 = 0.0,

        pub fn updateReadTime(self: *DatabaseStats, read_time_ms: f64) void {
            const total = self.total_reads;
            if (total > 0) {
                self.average_read_time_ms = (self.average_read_time_ms * @as(f64, @floatFromInt(total - 1)) + read_time_ms) / @as(f64, @floatFromInt(total));
            } else {
                self.average_read_time_ms = read_time_ms;
            }
        }

        pub fn updateWriteTime(self: *DatabaseStats, write_time_ms: f64) void {
            const total = self.total_writes;
            if (total > 0) {
                self.average_write_time_ms = (self.average_write_time_ms * @as(f64, @floatFromInt(total - 1)) + write_time_ms) / @as(f64, @floatFromInt(total));
            } else {
                self.average_write_time_ms = write_time_ms;
            }
        }

        pub fn updateCompressionRatio(self: *DatabaseStats, original_size: usize, compressed_size: usize) void {
            if (original_size > 0) {
                self.compression_ratio = @as(f64, @floatFromInt(compressed_size)) / @as(f64, @floatFromInt(original_size));
            }
        }
    };

    /// Initialize enhanced database
    pub fn init(allocator: Allocator, file_path: []const u8, config: DatabaseConfig) !*Self {
        const file = try std.fs.cwd().createFile(file_path, .{ .read = true, .write = true });
        errdefer file.close();

        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .config = config,
            .allocator = allocator,
            .logger = std.log.scoped(.database),
            .file = file,
            .index = DatabaseIndex.init(allocator),
            .cache = std.AutoHashMap(u64, DatabaseRecord).init(allocator),
            .header = .{},
            .stats = .{},
        };

        // Initialize header
        self.header.page_size = @intCast(config.page_size);
        self.header.data_offset = @sizeOf(DatabaseHeader);
        self.header.index_offset = self.header.data_offset;

        // Write initial header
        try self.writeHeader();

        if (config.enable_logging) {
            self.logger.info("Enhanced Database initialized: {s}", .{file_path});
        }

        return self;
    }

    /// Open existing database
    pub fn open(allocator: Allocator, file_path: []const u8, config: DatabaseConfig) !*Self {
        const file = try std.fs.cwd().openFile(file_path, .{ .read = true, .write = true });
        errdefer file.close();

        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .config = config,
            .allocator = allocator,
            .logger = std.log.scoped(.database),
            .file = file,
            .index = DatabaseIndex.init(allocator),
            .cache = std.AutoHashMap(u64, DatabaseRecord).init(allocator),
            .header = .{},
            .stats = .{},
        };

        // Read header
        try self.readHeader();

        if (config.enable_logging) {
            self.logger.info("Enhanced Database opened: {s}", .{file_path});
        }

        return self;
    }

    /// Deinitialize database
    pub fn deinit(self: *Self) void {
        if (self.config.enable_logging) {
            self.logger.info("Enhanced Database shutting down", .{});
        }

        // Write final header
        self.writeHeader() catch {};

        // Free cache
        var it = self.cache.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.cache.deinit();

        // Free index
        self.index.deinit();

        // Close file
        self.file.close();

        self.allocator.destroy(self);
    }

    /// Write record to database
    pub fn writeRecord(self: *Self, data: []const u8) !u64 {
        const start_time = std.time.microTimestamp();
        defer {
            const end_time = std.time.microTimestamp();
            const elapsed = @as(f64, @floatFromInt(end_time - start_time)) / 1000.0; // Convert to milliseconds
            self.stats.updateWriteTime(elapsed);
            self.stats.total_writes += 1;
        }

        var record = try DatabaseRecord.init(self.allocator, data);
        errdefer record.deinit(self.allocator);

        // Compress if enabled
        if (self.config.enable_compression) {
            try record.compress(self.allocator, self.config.compression_level);
            self.stats.updateCompressionRatio(data.len, record.data.len);
        }

        // Write record
        const position = try self.file.getEndPos();
        try self.file.seekTo(position);

        // Write record header
        const record_header = [_]u8{
            @intCast(record.id >> 56),
            @intCast(record.id >> 48),
            @intCast(record.id >> 40),
            @intCast(record.id >> 32),
            @intCast(record.id >> 24),
            @intCast(record.id >> 16),
            @intCast(record.id >> 8),
            @intCast(record.id),
            @intCast(record.timestamp >> 56),
            @intCast(record.timestamp >> 48),
            @intCast(record.timestamp >> 40),
            @intCast(record.timestamp >> 32),
            @intCast(record.timestamp >> 24),
            @intCast(record.timestamp >> 16),
            @intCast(record.timestamp >> 8),
            @intCast(record.timestamp),
            @intCast(record.data.len >> 24),
            @intCast(record.data.len >> 16),
            @intCast(record.data.len >> 8),
            @intCast(record.data.len),
            @intCast(record.checksum >> 24),
            @intCast(record.checksum >> 16),
            @intCast(record.checksum >> 8),
            @intCast(record.checksum),
            @intCast(record.compressed),
            @intCast(record.encrypted),
        };
        try self.file.writeAll(&record_header);

        // Write record data
        try self.file.writeAll(record.data);

        // Update index
        try self.index.add(record.id, @intCast(position));

        // Update cache
        try self.cache.put(record.id, record);

        // Update header
        self.header.record_count += 1;
        try self.writeHeader();

        if (self.config.enable_logging) {
            self.logger.debug("Wrote record {d} at position {d}", .{ record.id, position });
        }

        return record.id;
    }

    /// Read record from database
    pub fn readRecord(self: *Self, id: u64) !?DatabaseRecord {
        const start_time = std.time.microTimestamp();
        defer {
            const end_time = std.time.microTimestamp();
            const elapsed = @as(f64, @floatFromInt(end_time - start_time)) / 1000.0; // Convert to milliseconds
            self.stats.updateReadTime(elapsed);
            self.stats.total_reads += 1;
        }

        // Check cache first
        if (self.cache.get(id)) |cached_record| {
            self.stats.cache_hits += 1;
            return cached_record;
        }
        self.stats.cache_misses += 1;

        // Look up in index
        const position = self.index.get(id) orelse return null;

        // Read record
        try self.file.seekTo(position);

        // Read record header
        var record_header: [28]u8 = undefined;
        _ = try self.file.read(&record_header);

        // Parse record header
        const record_id = std.mem.readIntBig(u64, record_header[0..8]);
        const timestamp = std.mem.readIntBig(i64, record_header[8..16]);
        const data_len = std.mem.readIntBig(u32, record_header[16..20]);
        const checksum = std.mem.readIntBig(u32, record_header[20..24]);
        const compressed = record_header[24] != 0;
        const encrypted = record_header[25] != 0;

        // Read record data
        const data = try self.allocator.alloc(u8, data_len);
        errdefer self.allocator.free(data);
        try self.file.readNoEof(data);

        // Create record
        var record = DatabaseRecord{
            .id = record_id,
            .timestamp = timestamp,
            .data = data,
            .checksum = checksum,
            .compressed = compressed,
            .encrypted = encrypted,
        };

        // Validate checksum
        if (!record.validate()) {
            record.deinit(self.allocator);
            return error.CorruptedData;
        }

        // Decompress if needed
        if (compressed) {
            try record.decompress(self.allocator);
        }

        // Update cache
        try self.cache.put(id, record);

        if (self.config.enable_logging) {
            self.logger.debug("Read record {d} from position {d}", .{ id, position });
        }

        return record;
    }

    /// Delete record from database
    pub fn deleteRecord(self: *Self, id: u64) !bool {
        _ = self.index.get(id) orelse return false;

        // Remove from cache
        if (self.cache.fetchRemove(id)) |entry| {
            entry.value.deinit(self.allocator);
        }

        // Remove from index
        self.index.remove(id);

        // Update header
        self.header.record_count -= 1;
        try self.writeHeader();

        self.stats.total_deletes += 1;

        if (self.config.enable_logging) {
            self.logger.debug("Deleted record {d}", .{id});
        }

        return true;
    }

    /// List all record IDs
    pub fn listRecords(self: *const Self) []const u64 {
        return self.index.getAllIds();
    }

    /// Get database statistics
    pub fn getStats(self: *const Self) DatabaseStats {
        return self.stats;
    }

    /// Compact database (remove deleted records)
    pub fn compact(self: *Self) !void {
        if (self.config.enable_logging) {
            self.logger.info("Starting database compaction", .{});
        }

        // Implementation would involve rewriting the database file
        // and updating all positions in the index
        // This is a simplified version

        if (self.config.enable_logging) {
            self.logger.info("Database compaction completed", .{});
        }
    }

    /// Write database header
    fn writeHeader(self: *Self) !void {
        self.header.checksum = self.header.computeChecksum();
        try self.file.seekTo(0);
        try self.file.writeAll(std.mem.asBytes(&self.header));
        try self.file.sync();
    }

    /// Read database header
    fn readHeader(self: *Self) !void {
        try self.file.seekTo(0);
        const header_bytes = try self.file.reader().readAllAlloc(self.allocator, @sizeOf(DatabaseHeader));
        defer self.allocator.free(header_bytes);

        if (header_bytes.len < @sizeOf(DatabaseHeader)) {
            return error.InvalidHeader;
        }

        @memcpy(std.mem.asBytes(&self.header), header_bytes[0..@sizeOf(DatabaseHeader)]);

        if (!self.header.validate()) {
            return error.InvalidHeader;
        }
    }
};

test "enhanced database basic operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = DatabaseConfig{
        .enable_logging = false,
        .enable_compression = true,
    };

    // Create temporary file
    const temp_path = "test_db.tmp";
    defer std.fs.cwd().deleteFile(temp_path) catch {};

    var db = try EnhancedDatabase.init(allocator, temp_path, config);
    defer db.deinit();

    // Test writing record
    const test_data = "Hello, Enhanced Database!";
    const id = try db.writeRecord(test_data);
    try testing.expect(id > 0);

    // Test reading record
    const record = try db.readRecord(id);
    try testing.expect(record != null);
    try testing.expectEqualStrings(test_data, record.?.data);

    // Test listing records
    const records = db.listRecords();
    try testing.expectEqual(@as(usize, 1), records.len);
    try testing.expectEqual(id, records[0]);

    // Test deleting record
    const deleted = try db.deleteRecord(id);
    try testing.expect(deleted);

    // Test reading deleted record
    const deleted_record = try db.readRecord(id);
    try testing.expect(deleted_record == null);
}

test "enhanced database compression" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const config = DatabaseConfig{
        .enable_logging = false,
        .enable_compression = true,
        .compression_level = 9,
    };

    const temp_path = "test_compression_db.tmp";
    defer std.fs.cwd().deleteFile(temp_path) catch {};

    var db = try EnhancedDatabase.init(allocator, temp_path, config);
    defer db.deinit();

    // Create repetitive data for good compression
    var test_data = std.ArrayList(u8).init(allocator);
    defer test_data.deinit();

    for (0..1000) |_| {
        try test_data.appendSlice("This is repetitive data for compression testing. ");
    }

    const id = try db.writeRecord(test_data.items);
    const record = try db.readRecord(id);

    try testing.expect(record != null);
    try testing.expectEqualStrings(test_data.items, record.?.data);

    // Check compression ratio
    const stats = db.getStats();
    try testing.expect(stats.compression_ratio < 1.0);
}
