//! Tests for Batch Operations
//!
//! Covers BatchProcessor, BatchWriter, BatchOperationBuilder,
//! record validation, and ZON import/export.

const std = @import("std");
const testing = std.testing;
const batch = @import("batch.zig");
const batch_importer = @import("batch_importer.zig");

const BatchProcessor = batch.BatchProcessor;
const BatchWriter = batch.BatchWriter;
const BatchOperationBuilder = batch.BatchOperationBuilder;
const BatchRecord = batch.BatchRecord;
const BatchImporter = batch_importer.BatchImporter;
const ImportFormat = batch_importer.ImportFormat;

test "batch processor basic" {
    const allocator = testing.allocator;
    var processor = BatchProcessor.init(allocator, .{ .batch_size = 10 });
    defer processor.deinit();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    try processor.add(.{ .id = 1, .vector = &vector });
    try processor.add(.{ .id = 2, .vector = &vector });

    try processor.flush();

    const stats = processor.getStats();
    try testing.expectEqual(@as(usize, 2), stats.total_inserted);
}

test "batch insert" {
    const allocator = testing.allocator;
    var processor = BatchProcessor.init(allocator, .{});
    defer processor.deinit();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    const records = [_]BatchRecord{
        .{ .id = 1, .vector = &vector },
        .{ .id = 2, .vector = &vector },
        .{ .id = 3, .vector = &vector },
    };

    const result = try processor.insertBatch(&records);
    defer allocator.free(result.failed_ids);
    defer allocator.free(result.errors);

    try testing.expectEqual(@as(usize, 3), result.successful);
    try testing.expect(result.isComplete());
}

test "batch writer" {
    const allocator = testing.allocator;
    var writer = BatchWriter.init(allocator, .{ .batch_size = 2 });
    defer writer.deinit();

    try writer.start();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    try writer.write(.{ .id = 1, .vector = &vector });
    try writer.write(.{ .id = 2, .vector = &vector });
    try writer.write(.{ .id = 3, .vector = &vector });

    const result = try writer.finish();

    try testing.expectEqual(@as(usize, 3), result.total_written);
    try testing.expect(result.batches_processed >= 1);
}

test "batch operation builder" {
    const allocator = testing.allocator;
    var builder = BatchOperationBuilder.init(allocator);
    defer builder.deinit();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    _ = try builder.withBatchSize(100).addRecord(1, &vector);
    _ = try builder.addRecord(2, &vector);

    const result = try builder.execute();
    defer allocator.free(result.failed_ids);
    defer allocator.free(result.errors);

    try testing.expectEqual(@as(usize, 2), result.successful);
}

test "record validation" {
    const allocator = testing.allocator;
    var processor = BatchProcessor.init(allocator, .{ .validate_before_insert = true });
    defer processor.deinit();

    // Empty vector should be invalid
    const empty_vector = [_]f32{};
    const invalid_record = BatchRecord{ .id = 1, .vector = &empty_vector };
    try testing.expect(!processor.validateRecord(invalid_record));

    // NaN should be invalid
    const nan_vector = [_]f32{ 1.0, std.math.nan(f32), 3.0 };
    const nan_record = BatchRecord{ .id = 2, .vector = &nan_vector };
    try testing.expect(!processor.validateRecord(nan_record));

    // Valid vector
    const valid_vector = [_]f32{ 1.0, 2.0, 3.0 };
    const valid_record = BatchRecord{ .id = 3, .vector = &valid_vector };
    try testing.expect(processor.validateRecord(valid_record));
}

test "zon export basic" {
    const allocator = testing.allocator;
    var importer = BatchImporter.init(allocator, .zon, .{});

    const vector1 = [_]f32{ 1.0, 2.0, 3.0 };
    const vector2 = [_]f32{ 4.0, 5.0, 6.0 };

    const records = [_]BatchRecord{
        .{ .id = 1, .vector = &vector1, .metadata = "test1" },
        .{ .id = 2, .vector = &vector2, .text = "hello world" },
    };

    const exported = try importer.exportZon(&records);
    defer allocator.free(exported);

    // Verify ZON structure
    try testing.expect(std.mem.startsWith(u8, exported, ".{"));
    try testing.expect(std.mem.indexOf(u8, exported, ".records = .{") != null);
    try testing.expect(std.mem.indexOf(u8, exported, ".id = 1") != null);
    try testing.expect(std.mem.indexOf(u8, exported, ".id = 2") != null);
    try testing.expect(std.mem.indexOf(u8, exported, ".metadata = \"test1\"") != null);
    try testing.expect(std.mem.indexOf(u8, exported, ".text = \"hello world\"") != null);
}

test "zon string escaping in export" {
    const allocator = testing.allocator;
    var importer = BatchImporter.init(allocator, .zon, .{});

    const vector = [_]f32{ 1.0, 2.0 };
    const records = [_]BatchRecord{
        .{ .id = 1, .vector = &vector, .metadata = "line1\nline2\ttab\"quote\"" },
    };

    const exported = try importer.exportZon(&records);
    defer allocator.free(exported);

    // Verify escaping
    try testing.expect(std.mem.indexOf(u8, exported, "\\n") != null);
    try testing.expect(std.mem.indexOf(u8, exported, "\\t") != null);
    try testing.expect(std.mem.indexOf(u8, exported, "\\\"quote\\\"") != null);
}

test "import format enum includes zon" {
    // Verify zon is in the enum
    const format: ImportFormat = .zon;
    try testing.expectEqual(ImportFormat.zon, format);
}
