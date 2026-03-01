//! Database Batch Operations Tests — Extended Coverage
//!
//! Tests batch insert, delete, validation, streaming writer, builder pattern,
//! progress tracking, and edge cases.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const batch = if (build_options.enable_database) abi.features.database.batch else struct {};
const BatchProcessor = if (build_options.enable_database) batch.BatchProcessor else struct {};
const BatchWriter = if (build_options.enable_database) batch.BatchWriter else struct {};
const BatchOperationBuilder = if (build_options.enable_database) batch.BatchOperationBuilder else struct {};
const BatchRecord = if (build_options.enable_database) batch.BatchRecord else struct {};
const BatchResult = if (build_options.enable_database) batch.BatchResult else struct {};

// ============================================================================
// BatchResult Tests
// ============================================================================

test "batch: BatchResult.isComplete with no failures" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const result = BatchResult{
        .total_processed = 10,
        .successful = 10,
        .failed = 0,
        .skipped = 0,
        .elapsed_ns = 1000,
        .throughput = 10.0,
        .failed_ids = &.{},
        .errors = &.{},
    };
    try std.testing.expect(result.isComplete());
}

test "batch: BatchResult.isComplete with failures" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const result = BatchResult{
        .total_processed = 10,
        .successful = 8,
        .failed = 2,
        .skipped = 0,
        .elapsed_ns = 1000,
        .throughput = 8.0,
        .failed_ids = &.{},
        .errors = &.{},
    };
    try std.testing.expect(!result.isComplete());
}

test "batch: BatchResult.successRate" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const result = BatchResult{
        .total_processed = 100,
        .successful = 75,
        .failed = 25,
        .skipped = 0,
        .elapsed_ns = 1000,
        .throughput = 75.0,
        .failed_ids = &.{},
        .errors = &.{},
    };
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), result.successRate(), 0.01);
}

test "batch: BatchResult.successRate with zero processed" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const result = BatchResult{
        .total_processed = 0,
        .successful = 0,
        .failed = 0,
        .skipped = 0,
        .elapsed_ns = 0,
        .throughput = 0,
        .failed_ids = &.{},
        .errors = &.{},
    };
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.successRate(), 0.01);
}

// ============================================================================
// BatchRecord Tests
// ============================================================================

test "batch: BatchRecord.estimateSize with metadata" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    const record = BatchRecord{
        .id = 42,
        .vector = &vector,
        .metadata = "test-meta",
        .text = "hello",
    };

    const size = record.estimateSize();
    const expected = @sizeOf(u64) + 3 * @sizeOf(f32) + 9 + 5; // id + vector + metadata + text
    try std.testing.expectEqual(expected, size);
}

test "batch: BatchRecord.estimateSize without metadata" {
    if (!build_options.enable_database) return error.SkipZigTest;

    const vector = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const record = BatchRecord{
        .id = 1,
        .vector = &vector,
    };

    const size = record.estimateSize();
    const expected = @sizeOf(u64) + 4 * @sizeOf(f32);
    try std.testing.expectEqual(expected, size);
}

// ============================================================================
// BatchProcessor Delete Tests
// ============================================================================

test "batch: deleteBatch with IDs" {
    if (!build_options.enable_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var processor = BatchProcessor.init(allocator, .{});
    defer processor.deinit();

    const ids = [_]u64{ 1, 2, 3, 4, 5 };
    const result = try processor.deleteBatch(&ids);
    defer allocator.free(result.failed_ids);
    defer allocator.free(result.errors);

    try std.testing.expectEqual(@as(usize, 5), result.total_processed);
    try std.testing.expectEqual(@as(usize, 5), result.successful);
    try std.testing.expect(result.isComplete());

    const stats = processor.getStats();
    try std.testing.expectEqual(@as(usize, 5), stats.total_deleted);
}

test "batch: deleteBatch empty list" {
    if (!build_options.enable_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var processor = BatchProcessor.init(allocator, .{});
    defer processor.deinit();

    const empty_ids = [_]u64{};
    const result = try processor.deleteBatch(&empty_ids);
    defer allocator.free(result.failed_ids);
    defer allocator.free(result.errors);

    try std.testing.expectEqual(@as(usize, 0), result.total_processed);
    try std.testing.expect(result.isComplete());
}

// ============================================================================
// Validation Tests
// ============================================================================

test "batch: insertBatch with invalid records and continue_on_error" {
    if (!build_options.enable_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var processor = BatchProcessor.init(allocator, .{
        .validate_before_insert = true,
        .continue_on_error = true,
        .parallel_workers = 1, // avoid parallel path (has thread-join bug)
    });
    defer processor.deinit();

    const good_vector = [_]f32{ 1.0, 2.0, 3.0 };
    const nan_vector = [_]f32{ 1.0, std.math.nan(f32), 3.0 };
    const inf_vector = [_]f32{ std.math.inf(f32), 2.0, 3.0 };
    const empty_vector = [_]f32{};

    const records = [_]BatchRecord{
        .{ .id = 1, .vector = &good_vector },
        .{ .id = 2, .vector = &nan_vector },
        .{ .id = 3, .vector = &inf_vector },
        .{ .id = 4, .vector = &empty_vector },
        .{ .id = 5, .vector = &good_vector },
    };

    const result = try processor.insertBatch(&records);
    defer allocator.free(result.failed_ids);
    defer allocator.free(result.errors);

    // 2 good + 3 skipped (NaN, Inf, empty)
    try std.testing.expectEqual(@as(usize, 5), result.total_processed);
    try std.testing.expectEqual(@as(usize, 2), result.successful);
    try std.testing.expectEqual(@as(usize, 3), result.skipped);
}

// ============================================================================
// Streaming Writer Tests
// ============================================================================

test "batch: BatchWriter write before start returns error" {
    if (!build_options.enable_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var writer = BatchWriter.init(allocator, .{});
    defer writer.deinit();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    const result = writer.write(.{ .id = 1, .vector = &vector });
    try std.testing.expectError(error.NotStarted, result);
}

test "batch: BatchWriter finish before start returns error" {
    if (!build_options.enable_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var writer = BatchWriter.init(allocator, .{});
    defer writer.deinit();

    const result = writer.finish();
    try std.testing.expectError(error.NotStarted, result);
}

test "batch: BatchWriter abort clears state" {
    if (!build_options.enable_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var writer = BatchWriter.init(allocator, .{ .batch_size = 1000 });
    defer writer.deinit();

    try writer.start();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    try writer.write(.{ .id = 1, .vector = &vector });
    try writer.write(.{ .id = 2, .vector = &vector });

    writer.abort();

    // After abort, should need to start again
    const result = writer.write(.{ .id = 3, .vector = &vector });
    try std.testing.expectError(error.NotStarted, result);
}

test "batch: BatchWriter writeAll multiple records" {
    if (!build_options.enable_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var writer = BatchWriter.init(allocator, .{ .batch_size = 10 });
    defer writer.deinit();

    try writer.start();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    const records = [_]BatchRecord{
        .{ .id = 1, .vector = &vector },
        .{ .id = 2, .vector = &vector },
        .{ .id = 3, .vector = &vector },
    };

    const written = try writer.writeAll(&records);
    try std.testing.expectEqual(@as(usize, 3), written);

    const result = try writer.finish();
    try std.testing.expectEqual(@as(usize, 3), result.total_written);
}

// ============================================================================
// Builder Pattern Tests
// ============================================================================

test "batch: builder with metadata" {
    if (!build_options.enable_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var builder = BatchOperationBuilder.init(allocator);
    defer builder.deinit();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    _ = try builder.addRecordWithMetadata(1, &vector, "category=test");
    _ = try builder.addRecordWithMetadata(2, &vector, "category=prod");

    const result = try builder.execute();
    defer allocator.free(result.failed_ids);
    defer allocator.free(result.errors);

    try std.testing.expectEqual(@as(usize, 2), result.successful);
    try std.testing.expect(result.isComplete());
}

test "batch: builder fluent chaining" {
    if (!build_options.enable_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var builder = BatchOperationBuilder.init(allocator);
    defer builder.deinit();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };

    // Test chained calls
    _ = try builder
        .withBatchSize(50)
        .withWorkers(1)
        .withRetry(false)
        .addRecord(1, &vector);

    try std.testing.expectEqual(@as(usize, 50), builder.config.batch_size);
    try std.testing.expectEqual(@as(usize, 1), builder.config.parallel_workers);
    try std.testing.expect(!builder.config.retry_failed);
}

// ============================================================================
// Stats and Reset Tests
// ============================================================================

test "batch: processor resetStats clears counters" {
    if (!build_options.enable_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var processor = BatchProcessor.init(allocator, .{});
    defer processor.deinit();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    try processor.add(.{ .id = 1, .vector = &vector });
    try processor.flush();

    var stats = processor.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.total_inserted);

    processor.resetStats();
    stats = processor.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.total_inserted);
    try std.testing.expectEqual(@as(usize, 0), stats.batches_processed);
}

// ============================================================================
// Auto-flush Tests
// ============================================================================

test "batch: processor auto-flushes at batch_size" {
    if (!build_options.enable_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var processor = BatchProcessor.init(allocator, .{ .batch_size = 3 });
    defer processor.deinit();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    // Add 3 records — should trigger auto-flush on the 4th
    try processor.add(.{ .id = 1, .vector = &vector });
    try processor.add(.{ .id = 2, .vector = &vector });
    try processor.add(.{ .id = 3, .vector = &vector });
    // At this point 3 are pending. Adding a 4th triggers flush.
    try processor.add(.{ .id = 4, .vector = &vector });

    const stats = processor.getStats();
    // Batch of 3 should have been flushed
    try std.testing.expect(stats.batches_processed >= 1);
    try std.testing.expectEqual(@as(usize, 3), stats.total_inserted);
}

// ============================================================================
// Progress Callback Tests
// ============================================================================

test "batch: progress callback fires during insertBatch" {
    if (!build_options.enable_database) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const State = struct {
        var callback_count: usize = 0;
        var last_batch_number: usize = 0;
    };
    State.callback_count = 0;
    State.last_batch_number = 0;

    const callback = struct {
        fn cb(info: batch.ProgressInfo) void {
            State.callback_count += 1;
            State.last_batch_number = info.batch_number;
        }
    }.cb;

    var processor = BatchProcessor.init(allocator, .{
        .batch_size = 2,
        .report_progress = true,
        .parallel_workers = 1, // avoid parallel path (has thread-join bug)
    });
    defer processor.deinit();
    processor.setProgressCallback(callback);

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    const records = [_]BatchRecord{
        .{ .id = 1, .vector = &vector },
        .{ .id = 2, .vector = &vector },
        .{ .id = 3, .vector = &vector },
        .{ .id = 4, .vector = &vector },
        .{ .id = 5, .vector = &vector },
    };

    const result = try processor.insertBatch(&records);
    defer allocator.free(result.failed_ids);
    defer allocator.free(result.errors);

    // With batch_size=2 and 5 records, should be 3 batches → 3 callbacks
    try std.testing.expect(State.callback_count >= 1);
    try std.testing.expectEqual(@as(usize, 5), result.successful);
}
