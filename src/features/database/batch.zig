//! Batch operations for vector database.
//!
//! Provides efficient bulk insert, update, and delete operations
//! with configurable batching strategies and progress reporting.

const std = @import("std");
const time = @import("../../services/shared/time.zig");
const sync = @import("../../services/shared/sync.zig");

/// Batch operation configuration.
pub const BatchConfig = struct {
    /// Maximum records per batch.
    batch_size: usize = 1000,
    /// Maximum memory per batch (bytes).
    max_batch_memory: usize = 64 * 1024 * 1024, // 64MB
    /// Parallel workers for batch processing.
    parallel_workers: usize = 4,
    /// Flush interval in milliseconds.
    flush_interval_ms: u64 = 1000,
    /// Retry failed items.
    retry_failed: bool = true,
    /// Maximum retries per item.
    max_retries: u32 = 3,
    /// Enable progress reporting.
    report_progress: bool = true,
    /// Progress report interval (items).
    progress_interval: usize = 100,
    /// Validate data before insert.
    validate_before_insert: bool = true,
    /// Continue on error.
    continue_on_error: bool = true,
    /// Prefetch distance for batch operations (0 = disabled)
    prefetch_distance: usize = 4,
};

// ============================================================================
// Prefetching Utilities for Batch Operations
// ============================================================================

/// Prefetch vector data for upcoming batch operations.
/// This helps reduce cache misses when processing large batches sequentially.
pub fn prefetchBatchVectors(records: []const BatchRecord, current_idx: usize, distance: usize) void {
    if (distance == 0) return;

    const prefetch_idx = current_idx + distance;
    if (prefetch_idx < records.len) {
        const record = records[prefetch_idx];
        // Prefetch the vector data
        const ptr: [*]const f32 = @ptrCast(record.vector.ptr);
        @prefetch(ptr, .{ .rw = .read, .locality = 3, .cache = .data });
    }
}

/// Batch record for insert operations.
pub const BatchRecord = struct {
    id: u64,
    vector: []const f32,
    metadata: ?[]const u8 = null,
    text: ?[]const u8 = null,

    pub fn estimateSize(self: BatchRecord) usize {
        var size: usize = @sizeOf(u64) + self.vector.len * @sizeOf(f32);
        if (self.metadata) |m| size += m.len;
        if (self.text) |t| size += t.len;
        return size;
    }
};

/// Batch operation result.
pub const BatchResult = struct {
    /// Total items processed.
    total_processed: usize,
    /// Successfully processed items.
    successful: usize,
    /// Failed items.
    failed: usize,
    /// Skipped items (duplicates, invalid).
    skipped: usize,
    /// Total elapsed time (nanoseconds).
    elapsed_ns: u64,
    /// Items per second throughput.
    throughput: f64,
    /// Failed item IDs.
    failed_ids: []const u64,
    /// Error messages for failed items.
    errors: []const BatchError,

    pub fn isComplete(self: BatchResult) bool {
        return self.failed == 0 and self.skipped == 0;
    }

    pub fn successRate(self: BatchResult) f64 {
        if (self.total_processed == 0) return 1.0;
        return @as(f64, @floatFromInt(self.successful)) /
            @as(f64, @floatFromInt(self.total_processed));
    }
};

/// Batch error information.
pub const BatchError = struct {
    id: u64,
    error_code: ErrorCode,
    message: []const u8,
    retry_count: u32,
};

/// Error codes for batch operations.
pub const ErrorCode = enum {
    invalid_vector_dimension,
    duplicate_id,
    storage_error,
    validation_failed,
    timeout,
    memory_exceeded,
    unknown,
};

/// Progress callback type.
pub const ProgressCallback = *const fn (ProgressInfo) void;

/// Progress information.
pub const ProgressInfo = struct {
    /// Current batch number.
    batch_number: usize,
    /// Total batches.
    total_batches: usize,
    /// Items processed so far.
    items_processed: usize,
    /// Total items.
    total_items: usize,
    /// Current throughput (items/sec).
    current_throughput: f64,
    /// Estimated time remaining (nanoseconds).
    estimated_remaining_ns: u64,
    /// Current batch success rate.
    batch_success_rate: f64,
};

/// Batch processor for bulk operations.
pub const BatchProcessor = struct {
    allocator: std.mem.Allocator,
    config: BatchConfig,
    pending_records: std.ArrayListUnmanaged(BatchRecord),
    pending_size: usize,
    progress_callback: ?ProgressCallback,
    stats: ProcessorStats,
    mutex: sync.Mutex,

    const ProcessorStats = struct {
        total_inserted: usize = 0,
        total_deleted: usize = 0,
        total_updated: usize = 0,
        total_errors: usize = 0,
        batches_processed: usize = 0,
    };

    pub fn init(allocator: std.mem.Allocator, config: BatchConfig) BatchProcessor {
        return .{
            .allocator = allocator,
            .config = config,
            .pending_records = .{},
            .pending_size = 0,
            .progress_callback = null,
            .stats = .{},
            .mutex = .{},
        };
    }

    pub fn deinit(self: *BatchProcessor) void {
        self.pending_records.deinit(self.allocator);
        self.* = undefined;
    }

    /// Set progress callback.
    pub fn setProgressCallback(self: *BatchProcessor, callback: ProgressCallback) void {
        self.progress_callback = callback;
    }

    /// Add record to pending batch.
    pub fn add(self: *BatchProcessor, record: BatchRecord) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const record_size = record.estimateSize();

        // Check if we need to flush
        if (self.shouldFlush(record_size)) {
            try self.flushInternal();
        }

        try self.pending_records.append(self.allocator, record);
        self.pending_size += record_size;
    }

    /// Add multiple records.
    pub fn addBatch(self: *BatchProcessor, records: []const BatchRecord) !usize {
        var added: usize = 0;
        for (records) |record| {
            try self.add(record);
            added += 1;
        }
        return added;
    }

    /// Check if flush is needed.
    fn shouldFlush(self: *const BatchProcessor, additional_size: usize) bool {
        if (self.pending_records.items.len >= self.config.batch_size) return true;
        if (self.pending_size + additional_size > self.config.max_batch_memory) return true;
        return false;
    }

    /// Flush pending records (internal, must hold lock).
    fn flushInternal(self: *BatchProcessor) !void {
        if (self.pending_records.items.len == 0) return;

        // Process batch (placeholder - actual storage integration would go here)
        self.stats.total_inserted += self.pending_records.items.len;
        self.stats.batches_processed += 1;

        // Report progress
        if (self.config.report_progress) {
            if (self.progress_callback) |callback| {
                callback(.{
                    .batch_number = self.stats.batches_processed,
                    .total_batches = 0, // Unknown in streaming mode
                    .items_processed = self.stats.total_inserted,
                    .total_items = 0, // Unknown in streaming mode
                    .current_throughput = 0,
                    .estimated_remaining_ns = 0,
                    .batch_success_rate = 1.0,
                });
            }
        }

        // Clear pending
        self.pending_records.clearRetainingCapacity();
        self.pending_size = 0;
    }

    /// Flush pending records.
    pub fn flush(self: *BatchProcessor) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.flushInternal();
    }

    /// Insert batch of records (sequential).
    pub fn insertBatch(self: *BatchProcessor, records: []const BatchRecord) !BatchResult {
        // Use parallel insert if workers configured
        if (self.config.parallel_workers > 1) {
            return try self.insertBatchParallel(records);
        }

        return try self.insertBatchSequential(records);
    }

    /// Insert batch of records sequentially.
    fn insertBatchSequential(self: *BatchProcessor, records: []const BatchRecord) !BatchResult {
        const timer = time.Timer.start() catch |err| {
            // Timer unavailable on this platform - process without timing
            std.log.debug("Timer unavailable: {t}, processing without timing", .{err});
            return self.insertBatchSequentialNoTiming(records);
        };

        var successful: usize = 0;
        var failed: usize = 0;
        _ = &failed; // Reserved for future error handling
        var skipped: usize = 0;
        var failed_ids = std.ArrayListUnmanaged(u64){};
        var errors = std.ArrayListUnmanaged(BatchError){};
        defer failed_ids.deinit(self.allocator);
        defer errors.deinit(self.allocator);

        // Process in batches
        var batch_start: usize = 0;
        var batch_num: usize = 0;
        const total_batches = (records.len + self.config.batch_size - 1) / self.config.batch_size;

        while (batch_start < records.len) {
            const batch_end = @min(batch_start + self.config.batch_size, records.len);
            const batch = records[batch_start..batch_end];

            // Process batch items with prefetching
            for (batch, 0..) |record, batch_idx| {
                // Prefetch upcoming records for better cache utilization
                if (self.config.prefetch_distance > 0) {
                    prefetchBatchVectors(batch, batch_idx, self.config.prefetch_distance);
                }

                // Validate
                if (self.config.validate_before_insert) {
                    if (!self.validateRecord(record)) {
                        if (self.config.continue_on_error) {
                            skipped += 1;
                            continue;
                        } else {
                            return error.ValidationFailed;
                        }
                    }
                }

                // Insert (placeholder - actual storage would go here)
                successful += 1;
            }

            batch_num += 1;

            // Report progress
            if (self.config.report_progress and self.progress_callback != null) {
                const elapsed = timer.read();
                const throughput = if (elapsed > 0)
                    @as(f64, @floatFromInt(successful)) / (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0)
                else
                    0;

                const remaining_items = records.len - batch_end;
                const estimated_remaining: u64 = if (throughput > 0)
                    @intFromFloat(@as(f64, @floatFromInt(remaining_items)) / throughput * 1_000_000_000.0)
                else
                    0;

                self.progress_callback.?(.{
                    .batch_number = batch_num,
                    .total_batches = total_batches,
                    .items_processed = batch_end,
                    .total_items = records.len,
                    .current_throughput = throughput,
                    .estimated_remaining_ns = estimated_remaining,
                    .batch_success_rate = if (batch.len > 0)
                        @as(f64, @floatFromInt(successful)) / @as(f64, @floatFromInt(batch_end))
                    else
                        1.0,
                });
            }

            batch_start = batch_end;
        }

        const elapsed = timer.read();
        const throughput = if (elapsed > 0)
            @as(f64, @floatFromInt(successful)) / (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0)
        else
            0;

        // Update stats
        self.mutex.lock();
        self.stats.total_inserted += successful;
        self.stats.total_errors += failed;
        self.mutex.unlock();

        return .{
            .total_processed = records.len,
            .successful = successful,
            .failed = failed,
            .skipped = skipped,
            .elapsed_ns = elapsed,
            .throughput = throughput,
            .failed_ids = try failed_ids.toOwnedSlice(self.allocator),
            .errors = try errors.toOwnedSlice(self.allocator),
        };
    }

    /// Insert batch of records sequentially without timing (fallback for platforms without timer).
    fn insertBatchSequentialNoTiming(self: *BatchProcessor, records: []const BatchRecord) !BatchResult {
        var successful: usize = 0;
        var failed: usize = 0;
        _ = &failed;
        var skipped: usize = 0;
        var failed_ids = std.ArrayListUnmanaged(u64){};
        var errors = std.ArrayListUnmanaged(BatchError){};
        defer failed_ids.deinit(self.allocator);
        defer errors.deinit(self.allocator);

        for (records) |record| {
            if (self.config.validate_before_insert) {
                if (!self.validateRecord(record)) {
                    if (self.config.continue_on_error) {
                        skipped += 1;
                        continue;
                    } else {
                        return error.ValidationFailed;
                    }
                }
            }
            successful += 1;
        }

        self.mutex.lock();
        self.stats.total_inserted += successful;
        self.stats.total_errors += failed;
        self.mutex.unlock();

        return .{
            .total_processed = records.len,
            .successful = successful,
            .failed = failed,
            .skipped = skipped,
            .elapsed_ns = 0, // No timing available
            .throughput = 0,
            .failed_ids = try failed_ids.toOwnedSlice(self.allocator),
            .errors = try errors.toOwnedSlice(self.allocator),
        };
    }

    /// Insert batch of records in parallel using worker threads.
    fn insertBatchParallel(self: *BatchProcessor, records: []const BatchRecord) !BatchResult {
        const timer = time.Timer.start() catch {
            // Timer unavailable - fallback to sequential without timing
            return self.insertBatchSequentialNoTiming(records);
        };

        const num_workers = @min(self.config.parallel_workers, records.len);
        if (num_workers <= 1) {
            return try self.insertBatchSequential(records);
        }

        // Shared state for workers
        const WorkerState = struct {
            records_slice: []const BatchRecord,
            successful: std.atomic.Value(usize),
            skipped: std.atomic.Value(usize),
            failed: std.atomic.Value(usize),
            processor: *BatchProcessor,
        };

        var worker_state = WorkerState{
            .records_slice = records,
            .successful = std.atomic.Value(usize).init(0),
            .skipped = std.atomic.Value(usize).init(0),
            .failed = std.atomic.Value(usize).init(0),
            .processor = self,
        };

        // Create worker threads
        var threads = try self.allocator.alloc(std.Thread, num_workers);
        defer self.allocator.free(threads);

        const chunk_size = (records.len + num_workers - 1) / num_workers;

        // Worker function
        const workerFn = struct {
            fn run(state: *WorkerState, start_idx: usize, end_idx: usize) void {
                const slice = state.records_slice[start_idx..end_idx];

                for (slice) |record| {
                    // Validate
                    if (state.processor.config.validate_before_insert) {
                        if (!state.processor.validateRecord(record)) {
                            if (state.processor.config.continue_on_error) {
                                _ = state.skipped.fetchAdd(1, .monotonic);
                                continue;
                            }
                        }
                    }

                    // Process record (placeholder - actual storage would go here)
                    _ = state.successful.fetchAdd(1, .monotonic);
                }
            }
        }.run;

        // Spawn workers
        for (0..num_workers) |i| {
            const start_idx = i * chunk_size;
            const end_idx = @min(start_idx + chunk_size, records.len);

            if (start_idx >= records.len) break;

            threads[i] = try std.Thread.spawn(.{}, workerFn, .{ &worker_state, start_idx, end_idx });
        }

        // Wait for all workers to complete
        for (threads[0..num_workers]) |thread| {
            thread.join();
        }

        const successful = worker_state.successful.load(.monotonic);
        const skipped = worker_state.skipped.load(.monotonic);
        const failed = worker_state.failed.load(.monotonic);

        const elapsed = timer.read();
        const throughput = if (elapsed > 0)
            @as(f64, @floatFromInt(successful)) / (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0)
        else
            0;

        // Update stats
        self.mutex.lock();
        self.stats.total_inserted += successful;
        self.stats.total_errors += failed;
        self.mutex.unlock();

        return .{
            .total_processed = records.len,
            .successful = successful,
            .failed = failed,
            .skipped = skipped,
            .elapsed_ns = elapsed,
            .throughput = throughput,
            .failed_ids = &.{}, // Parallel version doesn't track individual failures
            .errors = &.{},
        };
    }

    /// Delete batch of IDs.
    pub fn deleteBatch(self: *BatchProcessor, ids: []const u64) !BatchResult {
        const timer = time.Timer.start() catch {
            // Timer unavailable - return result without timing
            var successful: usize = 0;
            for (ids) |id| {
                _ = id;
                successful += 1;
            }
            self.mutex.lock();
            self.stats.total_deleted += successful;
            self.mutex.unlock();
            return .{
                .total_processed = ids.len,
                .successful = successful,
                .failed = 0,
                .skipped = 0,
                .elapsed_ns = 0,
                .throughput = 0,
                .failed_ids = &.{},
                .errors = &.{},
            };
        };

        var successful: usize = 0;
        var failed: usize = 0;
        _ = &failed; // Reserved for future error handling
        var failed_ids = std.ArrayListUnmanaged(u64){};
        var errors = std.ArrayListUnmanaged(BatchError){};
        defer failed_ids.deinit(self.allocator);
        defer errors.deinit(self.allocator);

        for (ids) |id| {
            // Delete (placeholder - actual storage would go here)
            _ = id;
            successful += 1;
        }

        const elapsed = timer.read();
        const throughput = if (elapsed > 0)
            @as(f64, @floatFromInt(successful)) / (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0)
        else
            0;

        self.mutex.lock();
        self.stats.total_deleted += successful;
        self.stats.total_errors += failed;
        self.mutex.unlock();

        return .{
            .total_processed = ids.len,
            .successful = successful,
            .failed = failed,
            .skipped = 0,
            .elapsed_ns = elapsed,
            .throughput = throughput,
            .failed_ids = try failed_ids.toOwnedSlice(self.allocator),
            .errors = try errors.toOwnedSlice(self.allocator),
        };
    }

    /// Validate a record.
    fn validateRecord(self: *const BatchProcessor, record: BatchRecord) bool {
        _ = self;
        // Check vector is not empty
        if (record.vector.len == 0) return false;

        // Check for NaN/Inf values
        for (record.vector) |v| {
            if (std.math.isNan(v) or std.math.isInf(v)) return false;
        }

        return true;
    }

    /// Get processing statistics.
    pub fn getStats(self: *const BatchProcessor) ProcessorStats {
        return self.stats;
    }

    /// Reset statistics.
    pub fn resetStats(self: *BatchProcessor) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.stats = .{};
    }
};

/// Streaming batch writer for large datasets.
pub const BatchWriter = struct {
    allocator: std.mem.Allocator,
    processor: BatchProcessor,
    total_written: usize,
    started: bool,
    timer: ?time.Timer,

    pub fn init(allocator: std.mem.Allocator, config: BatchConfig) BatchWriter {
        return .{
            .allocator = allocator,
            .processor = BatchProcessor.init(allocator, config),
            .total_written = 0,
            .started = false,
            .timer = null,
        };
    }

    pub fn deinit(self: *BatchWriter) void {
        self.processor.deinit();
        self.* = undefined;
    }

    /// Start writing session.
    pub fn start(self: *BatchWriter) !void {
        self.timer = time.Timer.start() catch null;
        self.started = true;
        self.total_written = 0;
    }

    /// Write a single record.
    pub fn write(self: *BatchWriter, record: BatchRecord) !void {
        if (!self.started) return error.NotStarted;
        try self.processor.add(record);
        self.total_written += 1;
    }

    /// Write multiple records.
    pub fn writeAll(self: *BatchWriter, records: []const BatchRecord) !usize {
        if (!self.started) return error.NotStarted;
        const written = try self.processor.addBatch(records);
        self.total_written += written;
        return written;
    }

    /// Finish writing and flush remaining.
    pub fn finish(self: *BatchWriter) !WriterResult {
        if (!self.started) return error.NotStarted;

        try self.processor.flush();

        const elapsed = if (self.timer) |*t| t.read() else 0;
        const throughput = if (elapsed > 0)
            @as(f64, @floatFromInt(self.total_written)) / (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0)
        else
            0;

        self.started = false;

        return .{
            .total_written = self.total_written,
            .elapsed_ns = elapsed,
            .throughput = throughput,
            .batches_processed = self.processor.stats.batches_processed,
        };
    }

    /// Abort writing session.
    pub fn abort(self: *BatchWriter) void {
        self.processor.pending_records.clearRetainingCapacity();
        self.processor.pending_size = 0;
        self.started = false;
    }
};

/// Writer session result.
pub const WriterResult = struct {
    total_written: usize,
    elapsed_ns: u64,
    throughput: f64,
    batches_processed: usize,
};

/// Batch operation builder for fluent API.
pub const BatchOperationBuilder = struct {
    allocator: std.mem.Allocator,
    records: std.ArrayListUnmanaged(BatchRecord),
    config: BatchConfig,

    pub fn init(allocator: std.mem.Allocator) BatchOperationBuilder {
        return .{
            .allocator = allocator,
            .records = .{},
            .config = .{},
        };
    }

    pub fn deinit(self: *BatchOperationBuilder) void {
        self.records.deinit(self.allocator);
        self.* = undefined;
    }

    /// Set batch size.
    pub fn withBatchSize(self: *BatchOperationBuilder, size: usize) *BatchOperationBuilder {
        self.config.batch_size = size;
        return self;
    }

    /// Set parallel workers.
    pub fn withWorkers(self: *BatchOperationBuilder, workers: usize) *BatchOperationBuilder {
        self.config.parallel_workers = workers;
        return self;
    }

    /// Enable/disable retries.
    pub fn withRetry(self: *BatchOperationBuilder, enabled: bool) *BatchOperationBuilder {
        self.config.retry_failed = enabled;
        return self;
    }

    /// Add a record.
    pub fn addRecord(self: *BatchOperationBuilder, id: u64, vector: []const f32) !*BatchOperationBuilder {
        try self.records.append(self.allocator, .{
            .id = id,
            .vector = vector,
        });
        return self;
    }

    /// Add record with metadata.
    pub fn addRecordWithMetadata(
        self: *BatchOperationBuilder,
        id: u64,
        vector: []const f32,
        metadata: []const u8,
    ) !*BatchOperationBuilder {
        try self.records.append(self.allocator, .{
            .id = id,
            .vector = vector,
            .metadata = metadata,
        });
        return self;
    }

    /// Execute the batch insert.
    pub fn execute(self: *BatchOperationBuilder) !BatchResult {
        var processor = BatchProcessor.init(self.allocator, self.config);
        defer processor.deinit();
        return processor.insertBatch(self.records.items);
    }
};

// Re-export importer types for backward compatibility
const batch_importer = @import("batch_importer.zig");
pub const ImportFormat = batch_importer.ImportFormat;
pub const BatchImporter = batch_importer.BatchImporter;

test {
    _ = @import("batch_importer.zig");
    _ = @import("batch_test.zig");
}
