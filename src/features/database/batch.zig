//! Batch operations for vector database.
//!
//! Provides efficient bulk insert, update, and delete operations
//! with configurable batching strategies and progress reporting.

const std = @import("std");

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
};

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
    mutex: std.Thread.Mutex,

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
        var timer = std.time.Timer.start() catch {
            return error.TimerFailed;
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

            // Process batch items
            for (batch) |record| {
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

    /// Insert batch of records in parallel using worker threads.
    fn insertBatchParallel(self: *BatchProcessor, records: []const BatchRecord) !BatchResult {
        var timer = std.time.Timer.start() catch {
            return error.TimerFailed;
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
        var timer = std.time.Timer.start() catch {
            return error.TimerFailed;
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
    timer: ?std.time.Timer,

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
        self.timer = std.time.Timer.start() catch null;
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

/// Import/export formats.
pub const ImportFormat = enum {
    json,
    csv,
    parquet,
    npy,
    binary,
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
        var records = std.ArrayListUnmanaged(BatchRecord){};
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
        var records = std.ArrayListUnmanaged(BatchRecord){};
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
            var vector_list = std.ArrayListUnmanaged(f32){};
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
};

test "batch processor basic" {
    const allocator = std.testing.allocator;
    var processor = BatchProcessor.init(allocator, .{ .batch_size = 10 });
    defer processor.deinit();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    try processor.add(.{ .id = 1, .vector = &vector });
    try processor.add(.{ .id = 2, .vector = &vector });

    try processor.flush();

    const stats = processor.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.total_inserted);
}

test "batch insert" {
    const allocator = std.testing.allocator;
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

    try std.testing.expectEqual(@as(usize, 3), result.successful);
    try std.testing.expect(result.isComplete());
}

test "batch writer" {
    const allocator = std.testing.allocator;
    var writer = BatchWriter.init(allocator, .{ .batch_size = 2 });
    defer writer.deinit();

    try writer.start();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    try writer.write(.{ .id = 1, .vector = &vector });
    try writer.write(.{ .id = 2, .vector = &vector });
    try writer.write(.{ .id = 3, .vector = &vector });

    const result = try writer.finish();

    try std.testing.expectEqual(@as(usize, 3), result.total_written);
    try std.testing.expect(result.batches_processed >= 1);
}

test "batch operation builder" {
    const allocator = std.testing.allocator;
    var builder = BatchOperationBuilder.init(allocator);
    defer builder.deinit();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    _ = try builder.withBatchSize(100).addRecord(1, &vector);
    _ = try builder.addRecord(2, &vector);

    const result = try builder.execute();
    defer allocator.free(result.failed_ids);
    defer allocator.free(result.errors);

    try std.testing.expectEqual(@as(usize, 2), result.successful);
}

test "record validation" {
    const allocator = std.testing.allocator;
    var processor = BatchProcessor.init(allocator, .{ .validate_before_insert = true });
    defer processor.deinit();

    // Empty vector should be invalid
    const empty_vector = [_]f32{};
    const invalid_record = BatchRecord{ .id = 1, .vector = &empty_vector };
    try std.testing.expect(!processor.validateRecord(invalid_record));

    // NaN should be invalid
    const nan_vector = [_]f32{ 1.0, std.math.nan(f32), 3.0 };
    const nan_record = BatchRecord{ .id = 2, .vector = &nan_vector };
    try std.testing.expect(!processor.validateRecord(nan_record));

    // Valid vector
    const valid_vector = [_]f32{ 1.0, 2.0, 3.0 };
    const valid_record = BatchRecord{ .id = 3, .vector = &valid_vector };
    try std.testing.expect(processor.validateRecord(valid_record));
}
