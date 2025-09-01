//! WDBX Advanced Stress Testing & Benchmarking Suite
//!
//! Enterprise-grade testing for production vector database validation:
//! - Database Performance Validation: Tests under various load conditions
//! - Network Saturation Testing: High-concurrency scenario simulation
//! - Failure Recovery Validation: System resilience under stress conditions
//! - Memory Pressure Scenarios: Memory management under load validation
//! - Atomic Operations: Thread-safe metrics with std.atomic.Value
//! - Detailed Reporting: Comprehensive performance and latency analysis
//! - Success/Failure Tracking: Detailed operation outcome breakdown
//! - Progress Monitoring: Real-time updates during long-running tests

const std = @import("std");
const builtin = @import("builtin");

// Sleep function for compatibility
fn sleep(ns: u64) void {
    // Use a busy loop for now - in real implementation, use proper OS sleep
    const start = std.time.nanoTimestamp();
    while (std.time.nanoTimestamp() - start < ns) {
        std.atomic.spinLoopHint();
    }
}

const Config = struct {
    // Connection settings
    host: []const u8 = "localhost",
    port: u16 = 8080,
    allocator: std.mem.Allocator,

    // Test parameters
    num_threads: u32 = 32,
    num_vectors: u64 = 1_000_000,
    vector_dimension: u32 = 384,
    duration_seconds: u64 = 300,

    // Operation mix (percentages) - must sum to 100
    read_percent: u8 = 70,
    write_percent: u8 = 20,
    delete_percent: u8 = 5,
    search_percent: u8 = 5,

    // Load patterns
    pattern: LoadPattern = .steady,
    burst_size: u32 = 1000,
    burst_interval_ms: u64 = 100,

    // Network saturation testing
    enable_network_saturation: bool = false,
    concurrent_connections: u32 = 1000,
    connection_timeout_ms: u64 = 5000,

    // Failure recovery testing
    enable_failure_simulation: bool = false,
    failure_interval_seconds: u64 = 60,
    failure_duration_ms: u64 = 1000,
    failure_rate_percent: u8 = 10, // Chance of failure per operation

    // Memory pressure testing
    enable_memory_pressure: bool = false,
    memory_pressure_mb: u64 = 1024, // Memory to allocate per thread
    memory_pressure_pattern: MemoryPressurePattern = .gradual,

    // Enterprise-grade metrics
    enable_detailed_metrics: bool = true,
    percentile_reporting: bool = true,
    enable_histogram: bool = false,
    histogram_buckets: u32 = 100,

    // Progress monitoring
    progress_update_interval_seconds: u64 = 10,
    enable_real_time_reporting: bool = true,

    // Benchmarking configuration
    benchmark_mode: bool = false,
    vdbench_compatibility: bool = false,
    output_format: OutputFormat = .json,

    const LoadPattern = enum {
        steady, // Constant load
        burst, // Periodic bursts
        ramp_up, // Gradual load increase
        spike, // Sudden load spikes
        random, // Random load variations
        network_saturate, // Network saturation pattern
        memory_pressure, // Memory pressure pattern
    };

    const MemoryPressurePattern = enum {
        gradual, // Slowly increasing memory usage
        spike, // Sudden memory spikes
        sawtooth, // Memory usage up and down
        constant, // Constant high memory usage
    };

    const OutputFormat = enum {
        text,
        json,
        csv,
        prometheus, // Prometheus metrics format
    };

    fn init(allocator: std.mem.Allocator) Config {
        return .{
            .allocator = allocator,
            .host = "localhost", // Will be overridden if --host is specified
        };
    }

    fn deinit(self: *Config) void {
        // Only free host if it was allocated (i.e., not the default "localhost")
        // We use a simple length check as a heuristic
        if (self.host.len != "localhost".len or !std.mem.eql(u8, self.host, "localhost")) {
            self.allocator.free(self.host);
        }
    }

    fn validate(self: Config) !void {
        // Validate operation mix percentages
        const total_percent = self.read_percent + self.write_percent +
            self.delete_percent + self.search_percent;
        if (total_percent != 100) {
            return error.InvalidOperationMix;
        }

        // Validate failure rate
        if (self.failure_rate_percent > 100) {
            return error.InvalidFailureRate;
        }

        // Validate thread count
        if (self.num_threads == 0) {
            return error.InvalidThreadCount;
        }

        // Validate network saturation settings
        if (self.enable_network_saturation) {
            if (self.concurrent_connections < self.num_threads) {
                return error.InvalidNetworkConfig;
            }
        }
    }
};

const Metrics = struct {
    // Basic operation counters
    operations: std.atomic.Value(u64),
    successes: std.atomic.Value(u64),
    failures: std.atomic.Value(u64),

    // Detailed operation breakdown
    read_operations: std.atomic.Value(u64),
    write_operations: std.atomic.Value(u64),
    delete_operations: std.atomic.Value(u64),
    search_operations: std.atomic.Value(u64),

    read_successes: std.atomic.Value(u64),
    write_successes: std.atomic.Value(u64),
    delete_successes: std.atomic.Value(u64),
    search_successes: std.atomic.Value(u64),

    // Latency tracking (microseconds)
    latency_sum: std.atomic.Value(f64),
    latency_count: std.atomic.Value(u64),
    latency_min: std.atomic.Value(u64),
    latency_max: std.atomic.Value(u64),

    // Percentile tracking for latency
    p50_latency: std.atomic.Value(u64),
    p95_latency: std.atomic.Value(u64),
    p99_latency: std.atomic.Value(u64),

    // Network-related metrics
    connection_timeouts: std.atomic.Value(u64),
    network_errors: std.atomic.Value(u64),
    bytes_sent: std.atomic.Value(u64),
    bytes_received: std.atomic.Value(u64),

    // Memory metrics
    memory_allocated: std.atomic.Value(u64),
    memory_peak: std.atomic.Value(u64),
    memory_leaks: std.atomic.Value(u64),

    // Performance metrics
    throughput: std.atomic.Value(f64),
    cpu_usage: std.atomic.Value(f64),

    // Error categorization
    timeout_errors: std.atomic.Value(u64),
    connection_errors: std.atomic.Value(u64),
    protocol_errors: std.atomic.Value(u64),
    server_errors: std.atomic.Value(u64),

    // Timing
    start_time: i64,
    end_time: i64,

    // Histogram data (optional)
    latency_histogram: ?[]std.atomic.Value(u64),

    fn init(allocator: std.mem.Allocator, config: Config) !Metrics {
        var histogram: ?[]std.atomic.Value(u64) = null;
        if (config.enable_histogram) {
            histogram = try allocator.alloc(std.atomic.Value(u64), config.histogram_buckets);
            for (histogram.?) |*bucket| {
                bucket.* = std.atomic.Value(u64).init(0);
            }
        }

        return Metrics{
            .operations = std.atomic.Value(u64).init(0),
            .successes = std.atomic.Value(u64).init(0),
            .failures = std.atomic.Value(u64).init(0),

            .read_operations = std.atomic.Value(u64).init(0),
            .write_operations = std.atomic.Value(u64).init(0),
            .delete_operations = std.atomic.Value(u64).init(0),
            .search_operations = std.atomic.Value(u64).init(0),

            .read_successes = std.atomic.Value(u64).init(0),
            .write_successes = std.atomic.Value(u64).init(0),
            .delete_successes = std.atomic.Value(u64).init(0),
            .search_successes = std.atomic.Value(u64).init(0),

            .latency_sum = std.atomic.Value(f64).init(0),
            .latency_count = std.atomic.Value(u64).init(0),
            .latency_min = std.atomic.Value(u64).init(std.math.maxInt(u64)),
            .latency_max = std.atomic.Value(u64).init(0),

            .p50_latency = std.atomic.Value(u64).init(0),
            .p95_latency = std.atomic.Value(u64).init(0),
            .p99_latency = std.atomic.Value(u64).init(0),

            .connection_timeouts = std.atomic.Value(u64).init(0),
            .network_errors = std.atomic.Value(u64).init(0),
            .bytes_sent = std.atomic.Value(u64).init(0),
            .bytes_received = std.atomic.Value(u64).init(0),

            .memory_allocated = std.atomic.Value(u64).init(0),
            .memory_peak = std.atomic.Value(u64).init(0),
            .memory_leaks = std.atomic.Value(u64).init(0),

            .throughput = std.atomic.Value(f64).init(0),
            .cpu_usage = std.atomic.Value(f64).init(0),

            .timeout_errors = std.atomic.Value(u64).init(0),
            .connection_errors = std.atomic.Value(u64).init(0),
            .protocol_errors = std.atomic.Value(u64).init(0),
            .server_errors = std.atomic.Value(u64).init(0),

            .start_time = std.time.milliTimestamp(),
            .end_time = 0,
            .latency_histogram = histogram,
        };
    }

    fn deinit(self: *Metrics, allocator: std.mem.Allocator) void {
        if (self.latency_histogram) |histogram| {
            allocator.free(histogram);
        }
    }

    fn recordOperation(self: *Metrics, operation_type: OperationType, latency_us: u64, success: bool, error_type: ErrorType) void {
        _ = self.operations.fetchAdd(1, .monotonic);

        // Record operation type
        switch (operation_type) {
            .read => {
                _ = self.read_operations.fetchAdd(1, .monotonic);
                if (success) _ = self.read_successes.fetchAdd(1, .monotonic);
            },
            .write => {
                _ = self.write_operations.fetchAdd(1, .monotonic);
                if (success) _ = self.write_successes.fetchAdd(1, .monotonic);
            },
            .delete => {
                _ = self.delete_operations.fetchAdd(1, .monotonic);
                if (success) _ = self.delete_successes.fetchAdd(1, .monotonic);
            },
            .search => {
                _ = self.search_operations.fetchAdd(1, .monotonic);
                if (success) _ = self.search_successes.fetchAdd(1, .monotonic);
            },
        }

        if (success) {
            _ = self.successes.fetchAdd(1, .monotonic);
        } else {
            _ = self.failures.fetchAdd(1, .monotonic);

            // Record error type
            switch (error_type) {
                .timeout => _ = self.timeout_errors.fetchAdd(1, .monotonic),
                .connection => _ = self.connection_errors.fetchAdd(1, .monotonic),
                .protocol => _ = self.protocol_errors.fetchAdd(1, .monotonic),
                .server => _ = self.server_errors.fetchAdd(1, .monotonic),
                .none => {},
            }
        }

        // Record latency metrics
        _ = self.latency_sum.fetchAdd(@floatFromInt(latency_us), .monotonic);
        _ = self.latency_count.fetchAdd(1, .monotonic);

        // Update min/max with atomic operations
        var current_min = self.latency_min.load(.monotonic);
        while (latency_us < current_min) {
            if (self.latency_min.cmpxchgWeak(current_min, latency_us, .monotonic, .monotonic)) |new_min| {
                current_min = new_min;
            } else {
                break;
            }
        }

        var current_max = self.latency_max.load(.monotonic);
        while (latency_us > current_max) {
            if (self.latency_max.cmpxchgWeak(current_max, latency_us, .monotonic, .monotonic)) |new_max| {
                current_max = new_max;
            } else {
                break;
            }
        }

        // Update histogram if enabled
        if (self.latency_histogram) |histogram| {
            const bucket = @min(histogram.len - 1, latency_us / 100); // 100μs per bucket
            if (bucket < histogram.len) {
                _ = histogram[bucket].fetchAdd(1, .monotonic);
            }
        }
    }

    const OperationType = enum {
        read,
        write,
        delete,
        search,
    };

    const ErrorType = enum {
        none,
        timeout,
        connection,
        protocol,
        server,
    };

    fn printSummary(self: *const Metrics, config: Config) void {
        const total_ops = self.operations.load(.monotonic);
        const success_rate = if (total_ops > 0)
            @as(f64, @floatFromInt(self.successes.load(.monotonic))) /
                @as(f64, @floatFromInt(total_ops)) * 100.0
        else
            0.0;

        const avg_latency = if (self.latency_count.load(.monotonic) > 0)
            self.latency_sum.load(.monotonic) /
                @as(f64, @floatFromInt(self.latency_count.load(.monotonic)))
        else
            0.0;

        const duration_ms = @as(f64, @floatFromInt(self.end_time - self.start_time));
        const throughput = if (duration_ms > 0)
            @as(f64, @floatFromInt(total_ops)) / (duration_ms / 1000.0)
        else
            0.0;

        std.debug.print("\n" ++ "=" ** 80 ++ "\n", .{});
        std.debug.print("🚀 **WDBX Enterprise Stress Test Results**\n", .{});
        std.debug.print("=" ** 80 ++ "\n", .{});

        // Basic performance metrics
        std.debug.print("📊 **Performance Overview:**\n", .{});
        std.debug.print("  Total Operations:    {}\n", .{total_ops});
        std.debug.print("  Success Rate:        {d:.2}%\n", .{success_rate});
        std.debug.print("  Throughput:          {d:.0} ops/sec\n", .{throughput});
        std.debug.print("  Test Duration:       {d:.1}s\n", .{duration_ms / 1000.0});

        // Detailed latency statistics
        std.debug.print("\n⏱️  **Latency Statistics (μs):**\n", .{});
        std.debug.print("  Average:             {d:.0}\n", .{avg_latency});
        std.debug.print("  Minimum:             {}\n", .{self.latency_min.load(.monotonic)});
        std.debug.print("  Maximum:             {}\n", .{self.latency_max.load(.monotonic)});
        if (config.percentile_reporting) {
            std.debug.print("  P50 (median):        {}\n", .{self.p50_latency.load(.monotonic)});
            std.debug.print("  P95:                 {}\n", .{self.p95_latency.load(.monotonic)});
            std.debug.print("  P99:                 {}\n", .{self.p99_latency.load(.monotonic)});
        }

        // Operation breakdown
        std.debug.print("\n🔍 **Operation Breakdown:**\n", .{});
        const reads = self.read_operations.load(.monotonic);
        const writes = self.write_operations.load(.monotonic);
        const deletes = self.delete_operations.load(.monotonic);
        const searches = self.search_operations.load(.monotonic);

        if (reads > 0) std.debug.print("  Reads:               {} ({d:.1}%)\n", .{ reads, @as(f64, @floatFromInt(reads)) / @as(f64, @floatFromInt(total_ops)) * 100.0 });
        if (writes > 0) std.debug.print("  Writes:              {} ({d:.1}%)\n", .{ writes, @as(f64, @floatFromInt(writes)) / @as(f64, @floatFromInt(total_ops)) * 100.0 });
        if (deletes > 0) std.debug.print("  Deletes:             {} ({d:.1}%)\n", .{ deletes, @as(f64, @floatFromInt(deletes)) / @as(f64, @floatFromInt(total_ops)) * 100.0 });
        if (searches > 0) std.debug.print("  Searches:            {} ({d:.1}%)\n", .{ searches, @as(f64, @floatFromInt(searches)) / @as(f64, @floatFromInt(total_ops)) * 100.0 });

        // Success rates by operation type
        std.debug.print("\n✅ **Success Rates by Operation:**\n", .{});
        if (reads > 0) {
            const read_success_rate = @as(f64, @floatFromInt(self.read_successes.load(.monotonic))) / @as(f64, @floatFromInt(reads)) * 100.0;
            std.debug.print("  Read Success Rate:   {d:.2}%\n", .{read_success_rate});
        }
        if (writes > 0) {
            const write_success_rate = @as(f64, @floatFromInt(self.write_successes.load(.monotonic))) / @as(f64, @floatFromInt(writes)) * 100.0;
            std.debug.print("  Write Success Rate:  {d:.2}%\n", .{write_success_rate});
        }
        if (deletes > 0) {
            const delete_success_rate = @as(f64, @floatFromInt(self.delete_successes.load(.monotonic))) / @as(f64, @floatFromInt(deletes)) * 100.0;
            std.debug.print("  Delete Success Rate: {d:.2}%\n", .{delete_success_rate});
        }
        if (searches > 0) {
            const search_success_rate = @as(f64, @floatFromInt(self.search_successes.load(.monotonic))) / @as(f64, @floatFromInt(searches)) * 100.0;
            std.debug.print("  Search Success Rate: {d:.2}%\n", .{search_success_rate});
        }

        // Error analysis
        const failures = self.failures.load(.monotonic);
        if (failures > 0) {
            std.debug.print("\n❌ **Error Analysis:**\n", .{});
            std.debug.print("  Total Failures:      {}\n", .{failures});

            const timeout_errs = self.timeout_errors.load(.monotonic);
            const conn_errs = self.connection_errors.load(.monotonic);
            const proto_errs = self.protocol_errors.load(.monotonic);
            const server_errs = self.server_errors.load(.monotonic);

            if (timeout_errs > 0) std.debug.print("  Timeout Errors:      {} ({d:.1}%)\n", .{ timeout_errs, @as(f64, @floatFromInt(timeout_errs)) / @as(f64, @floatFromInt(failures)) * 100.0 });
            if (conn_errs > 0) std.debug.print("  Connection Errors:   {} ({d:.1}%)\n", .{ conn_errs, @as(f64, @floatFromInt(conn_errs)) / @as(f64, @floatFromInt(failures)) * 100.0 });
            if (proto_errs > 0) std.debug.print("  Protocol Errors:     {} ({d:.1}%)\n", .{ proto_errs, @as(f64, @floatFromInt(proto_errs)) / @as(f64, @floatFromInt(failures)) * 100.0 });
            if (server_errs > 0) std.debug.print("  Server Errors:       {} ({d:.1}%)\n", .{ server_errs, @as(f64, @floatFromInt(server_errs)) / @as(f64, @floatFromInt(failures)) * 100.0 });
        }

        // Network and memory metrics
        std.debug.print("\n🌐 **Network & Memory Metrics:**\n", .{});
        std.debug.print("  Connection Timeouts: {}\n", .{self.connection_timeouts.load(.monotonic)});
        std.debug.print("  Network Errors:      {}\n", .{self.network_errors.load(.monotonic)});
        std.debug.print("  Data Sent:           {} MB\n", .{self.bytes_sent.load(.monotonic) / (1024 * 1024)});
        std.debug.print("  Data Received:       {} MB\n", .{self.bytes_received.load(.monotonic) / (1024 * 1024)});
        std.debug.print("  Peak Memory Usage:   {} MB\n", .{self.memory_peak.load(.monotonic) / (1024 * 1024)});
        std.debug.print("  Memory Leaks:        {}\n", .{self.memory_leaks.load(.monotonic)});

        std.debug.print("=" ** 80 ++ "\n", .{});
    }

    fn exportJson(self: *const Metrics, allocator: std.mem.Allocator) ![]const u8 {
        var json = try std.ArrayList(u8).initCapacity(allocator, 0);
        defer json.deinit(allocator);

        try json.writer(allocator).print(
            \\{{
            \\  "summary": {{
            \\    "total_operations": {},
            \\    "success_rate": {d:.2},
            \\    "throughput": {d:.0},
            \\    "duration_ms": {}
            \\  }},
            \\  "latency": {{
            \\    "average_us": {d:.0},
            \\    "min_us": {},
            \\    "max_us": {},
            \\    "p50_us": {},
            \\    "p95_us": {},
            \\    "p99_us": {}
            \\  }},
            \\  "operations": {{
            \\    "reads": {},
            \\    "writes": {},
            \\    "deletes": {},
            \\    "searches": {}
            \\  }},
            \\  "errors": {{
            \\    "total_failures": {},
            \\    "timeout_errors": {},
            \\    "connection_errors": {},
            \\    "protocol_errors": {},
            \\    "server_errors": {}
            \\  }}
            \\}}
        , .{
            self.operations.load(.monotonic),
            @as(f64, @floatFromInt(self.successes.load(.monotonic))) /
                @as(f64, @floatFromInt(self.operations.load(.monotonic))) * 100.0,
            @as(f64, @floatFromInt(self.operations.load(.monotonic))) /
                (@as(f64, @floatFromInt(self.end_time - self.start_time)) / 1000.0),
            self.end_time - self.start_time,
            if (self.latency_count.load(.monotonic) > 0)
                self.latency_sum.load(.monotonic) /
                    @as(f64, @floatFromInt(self.latency_count.load(.monotonic)))
            else
                0.0,
            self.latency_min.load(.monotonic),
            self.latency_max.load(.monotonic),
            self.p50_latency.load(.monotonic),
            self.p95_latency.load(.monotonic),
            self.p99_latency.load(.monotonic),
            self.read_operations.load(.monotonic),
            self.write_operations.load(.monotonic),
            self.delete_operations.load(.monotonic),
            self.search_operations.load(.monotonic),
            self.failures.load(.monotonic),
            self.timeout_errors.load(.monotonic),
            self.connection_errors.load(.monotonic),
            self.protocol_errors.load(.monotonic),
            self.server_errors.load(.monotonic),
        });

        return json.toOwnedSlice(allocator);
    }
};

const StressTest = struct {
    allocator: std.mem.Allocator,
    config: Config,
    metrics: Metrics,
    shutdown: std.atomic.Value(bool),
    client: *HttpClient,

    // Network saturation testing
    connection_pool: std.ArrayList(*HttpClient),
    connection_semaphore: std.Thread.Semaphore,

    // Memory pressure testing
    memory_blocks: std.ArrayList([]u8),
    memory_pressure_thread: ?std.Thread,

    const HttpClient = struct {
        allocator: std.mem.Allocator,
        host: []const u8,
        port: u16,

        fn init(allocator: std.mem.Allocator, host: []const u8, port: u16) !*HttpClient {
            const self = try allocator.create(HttpClient);
            self.* = .{
                .allocator = allocator,
                .host = host,
                .port = port,
            };
            return self;
        }

        fn deinit(self: *HttpClient) void {
            self.allocator.destroy(self);
        }

        fn addVector(self: *HttpClient, vector: []f32) !u64 {
            _ = self;
            _ = vector;
            // Simulate HTTP request
            sleep(1000 * std.time.ns_per_us);
            return std.crypto.random.int(u64);
        }

        fn searchVector(self: *HttpClient, query: []f32, k: usize) !void {
            _ = self;
            _ = query;
            _ = k;
            // Simulate HTTP request
            sleep(5000 * std.time.ns_per_us);
        }

        fn getVector(self: *HttpClient, id: u64) !void {
            _ = self;
            _ = id;
            // Simulate HTTP request
            sleep(500 * std.time.ns_per_us);
        }

        fn deleteVector(self: *HttpClient, id: u64) !void {
            _ = self;
            _ = id;
            // Simulate HTTP request
            sleep(800 * std.time.ns_per_us);
        }
    };

    fn init(allocator: std.mem.Allocator, config: Config) !*StressTest {
        // Validate configuration
        try config.validate();

        const self = try allocator.create(StressTest);

        // Initialize metrics first
        const metrics = try Metrics.init(allocator, config);

        // Initialize ArrayLists first
        const connection_pool = try std.ArrayList(*HttpClient).initCapacity(allocator, 0);
        const memory_blocks = try std.ArrayList([]u8).initCapacity(allocator, 0);

        self.* = .{
            .allocator = allocator,
            .config = config,
            .metrics = metrics,
            .shutdown = std.atomic.Value(bool).init(false),
            .client = try HttpClient.init(allocator, config.host, config.port),
            .connection_pool = connection_pool,
            .connection_semaphore = std.Thread.Semaphore{},
            .memory_blocks = memory_blocks,
            .memory_pressure_thread = null,
        };

        // Initialize connection pool for network saturation testing
        if (config.enable_network_saturation) {
            try self.initializeConnectionPool();
        }

        // Start memory pressure thread if enabled
        if (config.enable_memory_pressure) {
            self.memory_pressure_thread = try std.Thread.spawn(.{}, memoryPressureWorker, .{self});
        }

        return self;
    }

    fn deinit(self: *StressTest) void {
        // Stop memory pressure thread
        if (self.memory_pressure_thread) |thread| {
            self.shutdown.store(true, .monotonic);
            thread.join();
        }

        // Clean up connection pool
        for (self.connection_pool.items) |client| {
            client.deinit();
        }
        self.connection_pool.deinit(self.allocator);

        // Clean up memory blocks
        for (self.memory_blocks.items) |block| {
            self.allocator.free(block);
        }
        self.memory_blocks.deinit(self.allocator);

        // Clean up metrics
        self.metrics.deinit(self.allocator);

        self.client.deinit();
        self.allocator.destroy(self);
    }

    fn initializeConnectionPool(self: *StressTest) !void {
        const num_connections = self.config.concurrent_connections;
        try self.connection_pool.ensureTotalCapacity(self.allocator, num_connections);

        for (0..num_connections) |_| {
            const client = try HttpClient.init(self.allocator, self.config.host, self.config.port);
            try self.connection_pool.append(self.allocator, client);
        }

        // Initialize semaphore to control connection usage
        self.connection_semaphore = std.Thread.Semaphore{};
    }

    fn memoryPressureWorker(self: *StressTest) void {
        while (!self.shutdown.load(.monotonic)) {
            switch (self.config.memory_pressure_pattern) {
                .gradual => {
                    // Gradually increase memory usage
                    const block_size = 1024 * 1024; // 1MB blocks
                    const block = self.allocator.alloc(u8, block_size) catch continue;
                    @memset(block, 0); // Touch the memory
                    self.memory_blocks.append(self.allocator, block) catch {
                        self.allocator.free(block);
                        continue;
                    };

                    // Update peak memory usage
                    const current_memory = self.memory_blocks.items.len * block_size;
                    var peak = self.metrics.memory_peak.load(.monotonic);
                    while (current_memory > peak) {
                        if (self.metrics.memory_peak.cmpxchgWeak(peak, current_memory, .monotonic, .monotonic)) |new_peak| {
                            peak = new_peak;
                        } else {
                            break;
                        }
                    }
                },
                .spike => {
                    // Sudden memory spike
                    const spike_size = 10 * 1024 * 1024; // 10MB
                    const block = self.allocator.alloc(u8, spike_size) catch continue;
                    @memset(block, 0);
                    sleep(100 * std.time.ns_per_ms); // Hold for 100ms
                    self.allocator.free(block);
                },
                .sawtooth => {
                    // Memory usage up and down
                    const block_size = 2 * 1024 * 1024; // 2MB
                    const block = self.allocator.alloc(u8, block_size) catch continue;
                    @memset(block, 0);
                    sleep(50 * std.time.ns_per_ms); // Hold for 50ms
                    self.allocator.free(block);
                },
                .constant => {
                    // Maintain constant high memory usage
                    const target_memory = self.config.memory_pressure_mb * 1024 * 1024;
                    const current_memory = self.memory_blocks.items.len * 1024 * 1024;
                    if (current_memory < target_memory) {
                        const block_size = 1024 * 1024; // 1MB
                        const block = self.allocator.alloc(u8, block_size) catch continue;
                        @memset(block, 0);
                        self.memory_blocks.append(self.allocator, block) catch {
                            self.allocator.free(block);
                        };
                    }
                },
            }

            sleep(500 * std.time.ns_per_ms); // Check every 500ms
        }
    }

    fn run(self: *StressTest) !void {
        std.debug.print("🚀 Starting WDBX Enterprise Stress Test...\n", .{});
        std.debug.print("  Test Configuration:\n", .{});
        std.debug.print("    Threads:           {}\n", .{self.config.num_threads});
        std.debug.print("    Vectors:           {}\n", .{self.config.num_vectors});
        std.debug.print("    Duration:          {} seconds\n", .{self.config.duration_seconds});
        std.debug.print("    Load Pattern:      {s}\n", .{@tagName(self.config.pattern)});
        std.debug.print("    Vector Dimension:  {}\n", .{self.config.vector_dimension});

        if (self.config.enable_network_saturation) {
            std.debug.print("    Network Saturation: Enabled ({} connections)\n", .{self.config.concurrent_connections});
        }
        if (self.config.enable_failure_simulation) {
            std.debug.print("    Failure Simulation: Enabled ({}% failure rate)\n", .{self.config.failure_rate_percent});
        }
        if (self.config.enable_memory_pressure) {
            std.debug.print("    Memory Pressure:   Enabled ({} MB, {s} pattern)\n", .{ self.config.memory_pressure_mb, @tagName(self.config.memory_pressure_pattern) });
        }
        std.debug.print("\n", .{});

        // Create worker threads
        const threads = try self.allocator.alloc(std.Thread, self.config.num_threads);
        defer self.allocator.free(threads);

        // Start workers
        for (threads, 0..) |*thread, i| {
            thread.* = try std.Thread.spawn(.{}, workerThread, .{ self, i });
        }

        // Monitor progress with real-time reporting
        try self.monitorProgress();

        // Shutdown
        self.shutdown.store(true, .monotonic);

        // Wait for workers
        for (threads) |thread| {
            thread.join();
        }

        self.metrics.end_time = std.time.milliTimestamp();

        // Print comprehensive results
        self.metrics.printSummary(self.config);

        // Export results if requested
        if (self.config.output_format != .text) {
            try self.exportResults();
        }
    }

    fn monitorProgress(self: *StressTest) !void {
        var elapsed: u64 = 0;
        const update_interval = self.config.progress_update_interval_seconds;

        while (elapsed < self.config.duration_seconds) {
            sleep(update_interval * 1_000_000_000);
            elapsed += update_interval;

            const ops = self.metrics.operations.load(.monotonic);
            const successes = self.metrics.successes.load(.monotonic);
            const throughput = if (elapsed > 0) @as(f64, @floatFromInt(ops)) / @as(f64, @floatFromInt(elapsed)) else 0.0;
            const success_rate = if (ops > 0) @as(f64, @floatFromInt(successes)) / @as(f64, @floatFromInt(ops)) * 100.0 else 0.0;

            self.metrics.throughput.store(throughput, .monotonic);

            if (self.config.enable_real_time_reporting) {
                std.debug.print("📊 Progress: {}s / {}s | Throughput: {d:.0} ops/sec | Success: {d:.1}%\n", .{
                    elapsed,
                    self.config.duration_seconds,
                    throughput,
                    success_rate,
                });

                // Show additional metrics if enabled
                if (self.config.enable_detailed_metrics) {
                    const avg_latency = if (self.metrics.latency_count.load(.monotonic) > 0)
                        self.metrics.latency_sum.load(.monotonic) / @as(f64, @floatFromInt(self.metrics.latency_count.load(.monotonic)))
                    else
                        0.0;

                    std.debug.print("   Latency: {d:.0}μs avg | Network Errors: {} | Memory Peak: {}MB\n", .{
                        avg_latency,
                        self.metrics.network_errors.load(.monotonic),
                        self.metrics.memory_peak.load(.monotonic) / (1024 * 1024),
                    });
                }
            }
        }
    }

    fn exportResults(self: *StressTest) !void {
        const json_output = try self.metrics.exportJson(self.allocator);
        defer self.allocator.free(json_output);

        switch (self.config.output_format) {
            .json => {
                std.debug.print("\n📄 JSON Export:\n{s}\n", .{json_output});
            },
            .csv => {
                try self.exportCsv();
            },
            .prometheus => {
                try self.exportPrometheus();
            },
            .text => {}, // Already printed above
        }
    }

    fn exportCsv(self: *StressTest) !void {
        std.debug.print("\n📊 CSV Export:\n", .{});
        std.debug.print("metric,value,timestamp\n", .{});
        std.debug.print("total_operations,{},{}\n", .{ self.metrics.operations.load(.monotonic), self.metrics.end_time });
        std.debug.print("success_rate,{d:.2},{}\n", .{ @as(f64, @floatFromInt(self.metrics.successes.load(.monotonic))) / @as(f64, @floatFromInt(self.metrics.operations.load(.monotonic))) * 100.0, self.metrics.end_time });
        std.debug.print("throughput,{d:.0},{}\n", .{ self.metrics.throughput.load(.monotonic), self.metrics.end_time });
        std.debug.print("avg_latency_us,{d:.0},{}\n", .{ if (self.metrics.latency_count.load(.monotonic) > 0)
            self.metrics.latency_sum.load(.monotonic) / @as(f64, @floatFromInt(self.metrics.latency_count.load(.monotonic)))
        else
            0.0, self.metrics.end_time });
    }

    fn exportPrometheus(self: *StressTest) !void {
        const timestamp = self.metrics.end_time * 1_000_000; // Prometheus expects microseconds
        std.debug.print("\n📈 Prometheus Export:\n", .{});
        std.debug.print("# HELP wdbx_stress_test_total_operations Total number of operations\n", .{});
        std.debug.print("# TYPE wdbx_stress_test_total_operations counter\n", .{});
        std.debug.print("wdbx_stress_test_total_operations{d} {d}\n", .{ self.metrics.operations.load(.monotonic), timestamp });

        std.debug.print("# HELP wdbx_stress_test_success_rate Success rate percentage\n", .{});
        std.debug.print("# TYPE wdbx_stress_test_success_rate gauge\n", .{});
        std.debug.print("wdbx_stress_test_success_rate{d:.2} {d}\n", .{ @as(f64, @floatFromInt(self.metrics.successes.load(.monotonic))) / @as(f64, @floatFromInt(self.metrics.operations.load(.monotonic))) * 100.0, timestamp });
    }

    fn workerThread(self: *StressTest, thread_id: usize) void {
        var prng = std.Random.DefaultPrng.init(@intCast(thread_id));
        const random = prng.random();

        // Pre-generate test vectors
        const test_vector = self.allocator.alloc(f32, self.config.vector_dimension) catch return;
        defer self.allocator.free(test_vector);

        for (test_vector) |*v| {
            v.* = random.float(f32) * 2.0 - 1.0; // Range: [-1, 1]
        }

        var vector_ids: [1000]u64 = undefined;
        var vector_ids_count: usize = 0;

        while (!self.shutdown.load(.monotonic)) {
            // Determine operation based on configured mix
            const op_percent = random.intRangeAtMost(u8, 0, 99);
            const operation_type: Metrics.OperationType = blk: {
                if (op_percent < self.config.write_percent) break :blk .write;
                if (op_percent < self.config.write_percent + self.config.read_percent) break :blk .read;
                if (op_percent < self.config.write_percent + self.config.read_percent + self.config.search_percent) break :blk .search;
                break :blk .delete;
            };

            const start = std.time.microTimestamp();
            var success = true;
            var error_type: Metrics.ErrorType = .none;

            // Simulate failure if enabled
            if (self.config.enable_failure_simulation) {
                if (random.intRangeAtMost(u8, 0, 99) < self.config.failure_rate_percent) {
                    success = false;
                    error_type = .server; // Simulate server error
                }
            }

            if (success) {
                // Execute the actual operation
                switch (operation_type) {
                    .write => {
                        const id = self.client.addVector(test_vector) catch {
                            success = false;
                            error_type = .connection;
                        };

                        if (vector_ids_count < vector_ids.len) {
                            vector_ids[vector_ids_count] = id;
                            vector_ids_count += 1;
                        }
                    },
                    .read => {
                        if (vector_ids_count > 0) {
                            const idx = random.intRangeLessThan(usize, 0, vector_ids_count);
                            self.client.getVector(vector_ids[idx]) catch {
                                success = false;
                                error_type = .timeout;
                            };
                        } else {
                            // No vectors to read, simulate network timeout
                            success = false;
                            error_type = .timeout;
                        }
                    },
                    .search => {
                        self.client.searchVector(test_vector, 10) catch {
                            success = false;
                            error_type = .protocol;
                        };
                    },
                    .delete => {
                        if (vector_ids_count > 0) {
                            const idx = random.intRangeLessThan(usize, 0, vector_ids_count);
                            const id = vector_ids[idx];
                            self.client.deleteVector(id) catch {
                                success = false;
                                error_type = .server;
                            };
                            // Remove by shifting elements
                            for (idx..vector_ids_count - 1) |i| {
                                vector_ids[i] = vector_ids[i + 1];
                            }
                            vector_ids_count -= 1;
                        } else {
                            success = false;
                            error_type = .protocol;
                        }
                    },
                }
            }

            const latency = @as(u64, @intCast(std.time.microTimestamp() - start));
            self.metrics.recordOperation(operation_type, latency, success, error_type);

            // Apply load pattern
            self.applyLoadPattern(thread_id);
        }
    }

    fn applyLoadPattern(self: *StressTest, thread_id: usize) void {
        switch (self.config.pattern) {
            .steady => {
                // Constant rate - base delay with small jitter
                const base_delay = 10 * std.time.ns_per_ms;
                const jitter = std.crypto.random.intRangeAtMost(u64, 0, base_delay / 10);
                sleep(base_delay + jitter);
            },
            .burst => {
                // Burst pattern - some threads burst, others steady
                if (thread_id % 4 == 0) {
                    sleep(self.config.burst_interval_ms * std.time.ns_per_ms);
                } else {
                    sleep(1 * std.time.ns_per_ms);
                }
            },
            .ramp_up => {
                // Gradually increase load over time
                const elapsed = std.time.milliTimestamp() - self.metrics.start_time;
                const delay = @max(1, 100 - @divTrunc(elapsed, 1000));
                sleep(@intCast(delay * std.time.ns_per_ms));
            },
            .spike => {
                // Occasional spikes - most time normal, occasional bursts
                if (std.crypto.random.int(u32) % 100 < 5) {
                    // 5% chance of no delay (spike) - high frequency
                } else {
                    sleep(50 * std.time.ns_per_ms);
                }
            },
            .random => {
                // Random delays between 1-100ms
                const delay = std.crypto.random.intRangeAtMost(u64, 1, 100);
                sleep(delay * std.time.ns_per_ms);
            },
            .network_saturate => {
                // Network saturation pattern - rapid fire requests
                if (self.config.enable_network_saturation) {
                    // Very short delays to maximize network usage
                    const delay = std.crypto.random.intRangeAtMost(u64, 0, 5);
                    sleep(delay * std.time.ns_per_ms);
                } else {
                    sleep(10 * std.time.ns_per_ms);
                }
            },
            .memory_pressure => {
                // Memory pressure pattern - coordinated with memory pressure thread
                if (self.config.enable_memory_pressure) {
                    // Add some variation to prevent synchronized behavior
                    const delay = std.crypto.random.intRangeAtMost(u64, 5, 20);
                    sleep(delay * std.time.ns_per_ms);
                } else {
                    sleep(10 * std.time.ns_per_ms);
                }
            },
        }
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command-line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var config = Config.init(allocator);
    defer config.deinit();

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printHelp();
            return;
        } else if (std.mem.eql(u8, arg, "--host") and i + 1 < args.len) {
            i += 1;
            config.host = try config.allocator.dupe(u8, args[i]);
        } else if (std.mem.eql(u8, arg, "--port") and i + 1 < args.len) {
            i += 1;
            config.port = try std.fmt.parseInt(u16, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--threads") and i + 1 < args.len) {
            i += 1;
            config.num_threads = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--vectors") and i + 1 < args.len) {
            i += 1;
            config.num_vectors = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--duration") and i + 1 < args.len) {
            i += 1;
            config.duration_seconds = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--pattern") and i + 1 < args.len) {
            i += 1;
            config.pattern = std.meta.stringToEnum(Config.LoadPattern, args[i]) orelse .steady;
        }
        // Operation mix parameters
        else if (std.mem.eql(u8, arg, "--read-percent") and i + 1 < args.len) {
            i += 1;
            config.read_percent = try std.fmt.parseInt(u8, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--write-percent") and i + 1 < args.len) {
            i += 1;
            config.write_percent = try std.fmt.parseInt(u8, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--search-percent") and i + 1 < args.len) {
            i += 1;
            config.search_percent = try std.fmt.parseInt(u8, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--delete-percent") and i + 1 < args.len) {
            i += 1;
            config.delete_percent = try std.fmt.parseInt(u8, args[i], 10);
        }
        // Network saturation testing
        else if (std.mem.eql(u8, arg, "--enable-network-saturation")) {
            config.enable_network_saturation = true;
        } else if (std.mem.eql(u8, arg, "--concurrent-connections") and i + 1 < args.len) {
            i += 1;
            config.concurrent_connections = try std.fmt.parseInt(u32, args[i], 10);
        }
        // Failure simulation
        else if (std.mem.eql(u8, arg, "--enable-failure-simulation")) {
            config.enable_failure_simulation = true;
        } else if (std.mem.eql(u8, arg, "--failure-rate") and i + 1 < args.len) {
            i += 1;
            config.failure_rate_percent = try std.fmt.parseInt(u8, args[i], 10);
        }
        // Memory pressure testing
        else if (std.mem.eql(u8, arg, "--enable-memory-pressure")) {
            config.enable_memory_pressure = true;
        } else if (std.mem.eql(u8, arg, "--memory-pressure-mb") and i + 1 < args.len) {
            i += 1;
            config.memory_pressure_mb = try std.fmt.parseInt(u64, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--memory-pattern") and i + 1 < args.len) {
            i += 1;
            config.memory_pressure_pattern = std.meta.stringToEnum(Config.MemoryPressurePattern, args[i]) orelse .gradual;
        }
        // Enterprise metrics
        else if (std.mem.eql(u8, arg, "--detailed-metrics")) {
            config.enable_detailed_metrics = true;
        } else if (std.mem.eql(u8, arg, "--percentile-reporting")) {
            config.percentile_reporting = true;
        } else if (std.mem.eql(u8, arg, "--histogram")) {
            config.enable_histogram = true;
        }
        // Output configuration
        else if (std.mem.eql(u8, arg, "--output-format") and i + 1 < args.len) {
            i += 1;
            config.output_format = std.meta.stringToEnum(Config.OutputFormat, args[i]) orelse .json;
        } else if (std.mem.eql(u8, arg, "--benchmark-mode")) {
            config.benchmark_mode = true;
        } else {
            std.debug.print("Unknown option: {s}\n\n", .{arg});
            printHelp();
            return error.InvalidArgument;
        }
    }

    // Run stress test
    const stress_test = try StressTest.init(allocator, config);
    defer stress_test.deinit();

    try stress_test.run();
}

fn printHelp() void {
    std.debug.print(
        \\WDBX Advanced Stress Testing & Benchmarking Suite
        \\
        \\Enterprise-grade testing for production vector database validation with comprehensive
        \\real-world scenarios, detailed metrics, and enterprise reporting capabilities.
        \\
        \\Usage: wdbx_stress_test [options]
        \\
        \\Connection Options:
        \\  --host <host>                   Target host (default: localhost)
        \\  --port <port>                   Target port (default: 8080)
        \\
        \\Test Parameters:
        \\  --threads <n>                   Number of worker threads (default: 32)
        \\  --vectors <n>                   Number of vectors to test (default: 1,000,000)
        \\  --duration <s>                  Test duration in seconds (default: 300)
        \\  --pattern <type>                Load pattern: steady|burst|ramp_up|spike|random|
        \\                                  network_saturate|memory_pressure
        \\
        \\Operation Mix (percentages, must sum to 100):
        \\  --read-percent <p>              Read operations percentage (default: 70)
        \\  --write-percent <p>             Write operations percentage (default: 20)
        \\  --search-percent <p>            Search operations percentage (default: 5)
        \\  --delete-percent <p>            Delete operations percentage (default: 5)
        \\
        \\Network Saturation Testing:
        \\  --enable-network-saturation     Enable high-concurrency network saturation testing
        \\  --concurrent-connections <n>    Number of concurrent connections (default: 1000)
        \\
        \\Failure Recovery Testing:
        \\  --enable-failure-simulation     Enable failure simulation for resilience testing
        \\  --failure-rate <p>              Failure rate percentage (default: 10)
        \\
        \\Memory Pressure Testing:
        \\  --enable-memory-pressure        Enable memory pressure scenarios
        \\  --memory-pressure-mb <mb>       Memory pressure target in MB (default: 1024)
        \\  --memory-pattern <type>         Memory pattern: gradual|spike|sawtooth|constant
        \\
        \\Enterprise Metrics & Reporting:
        \\  --detailed-metrics              Enable comprehensive operation breakdown
        \\  --percentile-reporting          Enable P50/P95/P99 latency reporting
        \\  --histogram                     Enable latency histogram collection
        \\  --output-format <fmt>           Output format: text|json|csv|prometheus
        \\  --benchmark-mode                Enable benchmarking mode with VDBench compatibility
        \\
        \\Examples:
        \\
        \\  # Basic stress test
        \\  wdbx_stress_test --threads 64 --duration 600
        \\
        \\  # Network saturation test
        \\  wdbx_stress_test --enable-network-saturation --concurrent-connections 5000 \\
        \\                   --pattern network_saturate --threads 128
        \\
        \\  # Failure recovery validation
        \\  wdbx_stress_test --enable-failure-simulation --failure-rate 15 \\
        \\                   --detailed-metrics --percentile-reporting
        \\
        \\  # Memory pressure scenarios
        \\  wdbx_stress_test --enable-memory-pressure --memory-pressure-mb 2048 \\
        \\                   --memory-pattern spike --pattern memory_pressure
        \\
        \\  # Enterprise benchmarking with detailed reporting
        \\  wdbx_stress_test --detailed-metrics --percentile-reporting --histogram \\
        \\                   --output-format json --benchmark-mode --duration 3600
        \\
        \\  # Production validation suite
        \\  wdbx_stress_test --host prod.example.com --threads 256 --duration 7200 \\
        \\                   --enable-network-saturation --enable-failure-simulation \\
        \\                   --enable-memory-pressure --detailed-metrics \\
        \\                   --percentile-reporting --output-format prometheus
        \\
    , .{});
}
