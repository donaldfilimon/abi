//! Abi AI Framework
//! Ultra-high-performance AI framework with GPU acceleration and advanced features.
//!
//! This framework provides:
//! - Multi-persona AI agents with extensible backends
//! - GPU-accelerated vector database with WDBX-AI format
//! - SIMD-optimized text and vector processing
//! - Lock-free concurrent data structures
//! - Cross-platform optimizations
//! - Neural network training and inference
//! - High-performance LSP server
//! - Hot code reloading support

const std = @import("std");
const builtin = @import("builtin");

/// Framework version information
pub const version = std.SemanticVersion{
    .major = 1,
    .minor = 0,
    .patch = 0,
    .pre = "alpha.1",
};

/// Build-time feature detection
pub const features = struct {
    pub const has_simd = @import("simd/mod.zig").config.has_simd;
    pub const vector_width = @import("simd/mod.zig").config.vector_width;
    pub const is_wasm = builtin.target.cpu.arch == .wasm32 or builtin.target.cpu.arch == .wasm64;
};

/// Re-export common types for convenience
pub const Allocator = std.mem.Allocator;
pub const ArrayList = std.ArrayListUnmanaged;
pub const HashMap = std.HashMapUnmanaged;
pub const MultiArrayList = std.MultiArrayList;

/// Core framework modules with improved organization
pub const core = @import("core/mod.zig");
pub const ai = @import("ai/mod.zig");
pub const agent = @import("agent.zig");
pub const neural = @import("neural.zig");

/// Performance and optimization modules
pub const simd = @import("simd/mod.zig");

/// Web and networking modules
pub const web_server = @import("web_server.zig");

/// Database modules
pub const database = @import("database.zig");

/// Platform-specific modules
pub const platform = @import("platform.zig");

/// Comprehensive framework error types
pub const Error = error{
    // Initialization and Setup
    InitializationFailed,
    InvalidConfiguration,
    FeatureNotSupported,
    PlatformNotSupported,

    // Resource Management
    ResourceNotFound,
    MemoryAllocationFailed,
    OutOfMemory,

    // Hardware and Acceleration
    GPUNotAvailable,
    GPUError,
    SIMDNotSupported,

    // Network and I/O
    NetworkError,
    ConnectionFailed,
    TimeoutError,
    PermissionDenied,
    IOError,

    // Data and Processing
    DatabaseError,
    AIModelError,
    ValidationError,
    InvalidInput,
    ComputationError,

    // Concurrency and Threading
    ConcurrencyError,
    LockError,
    ThreadError,

    // Serialization and Persistence
    SerializationError,
    DeserializationError,
    FileFormatError,
    CorruptedData,
} || Allocator.Error || std.fs.File.OpenError || std.Thread.SpawnError;

/// Global framework configuration
pub const Config = struct {
    /// Memory management configuration
    memory: MemoryConfig = .{},

    /// Performance configuration
    performance: PerformanceConfig = .{},

    /// AI model configuration
    ai: AIConfig = .{},

    /// Network configuration
    network: NetworkConfig = .{},

    /// Logging configuration
    logging: LoggingConfig = .{},

    /// Security configuration
    security: SecurityConfig = .{},

    pub const MemoryConfig = struct {
        /// Maximum memory usage in bytes
        max_memory: usize = 2 * 1024 * 1024 * 1024, // 2GB for native

        /// Memory allocator type
        allocator_type: AllocatorType = .general_purpose,

        /// Enable memory pooling
        enable_pooling: bool = true,

        /// Cache line size for alignment
        cache_line_size: u32 = 64,
    };

    pub const PerformanceConfig = struct {
        /// Number of worker threads (0 = auto-detect)
        thread_pool_size: u32 = 0,

        /// Enable SIMD optimizations
        enable_simd: bool = true,

        /// Enable GPU acceleration
        enable_gpu: bool = false,

        /// Enable performance profiling
        enable_profiling: bool = false,

        /// Enable hot code reloading
        enable_hot_reload: bool = false,

        /// Batch size for operations
        batch_size: u32 = 32,
    };

    pub const AIConfig = struct {
        /// Default AI persona
        default_persona: ai.PersonaType = .adaptive,

        /// Maximum response length
        max_response_length: usize = 4096,

        /// Context window size
        context_window: usize = 8192,

        /// Temperature for generation
        temperature: f32 = 0.7,

        /// Enable safety filters
        enable_safety: bool = true,

        /// Model backend
        backend: ai.Backend = .local,
    };

    pub const NetworkConfig = struct {
        /// Maximum concurrent connections
        max_connections: u32 = 1000,

        /// Connection timeout in milliseconds
        timeout_ms: u32 = 30000,

        /// Enable compression
        enable_compression: bool = true,

        /// Buffer size for network operations
        buffer_size: usize = 64 * 1024,
    };

    pub const LoggingConfig = struct {
        /// Log level
        level: std.log.Level = .info,

        /// Enable structured logging
        structured: bool = true,

        /// Log output format
        format: LogFormat = .json,

        /// Enable telemetry
        enable_telemetry: bool = false,
    };

    pub const SecurityConfig = struct {
        /// Enable sandboxing
        enable_sandbox: bool = true,

        /// Enable input validation
        enable_validation: bool = true,

        /// Maximum input size
        max_input_size: usize = 1024 * 1024, // 1MB

        /// Allowed operations
        allowed_operations: OperationSet = OperationSet.safe(),
    };
};

/// Allocator types supported by the framework
pub const AllocatorType = enum {
    general_purpose,
    arena,
    thread_safe,
    pool,
    debug,
    tracy,
    custom,
};

/// Log output formats
pub const LogFormat = enum {
    text,
    json,
    structured,
    compact,
};

/// Operation permission set
pub const OperationSet = packed struct {
    file_read: bool = true,
    file_write: bool = false,
    network: bool = true,
    gpu: bool = true,
    system: bool = false,

    pub fn safe() OperationSet {
        return .{
            .file_read = true,
            .file_write = false,
            .network = true,
            .gpu = true,
            .system = false,
        };
    }

    pub fn all() OperationSet {
        return .{
            .file_read = true,
            .file_write = true,
            .network = true,
            .gpu = true,
            .system = true,
        };
    }
};

/// Global framework context
pub const Context = struct {
    allocator: Allocator,
    config: Config,

    const Self = @This();

    /// Initialize the framework context
    pub fn init(allocator: Allocator, config: Config) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        // Create appropriate allocator based on configuration
        const framework_allocator = try createAllocator(allocator, config.memory.allocator_type);

        self.* = .{
            .allocator = framework_allocator,
            .config = config,
        };

        std.log.info("Abi Framework v{} initialized", .{version});
        return self;
    }

    /// Deinitialize the framework
    pub fn deinit(self: *Self) void {
        const allocator = self.allocator;
        allocator.destroy(self);
    }

    /// Get or create the global context
    pub fn global() !*Self {
        const GlobalState = struct {
            var instance: ?*Context = null;
            var mutex: std.Thread.Mutex = .{};
        };

        GlobalState.mutex.lock();
        defer GlobalState.mutex.unlock();

        if (GlobalState.instance) |ctx| {
            return ctx;
        }

        // Create with default configuration
        const allocator = std.heap.page_allocator;

        GlobalState.instance = try init(allocator, .{});
        return GlobalState.instance.?;
    }
};

fn createAllocator(base_allocator: Allocator, allocator_type: AllocatorType) !Allocator {
    return switch (allocator_type) {
        .general_purpose => blk: {
            const gpa = try base_allocator.create(std.heap.GeneralPurposeAllocator(.{}));
            gpa.* = .{};
            break :blk gpa.allocator();
        },
        .arena => blk: {
            const arena = try base_allocator.create(std.heap.ArenaAllocator);
            arena.* = std.heap.ArenaAllocator.init(base_allocator);
            break :blk arena.allocator();
        },
        .thread_safe => blk: {
            // Wrap any allocator to make it thread-safe
            const tsa = try base_allocator.create(std.heap.ThreadSafeAllocator);
            tsa.* = .{ .child_allocator = base_allocator };
            break :blk tsa.allocator();
        },
        .pool => blk: {
            const pool = try base_allocator.create(std.heap.GeneralPurposeAllocator(.{}));
            pool.* = .{};
            break :blk pool.allocator();
        },
        .debug => blk: {
            const debug_alloc = try base_allocator.create(std.heap.GeneralPurposeAllocator(.{}));
            debug_alloc.* = .{};
            break :blk debug_alloc.allocator();
        },
        .tracy => base_allocator,
        .custom => base_allocator,
    };
}

/// Standard options for the framework
pub const std_options = .{
    .log_level = if (builtin.mode == .Debug) .warn else .info,
    .log_scope_levels = &.{
        .{ .scope = .abi_framework, .level = .warn },
        .{ .scope = .abi_ai, .level = .warn },
        .{ .scope = .abi_gpu, .level = .warn },
    },
};

test "abi framework" {
    std.testing.refAllDecls(@This());

    // Test framework initialization
    const allocator = std.testing.allocator;
    const ctx = try Context.init(allocator, .{});
    defer ctx.deinit();

    // Verify context is initialized
    try std.testing.expect(true); // Context creation succeeded
}

comptime {
    const supported_platforms = [_]std.Target.Os.Tag{
        .linux,     .windows, .macos,   .freebsd, .openbsd,      .netbsd,
        .dragonfly, .solaris, .illumos, .haiku,   .freestanding, .uefi,
        .wasi,
    };

    const current_os = builtin.target.os.tag;
    var is_supported = false;

    for (supported_platforms) |os| {
        if (current_os == os) {
            is_supported = true;
            break;
        }
    }

    if (!is_supported) {
        @compileError("Platform not supported: " ++ @tagName(current_os));
    }
}
