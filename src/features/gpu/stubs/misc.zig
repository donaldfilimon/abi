const std = @import("std");

// ============================================================================
// Peer Transfer stubs
// ============================================================================

pub const peer_transfer = struct {
    pub const PeerTransferManager = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) !@This() {
            return error.GpuDisabled;
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const TransferCapability = struct {};
    pub const TransferHandle = struct {
        id: u64 = 0,
    };
    pub const TransferStatus = enum { pending, in_progress, completed, failed };
    pub const TransferOptions = struct {};
    pub const TransferError = error{ GpuDisabled, TransferFailed, PeerUnavailable };
    pub const TransferStats = struct {};
    pub const DeviceBuffer = struct {};
    pub const RecoveryStrategy = enum { retry, failover, abort };
};

// ============================================================================
// Mega GPU orchestration stubs
// ============================================================================

pub const mega = struct {
    pub const Coordinator = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) !@This() {
            return error.GpuDisabled;
        }
        pub fn deinit(_: *@This()) void {}
    };
    pub const BackendInstance = struct {};
    pub const WorkloadProfile = struct {};
    pub const WorkloadCategory = enum { compute, memory, mixed };
    pub const ScheduleDecision = struct {};
    pub const Precision = enum { f32, f16, bf16, i8 };
};

// ============================================================================
// Sync/Performance stubs
// ============================================================================

pub const sync_event = struct {
    pub const SyncEvent = struct {
        pub fn init() @This() {
            return .{};
        }
    };
};

pub const kernel_ring = struct {
    pub const KernelRing = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) !@This() {
            return error.GpuDisabled;
        }
        pub fn deinit(_: *@This()) void {}
    };
};

pub const adaptive_tiling = struct {
    pub const AdaptiveTiling = struct {
        pub fn init(_: anytype) @This() {
            return .{};
        }
        pub const TileConfig = struct {};
    };
};

// ============================================================================
// Empty namespace stubs for sub-modules
// ============================================================================

pub const profiling = struct {};
pub const occupancy = struct {};
pub const fusion = struct {};
pub const memory_pool_advanced = struct {};
pub const memory_pool_lockfree = struct {};
pub const std_gpu_kernels = struct {};
pub const unified = struct {};
pub const unified_buffer = struct {};
pub const interface = struct {};
pub const cuda_loader = struct {};
pub const builtin_kernels = struct {};
pub const error_handling = struct {
    pub const ErrorContext = struct {
        code: GpuErrorCode = .unknown,
        error_type: GpuErrorType = .runtime,
        message: []const u8 = "GPU disabled",
    };
    pub const GpuErrorCode = enum {
        unknown,
        out_of_memory,
        device_lost,
        invalid_operation,
        compilation_failed,
    };
    pub const GpuErrorType = enum {
        runtime,
        compilation,
        resource,
        synchronization,
    };
};
