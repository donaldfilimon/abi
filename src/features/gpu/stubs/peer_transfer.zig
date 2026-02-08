const std = @import("std");

pub const PeerTransferManager = struct {
    pub fn init(_: std.mem.Allocator) !@This() {
        return error.GpuDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const TransferCapability = struct {
    peer_to_peer: bool = false,
    unified_addressing: bool = false,
};

pub const TransferHandle = struct {
    id: u64 = 0,
    pub fn isValid(_: TransferHandle) bool {
        return false;
    }
};

pub const TransferStatus = enum { pending, in_progress, completed, failed };

pub const TransferOptions = struct {
    async_transfer: bool = true,
    compress: bool = false,
};

pub const TransferError = error{
    GpuDisabled,
    PeerNotReachable,
    TransferFailed,
    OutOfMemory,
};

pub const TransferStats = struct {
    bytes_transferred: u64 = 0,
    transfer_count: u64 = 0,
    avg_bandwidth_gbps: f64 = 0,
};

pub const DeviceBuffer = struct {
    device_id: usize = 0,
    size: usize = 0,
};

pub const RecoveryStrategy = enum { retry, failover, abort };
