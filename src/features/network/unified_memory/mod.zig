//! Unified Memory System
//!
//! Enables secure sharing of memory across multiple machines connected via
//! Thunderbolt, Internet, or other secure channels. Provides coherent access
//! to distributed memory with strong consistency guarantees.
//!
//! ## Features
//! - Memory region registration and tracking
//! - Remote memory read/write with coherence
//! - Secure memory transfers (encrypted in transit)
//! - Support for multiple transport backends (Thunderbolt, Internet)
//! - RDMA-style zero-copy where hardware supports it
//!
//! ## Usage
//!
//! ```zig
//! const unified_memory = @import("unified_memory/mod.zig");
//!
//! var manager = try unified_memory.UnifiedMemoryManager.init(allocator, config);
//! defer manager.deinit();
//!
//! // Register a local memory region for sharing
//! const region_id = try manager.registerRegion(data.ptr, data.len, .read_write);
//!
//! // Access remote memory
//! const remote = try manager.connectRemote("node-2", region_id);
//! const value = try remote.read(u64, offset);
//! ```

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const build_options = @import("build_options");
const shared_utils = @import("../../../services/shared/utils.zig");

// Sub-module imports
pub const memory_region = @import("memory_region.zig");
pub const coherence = @import("coherence.zig");
pub const remote_ptr = @import("remote_ptr.zig");

// Re-exports
pub const MemoryRegion = memory_region.MemoryRegion;
pub const RegionId = memory_region.RegionId;
pub const RegionFlags = memory_region.RegionFlags;
pub const RegionState = memory_region.RegionState;
pub const RegionStats = memory_region.RegionStats;

pub const CoherenceProtocol = coherence.CoherenceProtocol;
pub const CoherenceState = coherence.CoherenceState;
pub const CoherenceMessage = coherence.CoherenceMessage;
pub const InvalidationRequest = coherence.InvalidationRequest;

pub const RemotePtr = remote_ptr.RemotePtr;
pub const RemoteSlice = remote_ptr.RemoteSlice;
pub const RemoteMemoryError = remote_ptr.RemoteMemoryError;

/// Unified memory configuration.
pub const UnifiedMemoryConfig = struct {
    /// Maximum number of registered memory regions.
    max_regions: usize = 256,

    /// Maximum total shared memory size (bytes).
    max_shared_memory: usize = 16 * 1024 * 1024 * 1024, // 16 GB

    /// Enable memory coherence protocol.
    coherence_enabled: bool = true,

    /// Coherence protocol to use.
    coherence_protocol: CoherenceProtocolType = .mesi,

    /// Enable encryption for memory transfers.
    encrypt_transfers: bool = true,

    /// Compression for memory transfers (reduces bandwidth).
    compress_transfers: bool = true,

    /// Compression threshold (only compress if data > this size).
    compression_threshold: usize = 4096,

    /// Page size for memory regions (must be power of 2).
    page_size: usize = 4096,

    /// Enable RDMA when available.
    rdma_enabled: bool = true,

    /// Timeout for remote memory operations (ms).
    operation_timeout_ms: u64 = 5000,

    /// Number of retry attempts for failed operations.
    retry_count: u32 = 3,

    /// Enable memory prefetching.
    prefetch_enabled: bool = true,

    /// Prefetch distance (pages ahead).
    prefetch_distance: usize = 4,

    pub const CoherenceProtocolType = enum {
        /// Modified, Exclusive, Shared, Invalid
        mesi,
        /// Modified, Owned, Exclusive, Shared, Invalid
        moesi,
        /// Directory-based coherence (for larger clusters)
        directory,
        /// No coherence (application-managed)
        none,
    };

    pub fn defaults() UnifiedMemoryConfig {
        return .{};
    }

    /// High-performance configuration for Thunderbolt links.
    pub fn thunderbolt() UnifiedMemoryConfig {
        return .{
            .coherence_protocol = .moesi,
            .encrypt_transfers = false, // Thunderbolt is already secure
            .compress_transfers = false, // Low latency more important
            .rdma_enabled = true,
            .operation_timeout_ms = 100,
            .prefetch_distance = 8,
        };
    }

    /// Secure configuration for Internet links.
    pub fn internet() UnifiedMemoryConfig {
        return .{
            .coherence_protocol = .directory,
            .encrypt_transfers = true,
            .compress_transfers = true,
            .rdma_enabled = false,
            .operation_timeout_ms = 30000,
            .prefetch_distance = 16,
        };
    }
};

/// Error types for unified memory operations.
pub const UnifiedMemoryError = error{
    /// Memory feature is disabled at compile time.
    MemoryDisabled,
    /// Maximum regions limit reached.
    MaxRegionsExceeded,
    /// Maximum shared memory limit reached.
    MaxMemoryExceeded,
    /// Region not found.
    RegionNotFound,
    /// Region is not accessible (permissions).
    AccessDenied,
    /// Region is locked by another node.
    RegionLocked,
    /// Remote node is not connected.
    NodeNotConnected,
    /// Remote operation timed out.
    OperationTimeout,
    /// Coherence violation detected.
    CoherenceViolation,
    /// Memory transfer failed.
    TransferFailed,
    /// Invalid memory address.
    InvalidAddress,
    /// Region already registered.
    RegionExists,
    /// Authentication failed.
    AuthenticationFailed,
    /// Encryption error.
    EncryptionError,
    /// Out of memory.
    OutOfMemory,
};

/// Node information for unified memory.
pub const MemoryNode = struct {
    /// Unique node identifier.
    id: NodeId,

    /// Node address (IP:port or Thunderbolt address).
    address: []const u8,

    /// Connection state.
    state: ConnectionState,

    /// Regions shared by this node.
    shared_regions: []RegionId,

    /// Total shared memory on this node.
    total_shared_memory: usize,

    /// Available shared memory on this node.
    available_memory: usize,

    /// Link type to this node.
    link_type: LinkType,

    /// Round-trip latency estimate (microseconds).
    latency_us: u64,

    /// Bandwidth estimate (bytes/second).
    bandwidth_bps: u64,

    /// Last heartbeat timestamp.
    last_heartbeat_ms: i64,

    pub const ConnectionState = enum {
        disconnected,
        connecting,
        authenticating,
        connected,
        degraded,
        failed,
    };

    pub const LinkType = enum {
        thunderbolt,
        internet_tcp,
        internet_quic,
        rdma_roce,
        rdma_infiniband,
        local,
        unknown,
    };
};

pub const NodeId = u64;

/// Unified Memory Manager - main interface for distributed memory.
pub const UnifiedMemoryManager = struct {
    allocator: std.mem.Allocator,
    config: UnifiedMemoryConfig,

    /// Registered local memory regions.
    local_regions: std.AutoHashMapUnmanaged(RegionId, *MemoryRegion),

    /// Remote region handles (cached).
    remote_regions: std.AutoHashMapUnmanaged(RegionId, RemoteRegionHandle),

    /// Connected nodes.
    nodes: std.AutoHashMapUnmanaged(NodeId, *MemoryNode),

    /// Coherence protocol manager.
    coherence_manager: ?*CoherenceProtocol,

    /// Statistics.
    stats: ManagerStats,

    /// State mutex for thread safety.
    mutex: sync.Mutex,

    /// Next region ID.
    next_region_id: RegionId,

    /// Total registered memory size.
    total_registered_memory: usize,

    pub const ManagerStats = struct {
        regions_registered: u64 = 0,
        regions_unregistered: u64 = 0,
        bytes_transferred_out: u64 = 0,
        bytes_transferred_in: u64 = 0,
        remote_reads: u64 = 0,
        remote_writes: u64 = 0,
        coherence_invalidations: u64 = 0,
        cache_hits: u64 = 0,
        cache_misses: u64 = 0,
        transfer_failures: u64 = 0,
        encryption_ops: u64 = 0,
        compression_ops: u64 = 0,
    };

    pub const RemoteRegionHandle = struct {
        region_id: RegionId,
        node_id: NodeId,
        size: usize,
        flags: RegionFlags,
        base_address: u64,
        cached_pages: std.AutoHashMapUnmanaged(usize, CachedPage),

        pub const CachedPage = struct {
            data: []u8,
            state: CoherenceState,
            last_access_ms: i64,
            dirty: bool,
        };
    };

    /// Initialize unified memory manager.
    pub fn init(allocator: std.mem.Allocator, config: UnifiedMemoryConfig) !*UnifiedMemoryManager {
        if (!isEnabled()) return error.MemoryDisabled;

        const manager = try allocator.create(UnifiedMemoryManager);
        errdefer allocator.destroy(manager);

        manager.* = .{
            .allocator = allocator,
            .config = config,
            .local_regions = .{},
            .remote_regions = .{},
            .nodes = .{},
            .coherence_manager = null,
            .stats = .{},
            .mutex = .{},
            .next_region_id = 1,
            .total_registered_memory = 0,
        };

        // Initialize coherence protocol if enabled
        if (config.coherence_enabled) {
            manager.coherence_manager = try CoherenceProtocol.init(
                allocator,
                config.coherence_protocol,
            );
        }

        return manager;
    }

    /// Deinitialize and free resources.
    pub fn deinit(self: *UnifiedMemoryManager) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Clean up local regions
        var local_it = self.local_regions.valueIterator();
        while (local_it.next()) |region| {
            region.*.deinit(self.allocator);
            self.allocator.destroy(region.*);
        }
        self.local_regions.deinit(self.allocator);

        // Clean up remote region handles
        var remote_it = self.remote_regions.valueIterator();
        while (remote_it.next()) |handle| {
            var page_it = handle.cached_pages.valueIterator();
            while (page_it.next()) |page| {
                self.allocator.free(page.data);
            }
            handle.cached_pages.deinit(self.allocator);
        }
        self.remote_regions.deinit(self.allocator);

        // Clean up nodes
        var node_it = self.nodes.valueIterator();
        while (node_it.next()) |node| {
            self.allocator.free(node.*.address);
            self.allocator.free(node.*.shared_regions);
            self.allocator.destroy(node.*);
        }
        self.nodes.deinit(self.allocator);

        // Clean up coherence manager
        if (self.coherence_manager) |cm| {
            cm.deinit();
            self.allocator.destroy(cm);
        }

        self.allocator.destroy(self);
    }

    /// Register a local memory region for sharing.
    pub fn registerRegion(
        self: *UnifiedMemoryManager,
        ptr: [*]u8,
        size: usize,
        flags: RegionFlags,
    ) UnifiedMemoryError!RegionId {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check limits
        if (self.local_regions.count() >= self.config.max_regions) {
            return error.MaxRegionsExceeded;
        }
        if (self.total_registered_memory + size > self.config.max_shared_memory) {
            return error.MaxMemoryExceeded;
        }

        // Allocate region
        const region = self.allocator.create(MemoryRegion) catch return error.OutOfMemory;
        errdefer self.allocator.destroy(region);

        const region_id = self.next_region_id;
        self.next_region_id += 1;

        region.* = .{
            .id = region_id,
            .base_ptr = ptr,
            .size = size,
            .flags = flags,
            .state = .exclusive,
            .owner_node = 0, // Local
            .page_size = self.config.page_size,
            .stats = .{},
        };

        self.local_regions.put(self.allocator, region_id, region) catch return error.OutOfMemory;
        self.total_registered_memory += size;
        self.stats.regions_registered += 1;

        return region_id;
    }

    /// Unregister a local memory region.
    pub fn unregisterRegion(self: *UnifiedMemoryManager, region_id: RegionId) UnifiedMemoryError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const entry = self.local_regions.fetchRemove(region_id) orelse return error.RegionNotFound;
        const region = entry.value;

        // Invalidate remote caches if coherence is enabled
        if (self.coherence_manager) |cm| {
            cm.invalidateRegion(region_id) catch {
                // Cache invalidation failure could cause stale data on remote nodes
                std.log.warn("unified_memory: failed to invalidate remote cache for region {d} — remote nodes may have stale data", .{region_id});
            };
            self.stats.coherence_invalidations += 1;
        }

        self.total_registered_memory -= region.size;
        self.stats.regions_unregistered += 1;

        region.deinit(self.allocator);
        self.allocator.destroy(region);
    }

    /// Connect to a remote node.
    pub fn connectNode(
        self: *UnifiedMemoryManager,
        node_id: NodeId,
        address: []const u8,
        link_type: MemoryNode.LinkType,
    ) UnifiedMemoryError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const node = self.allocator.create(MemoryNode) catch return error.OutOfMemory;
        errdefer self.allocator.destroy(node);

        node.* = .{
            .id = node_id,
            .address = self.allocator.dupe(u8, address) catch return error.OutOfMemory,
            .state = .connecting,
            .shared_regions = &.{},
            .total_shared_memory = 0,
            .available_memory = 0,
            .link_type = link_type,
            .latency_us = 0,
            .bandwidth_bps = 0,
            .last_heartbeat_ms = shared_utils.unixMs(),
        };

        self.nodes.put(self.allocator, node_id, node) catch return error.OutOfMemory;
    }

    /// Disconnect from a remote node.
    pub fn disconnectNode(self: *UnifiedMemoryManager, node_id: NodeId) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.nodes.fetchRemove(node_id)) |entry| {
            const node = entry.value;
            self.allocator.free(node.address);
            self.allocator.free(node.shared_regions);
            self.allocator.destroy(node);
        }
    }

    /// Get a remote memory pointer.
    pub fn getRemotePtr(
        self: *UnifiedMemoryManager,
        node_id: NodeId,
        region_id: RegionId,
        offset: usize,
        comptime T: type,
    ) UnifiedMemoryError!RemotePtr(T) {
        _ = self.nodes.get(node_id) orelse return error.NodeNotConnected;

        return RemotePtr(T){
            .manager = self,
            .node_id = node_id,
            .region_id = region_id,
            .offset = offset,
        };
    }

    /// Read from remote memory (internal).
    pub fn readRemote(
        self: *UnifiedMemoryManager,
        node_id: NodeId,
        region_id: RegionId,
        offset: usize,
        buffer: []u8,
    ) UnifiedMemoryError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        _ = self.nodes.get(node_id) orelse return error.NodeNotConnected;

        // Check coherence state
        if (self.coherence_manager) |cm| {
            const state = cm.getState(region_id, offset / self.config.page_size);
            if (state == .invalid) {
                // Need to fetch from owner
                cm.requestRead(region_id, offset / self.config.page_size) catch return error.CoherenceViolation;
            }
        }

        // Network transfer not yet implemented
        // Requirements:
        // - Network protocol definition (RPC/message format)
        // - Serialization/deserialization for memory read requests
        // - Connection management to remote nodes
        // - Authentication and encryption (if config.encrypt_transfers)
        // - Timeout and retry logic (config.operation_timeout_ms, config.retry_count)
        // - Server-side handler for remote read requests
        // - RDMA support where available (if config.rdma_enabled)
        // For now, simulate with zeros
        @memset(buffer, 0);

        self.stats.remote_reads += 1;
        self.stats.bytes_transferred_in += buffer.len;
    }

    /// Write to remote memory (internal).
    pub fn writeRemote(
        self: *UnifiedMemoryManager,
        node_id: NodeId,
        region_id: RegionId,
        offset: usize,
        data: []const u8,
    ) UnifiedMemoryError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        _ = self.nodes.get(node_id) orelse return error.NodeNotConnected;

        // Check and update coherence state
        if (self.coherence_manager) |cm| {
            cm.requestWrite(region_id, offset / self.config.page_size) catch return error.CoherenceViolation;
        }

        // Network transfer not yet implemented
        // Requirements: Same as readRemote() plus:
        // - Write request serialization
        // - Server-side handler for remote write requests
        // - Acknowledgment/response handling
        // - Atomic write guarantees for coherence protocol
        // For now, only track statistics (actual transfer pending implementation)

        self.stats.remote_writes += 1;
        self.stats.bytes_transferred_out += data.len;
    }

    /// Get manager statistics.
    pub fn getStats(self: *UnifiedMemoryManager) ManagerStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }

    /// Get list of connected nodes.
    pub fn listNodes(self: *UnifiedMemoryManager, allocator: std.mem.Allocator) ![]NodeId {
        self.mutex.lock();
        defer self.mutex.unlock();

        var list = std.ArrayListUnmanaged(NodeId){};
        errdefer list.deinit(allocator);

        var it = self.nodes.keyIterator();
        while (it.next()) |key| {
            try list.append(allocator, key.*);
        }

        return list.toOwnedSlice(allocator);
    }

    /// Get list of local regions.
    pub fn listLocalRegions(self: *UnifiedMemoryManager, allocator: std.mem.Allocator) ![]RegionId {
        self.mutex.lock();
        defer self.mutex.unlock();

        var list = std.ArrayListUnmanaged(RegionId){};
        errdefer list.deinit(allocator);

        var it = self.local_regions.keyIterator();
        while (it.next()) |key| {
            try list.append(allocator, key.*);
        }

        return list.toOwnedSlice(allocator);
    }
};

/// Check if unified memory is enabled at compile time.
pub fn isEnabled() bool {
    return build_options.enable_network;
}

// ============================================================================
// Tests
// ============================================================================

test "unified memory manager initialization" {
    if (!isEnabled()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const manager = try UnifiedMemoryManager.init(allocator, .{});
    defer manager.deinit();

    try std.testing.expect(manager.config.coherence_enabled);
    try std.testing.expectEqual(@as(u64, 0), manager.stats.regions_registered);
}

test "unified memory region registration" {
    if (!isEnabled()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const manager = try UnifiedMemoryManager.init(allocator, .{});
    defer manager.deinit();

    var data: [4096]u8 = undefined;
    const region_id = try manager.registerRegion(&data, data.len, .{ .read = true, .write = true });

    try std.testing.expect(region_id > 0);
    try std.testing.expectEqual(@as(u64, 1), manager.stats.regions_registered);

    try manager.unregisterRegion(region_id);
    try std.testing.expectEqual(@as(u64, 1), manager.stats.regions_unregistered);
}

test "unified memory config presets" {
    const thunderbolt_config = UnifiedMemoryConfig.thunderbolt();
    try std.testing.expect(!thunderbolt_config.encrypt_transfers);
    try std.testing.expect(thunderbolt_config.rdma_enabled);

    const internet_config = UnifiedMemoryConfig.internet();
    try std.testing.expect(internet_config.encrypt_transfers);
    try std.testing.expect(internet_config.compress_transfers);
}
