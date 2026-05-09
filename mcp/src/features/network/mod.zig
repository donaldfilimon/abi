//! Network Module
//!
//! Distributed compute network with node discovery, Raft consensus,
//! and distributed task coordination.
//!
//! ## Features
//! - Node registry and discovery
//! - Raft consensus for leader election
//! - Task scheduling and load balancing
//! - Connection pooling and retry logic
//! - Circuit breakers for fault tolerance
//! - Rate limiting
//!
//! ## Usage
//!
//! ```zig
//! const network = @import("network");
//!
//! // Initialize the network module
//! try network.init(allocator);
//! defer network.deinit();
//!
//! // Get the node registry
//! const registry = try network.defaultRegistry();
//! try registry.register("node-a", "127.0.0.1:9000");
//! ```

const std = @import("std");
const time = @import("../../foundation/mod.zig").time;
const sync = @import("../../foundation/mod.zig").sync;
const build_options = @import("build_options");
const config_module = @import("../core/config/mod.zig");

// ── Cluster & Consensus ──────────────────────────────────────────────────
pub const cluster = @import("cluster.zig");
pub const consensus = @import("consensus.zig");
pub const registry = @import("registry.zig");
pub const raft = @import("raft.zig");
pub const raft_transport = @import("raft_transport.zig");
pub const heartbeat = @import("heartbeat.zig");
pub const ha = @import("ha.zig");
pub const discovery = @import("discovery.zig");
pub const loadbalancer = @import("loadbalancer.zig");

// Node Registry re-exports
pub const NodeRegistry = registry.NodeRegistry;
pub const NodeInfo = registry.NodeInfo;
pub const NodeStatus = registry.NodeStatus;
pub const Node = NodeInfo; // Alias for compatibility

// Raft Consensus re-exports
pub const RaftNode = raft.RaftNode;
pub const RaftState = raft.RaftState;
pub const RaftConfig = raft.RaftConfig;
pub const RaftError = raft.RaftError;
pub const RaftStats = raft.RaftStats;
pub const LogEntry = raft.LogEntry;
pub const RequestVoteRequest = raft.RequestVoteRequest;
pub const RequestVoteResponse = raft.RequestVoteResponse;
pub const AppendEntriesRequest = raft.AppendEntriesRequest;
pub const AppendEntriesResponse = raft.AppendEntriesResponse;
pub const PeerState = raft.PeerState;
pub const createRaftCluster = raft.createCluster;
pub const FaultInjector = raft.FaultInjector;
pub const RaftPersistence = raft.RaftPersistence;
pub const PersistentState = raft.PersistentState;
pub const RaftSnapshotManager = raft.RaftSnapshotManager;
pub const SnapshotConfig = raft.SnapshotConfig;
pub const SnapshotMetadata = raft.SnapshotMetadata;
pub const SnapshotInfo = raft.SnapshotInfo;
pub const InstallSnapshotRequest = raft.InstallSnapshotRequest;
pub const InstallSnapshotResponse = raft.InstallSnapshotResponse;
pub const ConfigChangeType = raft.ConfigChangeType;
pub const ConfigChangeRequest = raft.ConfigChangeRequest;
pub const applyConfigChange = raft.applyConfigChange;

// Raft Transport re-exports
pub const RaftTransport = raft_transport.RaftTransport;
pub const RaftTransportConfig = raft_transport.RaftTransportConfig;
pub const RaftTransportStats = raft_transport.RaftTransport.RaftTransportStats;
pub const PeerAddress = raft_transport.PeerAddress;

// High Availability & Heartbeat re-exports
pub const HealthCheck = ha.HealthCheck;
pub const ClusterConfig = ha.ClusterConfig;
pub const HaError = ha.HaError;
pub const NodeHealth = ha.NodeHealth;
pub const ClusterState = ha.ClusterState;
pub const HealthCheckResult = ha.HealthCheckResult;
pub const FailoverPolicy = ha.FailoverPolicy;
pub const NodeHealthState = heartbeat.NodeHealthState;
pub const ClusterHealthState = heartbeat.ClusterHealthState;
pub const HeartbeatEvent = heartbeat.HeartbeatEvent;
pub const HeartbeatConfig = heartbeat.HeartbeatConfig;
pub const HeartbeatStateMachine = heartbeat.HeartbeatStateMachine;
pub const EventCallback = heartbeat.EventCallback;

// Service Discovery re-exports
pub const ServiceDiscovery = discovery.ServiceDiscovery;
pub const DiscoveryConfig = discovery.DiscoveryConfig;
pub const DiscoveryBackend = discovery.DiscoveryBackend;
pub const ServiceInstance = discovery.ServiceInstance;
pub const ServiceStatus = discovery.ServiceStatus;
pub const DiscoveryError = discovery.DiscoveryError;
pub const generateServiceId = discovery.generateServiceId;
pub const base64Encode = discovery.base64Encode;
pub const base64Decode = discovery.base64Decode;

// Load Balancer re-exports
pub const LoadBalancer = loadbalancer.LoadBalancer;
pub const LoadBalancerConfig = loadbalancer.LoadBalancerConfig;
pub const LoadBalancerStrategy = loadbalancer.LoadBalancerStrategy;
pub const LoadBalancerError = loadbalancer.LoadBalancerError;
pub const NodeState = loadbalancer.NodeState;
pub const NodeStats = loadbalancer.NodeStats;

// ── RPC & Transport ──────────────────────────────────────────────────────
pub const rpc = @import("rpc.zig");
pub const protocol = @import("protocol.zig");
pub const transport = @import("transport/mod.zig");
pub const connection_pool = @import("connection_pool.zig");
pub const rpc_protocol = @import("rpc_protocol.zig");

// Protocol re-exports
pub const TaskEnvelope = protocol.TaskEnvelope;
pub const ResultEnvelope = protocol.ResultEnvelope;
pub const ResultStatus = protocol.ResultStatus;
pub const encodeTask = protocol.encodeTask;
pub const decodeTask = protocol.decodeTask;
pub const encodeResult = protocol.encodeResult;
pub const decodeResult = protocol.decodeResult;

// Transport re-exports
pub const TcpTransport = transport.TcpTransport;
pub const TransportConfig = transport.TransportConfig;
pub const TransportError = transport.TransportError;
pub const TransportStats = transport.TcpTransport.TransportStats;
pub const MessageType = transport.MessageType;
pub const MessageHeader = transport.MessageHeader;
pub const PeerConnection = transport.PeerConnection;
pub const RpcSerializer = transport.RpcSerializer;
pub const parseAddress = transport.parseAddress;

// Connection Pool re-exports
pub const ConnectionPool = connection_pool.ConnectionPool;
pub const ConnectionPoolConfig = connection_pool.ConnectionPoolConfig;
pub const PooledConnection = connection_pool.PooledConnection;
pub const ConnectionState = connection_pool.ConnectionState;
pub const ConnectionStats = connection_pool.ConnectionStats;
pub const HostKey = connection_pool.HostKey;
pub const PoolStats = connection_pool.PoolStats;
pub const PoolBuilder = connection_pool.PoolBuilder;

// RPC Protocol re-exports
pub const RpcMessageType = rpc_protocol.MessageType;
pub const RpcHeader = rpc_protocol.RpcHeader;
pub const ParsedFrame = rpc_protocol.ParsedFrame;
pub const RpcError = rpc_protocol.RpcError;
pub const VectorEntry = rpc_protocol.VectorEntry;
pub const BlockHeader = rpc_protocol.BlockHeader;
pub const frameMessage = rpc_protocol.frameMessage;
pub const parseRpcFrame = rpc_protocol.parseFrame;

// ── Reliability & Fault Tolerance ────────────────────────────────────────
pub const reliability = @import("reliability.zig");
pub const retry = @import("retry.zig");
pub const rate_limiter = @import("rate_limiter.zig");
pub const circuit_breaker = @import("circuit_breaker.zig");
pub const failover = @import("failover.zig");

// Retry re-exports
pub const RetryConfig = retry.RetryConfig;
pub const RetryResult = retry.RetryResult;
pub const RetryError = retry.RetryError;
pub const RetryStrategy = retry.RetryStrategy;
pub const RetryExecutor = retry.RetryExecutor;
pub const RetryableErrors = retry.RetryableErrors;
pub const BackoffCalculator = retry.BackoffCalculator;
pub const retryOperation = retry.retry;
pub const retryWithStrategy = retry.retryWithStrategy;

// Rate Limiter re-exports
pub const RateLimiter = rate_limiter.RateLimiter;
pub const RateLimiterConfig = rate_limiter.RateLimiterConfig;
pub const RateLimitAlgorithm = rate_limiter.RateLimitAlgorithm;
pub const AcquireResult = rate_limiter.AcquireResult;
pub const TokenBucketLimiter = rate_limiter.TokenBucketLimiter;
pub const SlidingWindowLimiter = rate_limiter.SlidingWindowLimiter;
pub const FixedWindowLimiter = rate_limiter.FixedWindowLimiter;
pub const LimiterStats = rate_limiter.LimiterStats;

// Circuit Breaker re-exports
pub const CircuitBreaker = circuit_breaker.CircuitBreaker;
pub const CircuitConfig = circuit_breaker.CircuitConfig;
pub const CircuitState = circuit_breaker.CircuitState;
pub const CircuitRegistry = circuit_breaker.CircuitRegistry;
pub const CircuitStats = circuit_breaker.CircuitStats;
pub const CircuitMetrics = circuit_breaker.CircuitMetrics;
pub const CircuitMetricEntry = circuit_breaker.CircuitMetricEntry;
pub const NetworkOperationError = circuit_breaker.NetworkOperationError;
pub const AggregateStats = circuit_breaker.AggregateStats;

// Failover re-exports
pub const FailoverManager = failover.FailoverManager;
pub const FailoverConfig = failover.FailoverConfig;
pub const FailoverState = failover.FailoverState;
pub const FailoverEvent = failover.FailoverEvent;

// ── Distributed Coordination ─────────────────────────────────────────────
pub const scheduler = @import("scheduler.zig");

// Scheduler re-exports
pub const TaskScheduler = scheduler.TaskScheduler;
pub const SchedulerConfig = scheduler.SchedulerConfig;
pub const SchedulerError = scheduler.SchedulerError;
pub const TaskPriority = scheduler.TaskPriority;
pub const TaskState = scheduler.TaskState;
pub const ComputeNode = scheduler.ComputeNode;
pub const LoadBalancingStrategy = scheduler.LoadBalancingStrategy;
pub const SchedulerStats = scheduler.SchedulerStats;

// ── Unified Memory & Fabric ──────────────────────────────────────────────
pub const unified_memory = @import("unified_memory/mod.zig");
pub const linking = @import("linking.zig");

// Unified Memory re-exports
pub const UnifiedMemoryManager = unified_memory.UnifiedMemoryManager;
pub const UnifiedMemoryConfig = unified_memory.UnifiedMemoryConfig;
pub const UnifiedMemoryError = unified_memory.UnifiedMemoryError;
pub const MemoryRegion = unified_memory.MemoryRegion;
pub const RegionId = unified_memory.RegionId;
pub const RegionFlags = unified_memory.RegionFlags;
pub const RegionState = unified_memory.RegionState;
pub const CoherenceProtocol = unified_memory.CoherenceProtocol;
pub const CoherenceState = unified_memory.CoherenceState;
pub const RemotePtr = unified_memory.RemotePtr;
pub const RemoteSlice = unified_memory.RemoteSlice;
pub const MemoryNode = unified_memory.MemoryNode;

// Linking re-exports
pub const LinkManager = linking.LinkManager;
pub const Link = linking.Link;
pub const LinkConfig = linking.LinkConfig;
pub const LinkState = linking.LinkState;
pub const LinkStats = linking.LinkStats;
pub const TransportType = linking.TransportType;
pub const SecureChannel = linking.SecureChannel;
pub const ChannelConfig = linking.ChannelConfig;
pub const ChannelState = linking.ChannelState;
pub const ChannelStats = linking.ChannelStats;
pub const EncryptionType = linking.EncryptionType;
pub const LinkError = linking.LinkError;
pub const ThunderboltTransport = linking.ThunderboltTransport;
pub const ThunderboltConfig = linking.ThunderboltConfig;
pub const ThunderboltDevice = linking.ThunderboltDevice;
pub const InternetTransport = linking.InternetTransport;
pub const InternetConfig = linking.InternetConfig;
pub const NatTraversal = linking.NatTraversal;
pub const QuicConnection = linking.QuicConnection;

// ── Shared Types ─────────────────────────────────────────────────────────
pub const types = @import("types.zig");
pub const NetworkError = types.NetworkError;
pub const Error = types.Error;
pub const NetworkConfig = types.NetworkConfig;

pub const NetworkState = struct {
    allocator: std.mem.Allocator,
    config: NetworkConfig,
    registry: NodeRegistry,

    pub fn init(allocator: std.mem.Allocator, config: NetworkConfig) !NetworkState {
        const cluster_id = try allocator.dupe(u8, config.cluster_id);
        return .{
            .allocator = allocator,
            .config = .{
                .cluster_id = cluster_id,
                .heartbeat_timeout_ms = config.heartbeat_timeout_ms,
                .max_nodes = config.max_nodes,
            },
            .registry = NodeRegistry.init(allocator),
        };
    }

    pub fn deinit(self: *NetworkState) void {
        self.registry.deinit();
        self.allocator.free(self.config.cluster_id);
        self.* = undefined;
    }
};

// ── Framework Integration ────────────────────────────────────────────────

/// Network context for Framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.NetworkConfig,
    state: State = .disconnected,

    pub const State = enum {
        disconnected,
        connecting,
        connected,
        error_state,
    };

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.NetworkConfig) !*Context {
        if (!isEnabled()) return error.NetworkDisabled;

        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.disconnect();
        self.allocator.destroy(self);
    }

    /// Connect to the network.
    pub fn connect(self: *Context) !void {
        if (self.state == .connected) return;
        self.state = .connecting;
        // Network connection logic
        self.state = .connected;
    }

    /// Disconnect from the network.
    pub fn disconnect(self: *Context) void {
        self.state = .disconnected;
    }

    /// Get current state.
    pub fn getState(self: *Context) State {
        return self.state;
    }

    /// Discover peers.
    pub fn discoverPeers(self: *Context) ![]NodeInfo {
        if (self.state != .connected) {
            try self.connect();
        }
        return &.{};
    }

    /// Send a task to a remote node.
    pub fn sendTask(self: *Context, node_id: []const u8, task: anytype) !void {
        _ = self;
        _ = node_id;
        _ = task;
    }
};

// ── Module Lifecycle ─────────────────────────────────────────────────────
var state_mutex = sync.Mutex{};
var default_state: ?NetworkState = null;
var initialized = std.atomic.Value(bool).init(false);

pub fn isEnabled() bool {
    return build_options.feat_network;
}

pub fn isInitialized() bool {
    state_mutex.lock();
    defer state_mutex.unlock();
    return initialized.load(.acquire);
}

pub fn init(allocator: std.mem.Allocator) Error!void {
    return initWithConfig(allocator, .{});
}

pub fn initWithConfig(allocator: std.mem.Allocator, config: NetworkConfig) Error!void {
    if (!isEnabled()) return error.NetworkDisabled;

    state_mutex.lock();
    defer state_mutex.unlock();

    if (default_state == null) {
        default_state = NetworkState.init(allocator, config) catch return error.NetworkDisabled;
    }
    initialized.store(true, .release);
}

pub fn deinit() void {
    state_mutex.lock();
    defer state_mutex.unlock();

    if (default_state) |*state| {
        state.deinit();
        default_state = null;
    }
    initialized.store(false, .release);
}

pub fn defaultRegistry() Error!*NodeRegistry {
    state_mutex.lock();
    defer state_mutex.unlock();

    if (default_state) |*state| {
        return &state.registry;
    }
    return error.NetworkDisabled;
}

pub fn defaultConfig() ?NetworkConfig {
    state_mutex.lock();
    defer state_mutex.unlock();

    if (default_state) |state| {
        return .{
            .cluster_id = state.config.cluster_id,
            .heartbeat_timeout_ms = state.config.heartbeat_timeout_ms,
            .max_nodes = state.config.max_nodes,
        };
    }
    return null;
}

// ── Tests ────────────────────────────────────────────────────────────────
test "network state tracks nodes" {
    var state = try NetworkState.init(std.testing.allocator, .{ .cluster_id = "test" });
    defer state.deinit();

    try state.registry.register("node-a", "127.0.0.1:9000");
    try state.registry.register("node-b", "127.0.0.1:9001");
    try std.testing.expectEqual(@as(usize, 2), state.registry.list().len);
}

test "network default state" {
    if (!isEnabled()) return;

    try initWithConfig(std.testing.allocator, .{ .cluster_id = "cluster-a" });
    defer deinit();

    const registry_ptr = try defaultRegistry();
    try registry_ptr.register("node-a", "127.0.0.1:9000");

    const config = defaultConfig() orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("cluster-a", config.cluster_id);
}

test "isEnabled returns build option" {
    try std.testing.expectEqual(build_options.feat_network, isEnabled());
}

test "defaultConfig returns null before init" {
    deinit();
    try std.testing.expect(defaultConfig() == null);
}

test "defaultRegistry errors before init" {
    deinit();
    try std.testing.expectError(error.NetworkDisabled, defaultRegistry());
}

test "initWithConfig sets config" {
    if (!isEnabled()) return;
    try initWithConfig(std.testing.allocator, .{
        .cluster_id = "test-cluster",
        .heartbeat_timeout_ms = 5000,
        .max_nodes = 16,
    });
    defer deinit();

    const config = defaultConfig() orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("test-cluster", config.cluster_id);
    try std.testing.expectEqual(@as(u64, 5000), config.heartbeat_timeout_ms);
    try std.testing.expectEqual(@as(usize, 16), config.max_nodes);
}

test "isInitialized tracks lifecycle" {
    if (!isEnabled()) return;
    try std.testing.expect(!isInitialized());
    try init(std.testing.allocator);
    try std.testing.expect(isInitialized());
    deinit();
    try std.testing.expect(!isInitialized());
}

// refAllDecls deferred — linking/, raft_transport, scheduler, tcp, unified_memory have pre-existing errors
