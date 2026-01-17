//! Network feature module for distributed compute coordination.
//!
//! Provides node registry, task/result serialization protocols, and cluster state
//! management for distributed computing scenarios.

const std = @import("std");
const build_options = @import("build_options");

const registry = @import("registry.zig");
const protocol = @import("protocol.zig");
const scheduler = @import("scheduler.zig");
const ha = @import("ha.zig");
const discovery = @import("discovery.zig");
const loadbalancer = @import("loadbalancer.zig");
pub const retry = @import("retry.zig");
pub const rate_limiter = @import("rate_limiter.zig");
pub const connection_pool = @import("connection_pool.zig");
pub const raft = @import("raft.zig");
pub const transport = @import("transport.zig");
pub const raft_transport = @import("raft_transport.zig");
pub const circuit_breaker = @import("circuit_breaker.zig");

pub const NodeRegistry = registry.NodeRegistry;
pub const NodeInfo = registry.NodeInfo;
pub const NodeStatus = registry.NodeStatus;

pub const TaskEnvelope = protocol.TaskEnvelope;
pub const ResultEnvelope = protocol.ResultEnvelope;
pub const ResultStatus = protocol.ResultStatus;
pub const encodeTask = protocol.encodeTask;
pub const decodeTask = protocol.decodeTask;
pub const encodeResult = protocol.encodeResult;
pub const decodeResult = protocol.decodeResult;

pub const TaskScheduler = scheduler.TaskScheduler;
pub const SchedulerConfig = scheduler.SchedulerConfig;
pub const SchedulerError = scheduler.SchedulerError;
pub const TaskPriority = scheduler.TaskPriority;
pub const TaskState = scheduler.TaskState;
pub const ComputeNode = scheduler.ComputeNode;
pub const LoadBalancingStrategy = scheduler.LoadBalancingStrategy;
pub const SchedulerStats = scheduler.SchedulerStats;

pub const HealthCheck = ha.HealthCheck;
pub const ClusterConfig = ha.ClusterConfig;
pub const HaError = ha.HaError;
pub const NodeHealth = ha.NodeHealth;
pub const ClusterState = ha.ClusterState;
pub const HealthCheckResult = ha.HealthCheckResult;
pub const FailoverPolicy = ha.FailoverPolicy;

// Service Discovery exports
pub const ServiceDiscovery = discovery.ServiceDiscovery;
pub const DiscoveryConfig = discovery.DiscoveryConfig;
pub const DiscoveryBackend = discovery.DiscoveryBackend;
pub const ServiceInstance = discovery.ServiceInstance;
pub const ServiceStatus = discovery.ServiceStatus;
pub const DiscoveryError = discovery.DiscoveryError;

// Load Balancer exports
pub const LoadBalancer = loadbalancer.LoadBalancer;
pub const LoadBalancerConfig = loadbalancer.LoadBalancerConfig;
pub const LoadBalancerStrategy = loadbalancer.LoadBalancerStrategy;
pub const LoadBalancerError = loadbalancer.LoadBalancerError;
pub const NodeState = loadbalancer.NodeState;
pub const NodeStats = loadbalancer.NodeStats;

// Retry exports
pub const RetryConfig = retry.RetryConfig;
pub const RetryResult = retry.RetryResult;
pub const RetryError = retry.RetryError;
pub const RetryStrategy = retry.RetryStrategy;
pub const RetryExecutor = retry.RetryExecutor;
pub const RetryableErrors = retry.RetryableErrors;
pub const BackoffCalculator = retry.BackoffCalculator;
pub const retryOperation = retry.retry;
pub const retryWithStrategy = retry.retryWithStrategy;

// Rate Limiter exports
pub const RateLimiter = rate_limiter.RateLimiter;
pub const RateLimiterConfig = rate_limiter.RateLimiterConfig;
pub const RateLimitAlgorithm = rate_limiter.RateLimitAlgorithm;
pub const AcquireResult = rate_limiter.AcquireResult;
pub const TokenBucketLimiter = rate_limiter.TokenBucketLimiter;
pub const SlidingWindowLimiter = rate_limiter.SlidingWindowLimiter;
pub const FixedWindowLimiter = rate_limiter.FixedWindowLimiter;
pub const LimiterStats = rate_limiter.LimiterStats;

// Connection Pool exports
pub const ConnectionPool = connection_pool.ConnectionPool;
pub const ConnectionPoolConfig = connection_pool.ConnectionPoolConfig;
pub const PooledConnection = connection_pool.PooledConnection;
pub const ConnectionState = connection_pool.ConnectionState;
pub const ConnectionStats = connection_pool.ConnectionStats;
pub const HostKey = connection_pool.HostKey;
pub const PoolStats = connection_pool.PoolStats;
pub const PoolBuilder = connection_pool.PoolBuilder;

// Raft consensus exports
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

// Raft persistence exports
pub const RaftPersistence = raft.RaftPersistence;
pub const PersistentState = raft.PersistentState;

// Raft snapshot exports
pub const RaftSnapshotManager = raft.RaftSnapshotManager;
pub const SnapshotConfig = raft.SnapshotConfig;
pub const SnapshotMetadata = raft.SnapshotMetadata;
pub const SnapshotInfo = raft.SnapshotInfo;
pub const InstallSnapshotRequest = raft.InstallSnapshotRequest;
pub const InstallSnapshotResponse = raft.InstallSnapshotResponse;

// Raft membership change exports
pub const ConfigChangeType = raft.ConfigChangeType;
pub const ConfigChangeRequest = raft.ConfigChangeRequest;
pub const applyConfigChange = raft.applyConfigChange;

// Transport exports
pub const TcpTransport = transport.TcpTransport;
pub const TransportConfig = transport.TransportConfig;
pub const TransportError = transport.TransportError;
pub const TransportStats = transport.TcpTransport.TransportStats;
pub const MessageType = transport.MessageType;
pub const MessageHeader = transport.MessageHeader;
pub const PeerConnection = transport.PeerConnection;
pub const RpcSerializer = transport.RpcSerializer;
pub const parseAddress = transport.parseAddress;

// Raft Transport exports
pub const RaftTransport = raft_transport.RaftTransport;
pub const RaftTransportConfig = raft_transport.RaftTransportConfig;
pub const RaftTransportStats = raft_transport.RaftTransport.RaftTransportStats;
pub const PeerAddress = raft_transport.PeerAddress;

// Circuit Breaker exports
pub const CircuitBreaker = circuit_breaker.CircuitBreaker;
pub const CircuitConfig = circuit_breaker.CircuitConfig;
pub const CircuitState = circuit_breaker.CircuitState;
pub const CircuitRegistry = circuit_breaker.CircuitRegistry;
pub const CircuitStats = circuit_breaker.CircuitStats;
pub const CircuitMetrics = circuit_breaker.CircuitMetrics;
pub const CircuitMetricEntry = circuit_breaker.CircuitMetricEntry;
pub const NetworkOperationError = circuit_breaker.NetworkOperationError;
pub const AggregateStats = circuit_breaker.AggregateStats;

// Failover Manager exports
pub const failover = @import("failover.zig");
pub const FailoverManager = failover.FailoverManager;
pub const FailoverConfig = failover.FailoverConfig;
pub const FailoverState = failover.FailoverState;
pub const FailoverEvent = failover.FailoverEvent;

pub const NetworkError = error{
    NetworkDisabled,
    NotInitialized,
};

const DEFAULT_CLUSTER_ID = "default";
const DEFAULT_HEARTBEAT_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_MAX_NODES: usize = 256;

pub const NetworkConfig = struct {
    cluster_id: []const u8 = DEFAULT_CLUSTER_ID,
    heartbeat_timeout_ms: u64 = DEFAULT_HEARTBEAT_TIMEOUT_MS,
    max_nodes: usize = DEFAULT_MAX_NODES,
};

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

var state_mutex = std.Thread.Mutex{};
var default_state: ?NetworkState = null;
var initialized: bool = false;

pub fn isEnabled() bool {
    return build_options.enable_network;
}

pub fn isInitialized() bool {
    state_mutex.lock();
    defer state_mutex.unlock();
    return initialized;
}

pub fn init(allocator: std.mem.Allocator) !void {
    return initWithConfig(allocator, .{});
}

pub fn initWithConfig(allocator: std.mem.Allocator, config: NetworkConfig) !void {
    if (!isEnabled()) return NetworkError.NetworkDisabled;

    state_mutex.lock();
    defer state_mutex.unlock();

    if (default_state == null) {
        default_state = try NetworkState.init(allocator, config);
    }
    initialized = true;
}

pub fn deinit() void {
    state_mutex.lock();
    defer state_mutex.unlock();

    if (default_state) |*state| {
        state.deinit();
        default_state = null;
    }
    initialized = false;
}

pub fn defaultRegistry() NetworkError!*NodeRegistry {
    state_mutex.lock();
    defer state_mutex.unlock();

    if (default_state) |*state| {
        return &state.registry;
    }
    return NetworkError.NotInitialized;
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
