//! Stub for Network feature when disabled.
//!
//! Mirrors the full API of mod.zig, returning error.NetworkDisabled for all operations.

const std = @import("std");

pub const NetworkError = error{
    NetworkDisabled,
    NotInitialized,
};

// Node Registry
pub const NodeRegistry = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) @This() {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }

    pub fn register(self: *@This(), id: []const u8, address: []const u8) NetworkError!void {
        _ = self;
        _ = id;
        _ = address;
        return error.NetworkDisabled;
    }

    pub fn unregister(self: *@This(), id: []const u8) bool {
        _ = self;
        _ = id;
        return false;
    }

    pub fn touch(self: *@This(), id: []const u8) bool {
        _ = self;
        _ = id;
        return false;
    }

    pub fn setStatus(self: *@This(), id: []const u8, status: NodeStatus) bool {
        _ = self;
        _ = id;
        _ = status;
        return false;
    }

    pub fn list(self: *@This()) []NodeInfo {
        _ = self;
        return &.{};
    }
};

pub const NodeInfo = struct {
    id: []const u8 = "",
    address: []const u8 = "",
    status: NodeStatus = .healthy,
    last_seen_ms: i64 = 0,
};

pub const NodeStatus = enum {
    healthy,
    degraded,
    offline,
};

// Protocol
pub const TaskEnvelope = struct {
    id: []const u8 = "",
    payload: []const u8 = "",
};

pub const ResultEnvelope = struct {
    task_id: []const u8 = "",
    status: ResultStatus = .pending,
    payload: ?[]const u8 = null,
};

pub const ResultStatus = enum {
    pending,
    completed,
    failed,
};

pub fn encodeTask(allocator: std.mem.Allocator, task: TaskEnvelope) NetworkError![]u8 {
    _ = allocator;
    _ = task;
    return error.NetworkDisabled;
}

pub fn decodeTask(allocator: std.mem.Allocator, data: []const u8) NetworkError!TaskEnvelope {
    _ = allocator;
    _ = data;
    return error.NetworkDisabled;
}

pub fn encodeResult(allocator: std.mem.Allocator, result: ResultEnvelope) NetworkError![]u8 {
    _ = allocator;
    _ = result;
    return error.NetworkDisabled;
}

pub fn decodeResult(allocator: std.mem.Allocator, data: []const u8) NetworkError!ResultEnvelope {
    _ = allocator;
    _ = data;
    return error.NetworkDisabled;
}

// Scheduler
pub const TaskScheduler = struct {
    pub fn init(allocator: std.mem.Allocator, config: SchedulerConfig) NetworkError!@This() {
        _ = allocator;
        _ = config;
        return error.NetworkDisabled;
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const SchedulerConfig = struct {
    max_concurrent_tasks: u32 = 100,
    strategy: LoadBalancingStrategy = .round_robin,
};

pub const SchedulerError = error{
    NetworkDisabled,
    NoAvailableNodes,
    TaskNotFound,
};

pub const TaskPriority = enum { low, normal, high, critical };
pub const TaskState = enum { pending, running, completed, failed };

pub const ComputeNode = struct {
    id: []const u8 = "",
    address: []const u8 = "",
    capacity: u32 = 0,
};

pub const LoadBalancingStrategy = enum {
    round_robin,
    least_loaded,
    random,
    weighted,
};

pub const SchedulerStats = struct {
    tasks_scheduled: u64 = 0,
    tasks_completed: u64 = 0,
    tasks_failed: u64 = 0,
};

// High Availability
pub const HealthCheck = struct {
    pub fn init(allocator: std.mem.Allocator, config: ClusterConfig) @This() {
        _ = allocator;
        _ = config;
        return .{};
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const ClusterConfig = struct {
    heartbeat_interval_ms: u64 = 5000,
    failure_threshold: u32 = 3,
};

pub const HaError = error{
    NetworkDisabled,
    NodeNotFound,
};

pub const NodeHealth = enum { healthy, degraded, unhealthy };
pub const ClusterState = enum { healthy, degraded, critical };

pub const HealthCheckResult = struct {
    node_id: []const u8 = "",
    health: NodeHealth = .healthy,
    latency_ms: u64 = 0,
};

pub const FailoverPolicy = enum { automatic, manual, disabled };

// Service Discovery
pub const ServiceDiscovery = struct {
    pub fn init(allocator: std.mem.Allocator, config: DiscoveryConfig) NetworkError!@This() {
        _ = allocator;
        _ = config;
        return error.NetworkDisabled;
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const DiscoveryConfig = struct {
    backend: DiscoveryBackend = .static,
    refresh_interval_ms: u64 = 30_000,
};

pub const DiscoveryBackend = enum { static, dns, consul, etcd };

pub const ServiceInstance = struct {
    id: []const u8 = "",
    name: []const u8 = "",
    address: []const u8 = "",
    port: u16 = 0,
    status: ServiceStatus = .unknown,
};

pub const ServiceStatus = enum { unknown, up, down, starting };
pub const DiscoveryError = error{ NetworkDisabled, ServiceNotFound };

// Load Balancer
pub const LoadBalancer = struct {
    pub fn init(allocator: std.mem.Allocator, config: LoadBalancerConfig) @This() {
        _ = allocator;
        _ = config;
        return .{};
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const LoadBalancerConfig = struct {
    strategy: LoadBalancerStrategy = .round_robin,
    health_check_interval_ms: u64 = 10_000,
};

pub const LoadBalancerStrategy = enum { round_robin, least_connections, weighted, ip_hash };
pub const LoadBalancerError = error{ NetworkDisabled, NoHealthyNodes };
pub const NodeState = enum { active, draining, inactive };

pub const NodeStats = struct {
    connections: u32 = 0,
    requests: u64 = 0,
    errors: u64 = 0,
};

// Retry namespace
pub const retry = struct {
    pub const RetryConfig = stub_root.RetryConfig;
    pub const RetryResult = stub_root.RetryResult;
    pub const RetryError = stub_root.RetryError;
    pub const RetryStrategy = stub_root.RetryStrategy;
    pub const RetryExecutor = stub_root.RetryExecutor;
    pub const RetryableErrors = stub_root.RetryableErrors;
    pub const BackoffCalculator = stub_root.BackoffCalculator;
    pub const retryFunc = stub_root.retryOperation;
    pub const retryWithStrategy = stub_root.retryWithStrategyFn;
};

const stub_root = @This();

pub const RetryConfig = struct {
    max_retries: u32 = 3,
    base_delay_ms: u64 = 100,
    max_delay_ms: u64 = 10_000,
    strategy: RetryStrategy = .exponential,
};

pub const RetryResult = union(enum) {
    success: void,
    failure: RetryError,
};

pub const RetryError = error{ NetworkDisabled, MaxRetriesExceeded };
pub const RetryStrategy = enum { constant, linear, exponential, decorrelated_jitter };

pub const RetryExecutor = struct {
    pub fn init(config: RetryConfig) @This() {
        _ = config;
        return .{};
    }
};

pub const RetryableErrors = struct {};
pub const BackoffCalculator = struct {};

pub fn retryOperation(config: RetryConfig, operation: anytype) RetryError!void {
    _ = config;
    _ = operation;
    return error.NetworkDisabled;
}

pub fn retryWithStrategyFn(strategy: RetryStrategy, operation: anytype) RetryError!void {
    _ = strategy;
    _ = operation;
    return error.NetworkDisabled;
}

// Rate Limiter namespace
pub const rate_limiter = struct {
    pub const RateLimiter = stub_root.RateLimiter;
    pub const RateLimiterConfig = stub_root.RateLimiterConfig;
    pub const RateLimitAlgorithm = stub_root.RateLimitAlgorithm;
    pub const AcquireResult = stub_root.AcquireResult;
    pub const TokenBucketLimiter = stub_root.TokenBucketLimiter;
    pub const SlidingWindowLimiter = stub_root.SlidingWindowLimiter;
    pub const FixedWindowLimiter = stub_root.FixedWindowLimiter;
    pub const LimiterStats = stub_root.LimiterStats;
};

pub const RateLimiter = struct {
    pub fn init(allocator: std.mem.Allocator, config: RateLimiterConfig) @This() {
        _ = allocator;
        _ = config;
        return .{};
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const RateLimiterConfig = struct {
    requests_per_second: u32 = 100,
    burst_size: u32 = 10,
    algorithm: RateLimitAlgorithm = .token_bucket,
};

pub const RateLimitAlgorithm = enum { token_bucket, sliding_window, fixed_window };
pub const AcquireResult = enum { acquired, rejected, queued };
pub const TokenBucketLimiter = struct {};
pub const SlidingWindowLimiter = struct {};
pub const FixedWindowLimiter = struct {};
pub const LimiterStats = struct { requests: u64 = 0, rejected: u64 = 0 };

// Connection Pool namespace
pub const connection_pool = struct {
    pub const ConnectionPool = stub_root.ConnectionPool;
    pub const ConnectionPoolConfig = stub_root.ConnectionPoolConfig;
    pub const PooledConnection = stub_root.PooledConnection;
    pub const ConnectionState = stub_root.ConnectionState;
    pub const ConnectionStats = stub_root.ConnectionStats;
    pub const HostKey = stub_root.HostKey;
    pub const PoolStats = stub_root.PoolStats;
    pub const PoolBuilder = stub_root.PoolBuilder;
};

pub const ConnectionPool = struct {
    pub fn init(allocator: std.mem.Allocator, config: ConnectionPoolConfig) @This() {
        _ = allocator;
        _ = config;
        return .{};
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const ConnectionPoolConfig = struct {
    min_connections: u32 = 1,
    max_connections: u32 = 10,
    idle_timeout_ms: u64 = 60_000,
};

pub const PooledConnection = struct {};
pub const ConnectionState = enum { idle, active, closed };
pub const ConnectionStats = struct { active: u32 = 0, idle: u32 = 0 };
pub const HostKey = struct { host: []const u8 = "", port: u16 = 0 };
pub const PoolStats = struct { total: u32 = 0, available: u32 = 0 };
pub const PoolBuilder = struct {};

// Raft namespace
pub const raft = struct {
    pub const RaftNode = stub_root.RaftNode;
    pub const RaftState = stub_root.RaftState;
    pub const RaftConfig = stub_root.RaftConfig;
    pub const RaftError = stub_root.RaftError;
    pub const RaftStats = stub_root.RaftStats;
    pub const LogEntry = stub_root.LogEntry;
    pub const RequestVoteRequest = stub_root.RequestVoteRequest;
    pub const RequestVoteResponse = stub_root.RequestVoteResponse;
    pub const AppendEntriesRequest = stub_root.AppendEntriesRequest;
    pub const AppendEntriesResponse = stub_root.AppendEntriesResponse;
    pub const PeerState = stub_root.PeerState;
    pub const createCluster = stub_root.createRaftCluster;
    pub const RaftPersistence = stub_root.RaftPersistence;
    pub const PersistentState = stub_root.PersistentState;
    pub const RaftSnapshotManager = stub_root.RaftSnapshotManager;
    pub const SnapshotConfig = stub_root.SnapshotConfig;
    pub const SnapshotMetadata = stub_root.SnapshotMetadata;
    pub const SnapshotInfo = stub_root.SnapshotInfo;
    pub const InstallSnapshotRequest = stub_root.InstallSnapshotRequest;
    pub const InstallSnapshotResponse = stub_root.InstallSnapshotResponse;
    pub const ConfigChangeType = stub_root.ConfigChangeType;
    pub const ConfigChangeRequest = stub_root.ConfigChangeRequest;
    pub const applyConfigChange = stub_root.applyConfigChangeFn;
};

pub const RaftNode = struct {
    pub fn init(allocator: std.mem.Allocator, config: RaftConfig) NetworkError!@This() {
        _ = allocator;
        _ = config;
        return error.NetworkDisabled;
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const RaftState = enum { follower, candidate, leader };

pub const RaftConfig = struct {
    node_id: []const u8 = "",
    election_timeout_ms: u64 = 150,
    heartbeat_interval_ms: u64 = 50,
};

pub const RaftError = error{ NetworkDisabled, NotLeader, LogInconsistency };
pub const RaftStats = struct { term: u64 = 0, commit_index: u64 = 0 };
pub const LogEntry = struct { term: u64 = 0, index: u64 = 0, data: []const u8 = "" };

pub const RequestVoteRequest = struct { term: u64 = 0, candidate_id: []const u8 = "" };
pub const RequestVoteResponse = struct { term: u64 = 0, vote_granted: bool = false };
pub const AppendEntriesRequest = struct { term: u64 = 0, leader_id: []const u8 = "" };
pub const AppendEntriesResponse = struct { term: u64 = 0, success: bool = false };
pub const PeerState = struct { id: []const u8 = "", next_index: u64 = 0 };

pub fn createRaftCluster(allocator: std.mem.Allocator, configs: []const RaftConfig) NetworkError![]RaftNode {
    _ = allocator;
    _ = configs;
    return error.NetworkDisabled;
}

pub const RaftPersistence = struct {};
pub const PersistentState = struct { term: u64 = 0, voted_for: ?[]const u8 = null };

pub const RaftSnapshotManager = struct {};
pub const SnapshotConfig = struct { threshold: u64 = 10000 };
pub const SnapshotMetadata = struct { last_included_index: u64 = 0, last_included_term: u64 = 0 };
pub const SnapshotInfo = struct { metadata: SnapshotMetadata = .{}, size: u64 = 0 };
pub const InstallSnapshotRequest = struct { term: u64 = 0, leader_id: []const u8 = "" };
pub const InstallSnapshotResponse = struct { term: u64 = 0 };

pub const ConfigChangeType = enum { add_node, remove_node };
pub const ConfigChangeRequest = struct { change_type: ConfigChangeType = .add_node, node_id: []const u8 = "" };

pub fn applyConfigChangeFn(node: *RaftNode, request: ConfigChangeRequest) NetworkError!void {
    _ = node;
    _ = request;
    return error.NetworkDisabled;
}

// Transport namespace
pub const transport = struct {
    pub const TcpTransport = stub_root.TcpTransport;
    pub const TransportConfig = stub_root.TransportConfig;
    pub const TransportError = stub_root.TransportError;
    pub const MessageType = stub_root.MessageType;
    pub const MessageHeader = stub_root.MessageHeader;
    pub const PeerConnection = stub_root.PeerConnection;
    pub const RpcSerializer = stub_root.RpcSerializer;
    pub const parseAddress = stub_root.parseAddressFn;
};

pub const TcpTransport = struct {
    pub const TransportStats = struct { bytes_sent: u64 = 0, bytes_received: u64 = 0 };

    pub fn init(allocator: std.mem.Allocator, config: TransportConfig) NetworkError!@This() {
        _ = allocator;
        _ = config;
        return error.NetworkDisabled;
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const TransportConfig = struct {
    bind_address: []const u8 = "0.0.0.0",
    bind_port: u16 = 0,
};

pub const TransportError = error{ NetworkDisabled, ConnectionFailed };
pub const TransportStats = TcpTransport.TransportStats;
pub const MessageType = enum { request, response, heartbeat };
pub const MessageHeader = struct { msg_type: MessageType = .request, length: u32 = 0 };
pub const PeerConnection = struct { address: []const u8 = "", connected: bool = false };
pub const RpcSerializer = struct {};

pub fn parseAddressFn(address: []const u8) ?struct { host: []const u8, port: u16 } {
    _ = address;
    return null;
}

pub const parseAddress = parseAddressFn;

// Raft Transport namespace
pub const raft_transport = struct {
    pub const RaftTransport = stub_root.RaftTransport;
    pub const RaftTransportConfig = stub_root.RaftTransportConfig;
    pub const PeerAddress = stub_root.PeerAddress;
};

pub const RaftTransport = struct {
    pub const RaftTransportStats = struct { messages_sent: u64 = 0, messages_received: u64 = 0 };

    pub fn init(allocator: std.mem.Allocator, config: RaftTransportConfig) NetworkError!@This() {
        _ = allocator;
        _ = config;
        return error.NetworkDisabled;
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const RaftTransportConfig = struct {
    bind_address: []const u8 = "0.0.0.0",
    bind_port: u16 = 0,
};

pub const RaftTransportStats = RaftTransport.RaftTransportStats;
pub const PeerAddress = struct { host: []const u8 = "", port: u16 = 0 };

// Circuit Breaker namespace
pub const circuit_breaker = struct {
    pub const CircuitBreaker = stub_root.CircuitBreaker;
    pub const CircuitConfig = stub_root.CircuitConfig;
    pub const CircuitState = stub_root.CircuitState;
    pub const CircuitRegistry = stub_root.CircuitRegistry;
    pub const CircuitStats = stub_root.CircuitStats;
    pub const CircuitMetrics = stub_root.CircuitMetrics;
    pub const CircuitMetricEntry = stub_root.CircuitMetricEntry;
    pub const NetworkOperationError = stub_root.NetworkOperationError;
    pub const AggregateStats = stub_root.AggregateStats;
};

pub const CircuitBreaker = struct {
    pub fn init(allocator: std.mem.Allocator, config: CircuitConfig) @This() {
        _ = allocator;
        _ = config;
        return .{};
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const CircuitConfig = struct {
    failure_threshold: u32 = 5,
    reset_timeout_ms: u64 = 30_000,
};

pub const CircuitState = enum { closed, open, half_open };

pub const CircuitRegistry = struct {
    pub fn init(allocator: std.mem.Allocator) @This() {
        _ = allocator;
        return .{};
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const CircuitStats = struct { failures: u32 = 0, successes: u32 = 0 };
pub const CircuitMetrics = struct {};
pub const CircuitMetricEntry = struct { timestamp: i64 = 0, success: bool = false };
pub const NetworkOperationError = error{ NetworkDisabled, CircuitOpen };
pub const AggregateStats = struct { total_calls: u64 = 0, total_failures: u64 = 0 };

// Network State and Config
pub const NetworkConfig = struct {
    cluster_id: []const u8 = "default",
    heartbeat_timeout_ms: u64 = 30_000,
    max_nodes: usize = 256,
};

pub const NetworkState = struct {
    allocator: std.mem.Allocator,
    config: NetworkConfig,
    registry: NodeRegistry,

    pub fn init(allocator: std.mem.Allocator, config: NetworkConfig) NetworkError!NetworkState {
        _ = allocator;
        _ = config;
        return error.NetworkDisabled;
    }

    pub fn deinit(self: *NetworkState) void {
        _ = self;
    }
};

// Module-level functions
pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

pub fn init(allocator: std.mem.Allocator) NetworkError!void {
    _ = allocator;
    return error.NetworkDisabled;
}

pub fn initWithConfig(allocator: std.mem.Allocator, config: NetworkConfig) NetworkError!void {
    _ = allocator;
    _ = config;
    return error.NetworkDisabled;
}

pub fn deinit() void {}

pub fn defaultRegistry() NetworkError!*NodeRegistry {
    return error.NetworkDisabled;
}

pub fn defaultConfig() ?NetworkConfig {
    return null;
}
