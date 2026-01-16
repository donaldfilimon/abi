//! Stub for Network feature when disabled.
//!
//! Mirrors the full API of mod.zig, returning error.NetworkDisabled for all operations.

const std = @import("std");

pub const NetworkError = error{
    NetworkDisabled,
    NotInitialized,
    ConnectionFailed,
    Timeout,
};

// Core Network Types
pub const NetworkConfig = struct {
    cluster_id: []const u8 = "default",
    heartbeat_timeout_ms: u64 = 30_000,
    max_nodes: usize = 256,
};

pub const NetworkState = struct {
    allocator: std.mem.Allocator,
    config: NetworkConfig,
    registry: NodeRegistry,

    pub fn init(allocator: std.mem.Allocator, config: NetworkConfig) NetworkError!@This() {
        _ = allocator;
        _ = config;
        return error.NetworkDisabled;
    }

    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const NodeStatus = enum {
    healthy,
    degraded,
    offline,
};

pub const NodeInfo = struct {
    id: []const u8 = "",
    address: []const u8 = "",
    status: NodeStatus = .healthy,
    last_seen_ms: i64 = 0,
};

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

    pub fn list(self: *@This()) []const NodeInfo {
        _ = self;
        return &.{};
    }
};

// Protocol Types
pub const TaskEnvelope = struct {
    task_id: u64 = 0,
    payload: []const u8 = &.{},
    priority: TaskPriority = .normal,
};

pub const ResultEnvelope = struct {
    task_id: u64 = 0,
    status: ResultStatus = .success,
    payload: []const u8 = &.{},
    error_message: ?[]const u8 = null,
};

pub const ResultStatus = enum {
    success,
    failure,
    timeout,
    cancelled,
};

pub fn encodeTask(allocator: std.mem.Allocator, envelope: TaskEnvelope) NetworkError![]u8 {
    _ = allocator;
    _ = envelope;
    return error.NetworkDisabled;
}

pub fn decodeTask(allocator: std.mem.Allocator, data: []const u8) NetworkError!TaskEnvelope {
    _ = allocator;
    _ = data;
    return error.NetworkDisabled;
}

pub fn encodeResult(allocator: std.mem.Allocator, envelope: ResultEnvelope) NetworkError![]u8 {
    _ = allocator;
    _ = envelope;
    return error.NetworkDisabled;
}

pub fn decodeResult(allocator: std.mem.Allocator, data: []const u8) NetworkError!ResultEnvelope {
    _ = allocator;
    _ = data;
    return error.NetworkDisabled;
}

// Scheduler Types
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
    max_concurrent: usize = 100,
    timeout_ms: u64 = 30_000,
    strategy: LoadBalancingStrategy = .round_robin,
};

pub const SchedulerError = error{
    NetworkDisabled,
    NoNodesAvailable,
    TaskTimeout,
    SchedulingFailed,
};

pub const TaskPriority = enum {
    low,
    normal,
    high,
    critical,
};

pub const TaskState = enum {
    pending,
    running,
    completed,
    failed,
    cancelled,
};

pub const ComputeNode = struct {
    id: []const u8 = "",
    address: []const u8 = "",
    capacity: u32 = 0,
    current_load: u32 = 0,
};

pub const LoadBalancingStrategy = enum {
    round_robin,
    least_loaded,
    random,
    weighted,
};

pub const SchedulerStats = struct {
    total_tasks: u64 = 0,
    completed_tasks: u64 = 0,
    failed_tasks: u64 = 0,
    avg_latency_ms: f64 = 0.0,
};

// HA Types
pub const HealthCheck = struct {
    pub fn init(allocator: std.mem.Allocator) @This() {
        _ = allocator;
        return .{};
    }
    pub fn deinit(self: *@This()) void {
        _ = self;
    }
    pub fn check(self: *@This(), node_id: []const u8) NetworkError!HealthCheckResult {
        _ = self;
        _ = node_id;
        return error.NetworkDisabled;
    }
};

pub const ClusterConfig = struct {
    min_nodes: usize = 1,
    quorum_size: usize = 1,
    failover_policy: FailoverPolicy = .automatic,
};

pub const HaError = error{
    NetworkDisabled,
    QuorumLost,
    FailoverFailed,
    NodeUnreachable,
};

pub const NodeHealth = enum {
    healthy,
    unhealthy,
    unknown,
};

pub const ClusterState = enum {
    healthy,
    degraded,
    critical,
    offline,
};

pub const HealthCheckResult = struct {
    node_id: []const u8 = "",
    health: NodeHealth = .unknown,
    latency_ms: u64 = 0,
    error_message: ?[]const u8 = null,
};

pub const FailoverPolicy = enum {
    manual,
    automatic,
    disabled,
};

// Service Discovery Types
pub const ServiceDiscovery = struct {
    pub fn init(allocator: std.mem.Allocator, config: DiscoveryConfig) @This() {
        _ = allocator;
        _ = config;
        return .{};
    }
    pub fn deinit(self: *@This()) void {
        _ = self;
    }
    pub fn register(self: *@This(), instance: ServiceInstance) NetworkError!void {
        _ = self;
        _ = instance;
        return error.NetworkDisabled;
    }
    pub fn discover(self: *@This(), service_name: []const u8) NetworkError![]ServiceInstance {
        _ = self;
        _ = service_name;
        return error.NetworkDisabled;
    }
};

pub const DiscoveryConfig = struct {
    backend: DiscoveryBackend = .manual,
    refresh_interval_ms: u64 = 30_000,
};

pub const DiscoveryBackend = enum {
    manual,
    consul,
    kubernetes,
    dns,
};

pub const ServiceInstance = struct {
    id: []const u8 = "",
    service_name: []const u8 = "",
    address: []const u8 = "",
    port: u16 = 0,
    status: ServiceStatus = .unknown,
};

pub const ServiceStatus = enum {
    healthy,
    unhealthy,
    unknown,
};

pub const DiscoveryError = error{
    NetworkDisabled,
    ServiceNotFound,
    RegistrationFailed,
};

// Load Balancer Types
pub const LoadBalancer = struct {
    pub fn init(allocator: std.mem.Allocator, config: LoadBalancerConfig) @This() {
        _ = allocator;
        _ = config;
        return .{};
    }
    pub fn deinit(self: *@This()) void {
        _ = self;
    }
    pub fn selectNode(self: *@This()) ?*NodeStats {
        _ = self;
        return null;
    }
};

pub const LoadBalancerConfig = struct {
    strategy: LoadBalancerStrategy = .round_robin,
    health_check_interval_ms: u64 = 10_000,
};

pub const LoadBalancerStrategy = enum {
    round_robin,
    least_connections,
    weighted_round_robin,
    ip_hash,
};

pub const LoadBalancerError = error{
    NetworkDisabled,
    NoHealthyNodes,
    SelectionFailed,
};

pub const NodeState = enum {
    active,
    draining,
    inactive,
};

pub const NodeStats = struct {
    id: []const u8 = "",
    connections: u32 = 0,
    requests_per_sec: f64 = 0.0,
    state: NodeState = .inactive,
};

// Retry Types
pub const retry = struct {
    pub const RetryConfig = @import("stub.zig").RetryConfig;
    pub const RetryResult = @import("stub.zig").RetryResult;
    pub const RetryError = @import("stub.zig").RetryError;
    pub const RetryStrategy = @import("stub.zig").RetryStrategy;
    pub const RetryExecutor = @import("stub.zig").RetryExecutor;
    pub const RetryableErrors = @import("stub.zig").RetryableErrors;
    pub const BackoffCalculator = @import("stub.zig").BackoffCalculator;
    pub const retry_fn = retryOperation;
    pub const retryWithStrategy = retryWithStrategyFn;
};

pub const RetryConfig = struct {
    max_attempts: u32 = 3,
    initial_delay_ms: u64 = 100,
    max_delay_ms: u64 = 10_000,
    multiplier: f64 = 2.0,
    strategy: RetryStrategy = .exponential_backoff,
};

pub const RetryResult = struct {
    success: bool = false,
    attempts: u32 = 0,
    total_delay_ms: u64 = 0,
    last_error: ?[]const u8 = null,
};

pub const RetryError = error{
    NetworkDisabled,
    MaxAttemptsReached,
    NonRetryableError,
};

pub const RetryStrategy = enum {
    fixed_delay,
    exponential_backoff,
    linear_backoff,
    jittered_backoff,
};

pub const RetryExecutor = struct {
    pub fn init(config: RetryConfig) @This() {
        _ = config;
        return .{};
    }
};

pub const RetryableErrors = struct {
    errors: []const anyerror = &.{},
};

pub const BackoffCalculator = struct {
    pub fn calculate(attempt: u32, config: RetryConfig) u64 {
        _ = attempt;
        _ = config;
        return 0;
    }
};

pub fn retryOperation(comptime T: type, operation: anytype, config: RetryConfig) RetryError!T {
    _ = operation;
    _ = config;
    return error.NetworkDisabled;
}

pub fn retryWithStrategyFn(comptime T: type, operation: anytype, strategy: RetryStrategy) RetryError!T {
    _ = operation;
    _ = strategy;
    return error.NetworkDisabled;
}

// Rate Limiter Types
pub const rate_limiter = struct {
    pub const RateLimiter = @import("stub.zig").RateLimiter;
    pub const RateLimiterConfig = @import("stub.zig").RateLimiterConfig;
    pub const RateLimitAlgorithm = @import("stub.zig").RateLimitAlgorithm;
    pub const AcquireResult = @import("stub.zig").AcquireResult;
    pub const TokenBucketLimiter = @import("stub.zig").TokenBucketLimiter;
    pub const SlidingWindowLimiter = @import("stub.zig").SlidingWindowLimiter;
    pub const FixedWindowLimiter = @import("stub.zig").FixedWindowLimiter;
    pub const LimiterStats = @import("stub.zig").LimiterStats;
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
    pub fn acquire(self: *@This()) AcquireResult {
        _ = self;
        return .{ .allowed = false };
    }
};

pub const RateLimiterConfig = struct {
    algorithm: RateLimitAlgorithm = .token_bucket,
    rate: f64 = 100.0,
    burst: u32 = 10,
    window_ms: u64 = 1000,
};

pub const RateLimitAlgorithm = enum {
    token_bucket,
    sliding_window,
    fixed_window,
    leaky_bucket,
};

pub const AcquireResult = struct {
    allowed: bool = false,
    wait_time_ms: u64 = 0,
    remaining: u32 = 0,
};

pub const TokenBucketLimiter = struct {
    pub fn init(rate: f64, burst: u32) @This() {
        _ = rate;
        _ = burst;
        return .{};
    }
};

pub const SlidingWindowLimiter = struct {
    pub fn init(allocator: std.mem.Allocator, window_ms: u64, max_requests: u32) @This() {
        _ = allocator;
        _ = window_ms;
        _ = max_requests;
        return .{};
    }
    pub fn deinit(self: *@This()) void {
        _ = self;
    }
};

pub const FixedWindowLimiter = struct {
    pub fn init(window_ms: u64, max_requests: u32) @This() {
        _ = window_ms;
        _ = max_requests;
        return .{};
    }
};

pub const LimiterStats = struct {
    total_requests: u64 = 0,
    allowed_requests: u64 = 0,
    rejected_requests: u64 = 0,
};

// Connection Pool Types
pub const connection_pool = struct {
    pub const ConnectionPool = @import("stub.zig").ConnectionPool;
    pub const ConnectionPoolConfig = @import("stub.zig").ConnectionPoolConfig;
    pub const PooledConnection = @import("stub.zig").PooledConnection;
    pub const ConnectionState = @import("stub.zig").ConnectionState;
    pub const ConnectionStats = @import("stub.zig").ConnectionStats;
    pub const HostKey = @import("stub.zig").HostKey;
    pub const PoolStats = @import("stub.zig").PoolStats;
    pub const PoolBuilder = @import("stub.zig").PoolBuilder;
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
    pub fn acquire(self: *@This(), host: HostKey) NetworkError!PooledConnection {
        _ = self;
        _ = host;
        return error.NetworkDisabled;
    }
    pub fn release(self: *@This(), conn: PooledConnection) void {
        _ = self;
        _ = conn;
    }
};

pub const ConnectionPoolConfig = struct {
    max_connections_per_host: u32 = 10,
    idle_timeout_ms: u64 = 60_000,
    connect_timeout_ms: u64 = 5_000,
};

pub const PooledConnection = struct {
    id: u64 = 0,
    host: HostKey = .{},
    state: ConnectionState = .idle,
};

pub const ConnectionState = enum {
    idle,
    in_use,
    closing,
    closed,
};

pub const ConnectionStats = struct {
    created: u64 = 0,
    reused: u64 = 0,
    closed: u64 = 0,
};

pub const HostKey = struct {
    host: []const u8 = "",
    port: u16 = 0,
};

pub const PoolStats = struct {
    total_connections: u32 = 0,
    active_connections: u32 = 0,
    idle_connections: u32 = 0,
};

pub const PoolBuilder = struct {
    pub fn init() @This() {
        return .{};
    }
    pub fn maxConnectionsPerHost(self: *@This(), max: u32) *@This() {
        _ = max;
        return self;
    }
    pub fn build(self: *@This(), allocator: std.mem.Allocator) ConnectionPool {
        _ = self;
        return ConnectionPool.init(allocator, .{});
    }
};

// Raft Consensus Types
pub const raft = struct {
    pub const RaftNode = @import("stub.zig").RaftNode;
    pub const RaftState = @import("stub.zig").RaftState;
    pub const RaftConfig = @import("stub.zig").RaftConfig;
    pub const RaftError = @import("stub.zig").RaftError;
    pub const RaftStats = @import("stub.zig").RaftStats;
    pub const LogEntry = @import("stub.zig").LogEntry;
    pub const RequestVoteRequest = @import("stub.zig").RequestVoteRequest;
    pub const RequestVoteResponse = @import("stub.zig").RequestVoteResponse;
    pub const AppendEntriesRequest = @import("stub.zig").AppendEntriesRequest;
    pub const AppendEntriesResponse = @import("stub.zig").AppendEntriesResponse;
    pub const PeerState = @import("stub.zig").PeerState;
    pub const createCluster = createRaftCluster;
};

pub const RaftNode = struct {
    pub fn init(allocator: std.mem.Allocator, node_id: []const u8, config: RaftConfig) NetworkError!@This() {
        _ = allocator;
        _ = node_id;
        _ = config;
        return error.NetworkDisabled;
    }
    pub fn deinit(self: *@This()) void {
        _ = self;
    }
    pub fn addPeer(self: *@This(), peer_id: []const u8) NetworkError!void {
        _ = self;
        _ = peer_id;
        return error.NetworkDisabled;
    }
    pub fn tick(self: *@This(), elapsed_ms: u64) NetworkError!void {
        _ = self;
        _ = elapsed_ms;
        return error.NetworkDisabled;
    }
    pub fn isLeader(self: *const @This()) bool {
        _ = self;
        return false;
    }
    pub fn appendCommand(self: *@This(), command: []const u8) NetworkError!u64 {
        _ = self;
        _ = command;
        return error.NetworkDisabled;
    }
};

pub const RaftState = enum {
    follower,
    candidate,
    leader,
};

pub const RaftConfig = struct {
    election_timeout_min_ms: u64 = 150,
    election_timeout_max_ms: u64 = 300,
    heartbeat_interval_ms: u64 = 50,
};

pub const RaftError = error{
    NetworkDisabled,
    NotLeader,
    ElectionFailed,
    LogInconsistent,
};

pub const RaftStats = struct {
    current_term: u64 = 0,
    commit_index: u64 = 0,
    last_applied: u64 = 0,
    state: RaftState = .follower,
};

pub const LogEntry = struct {
    term: u64 = 0,
    index: u64 = 0,
    command: []const u8 = &.{},
};

pub const RequestVoteRequest = struct {
    term: u64 = 0,
    candidate_id: []const u8 = "",
    last_log_index: u64 = 0,
    last_log_term: u64 = 0,
};

pub const RequestVoteResponse = struct {
    term: u64 = 0,
    vote_granted: bool = false,
};

pub const AppendEntriesRequest = struct {
    term: u64 = 0,
    leader_id: []const u8 = "",
    prev_log_index: u64 = 0,
    prev_log_term: u64 = 0,
    entries: []const LogEntry = &.{},
    leader_commit: u64 = 0,
};

pub const AppendEntriesResponse = struct {
    term: u64 = 0,
    success: bool = false,
    match_index: u64 = 0,
};

pub const PeerState = struct {
    id: []const u8 = "",
    next_index: u64 = 0,
    match_index: u64 = 0,
    vote_granted: bool = false,
};

pub fn createRaftCluster(allocator: std.mem.Allocator, node_ids: []const []const u8, config: RaftConfig) NetworkError![]RaftNode {
    _ = allocator;
    _ = node_ids;
    _ = config;
    return error.NetworkDisabled;
}

// Module Lifecycle
var initialized: bool = false;

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return initialized;
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

pub fn deinit() void {
    initialized = false;
}

pub fn defaultRegistry() NetworkError!*NodeRegistry {
    return error.NotInitialized;
}

pub fn defaultConfig() ?NetworkConfig {
    return null;
}
