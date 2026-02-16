//! Network Stub Module
//!
//! API-compatible no-op implementations when network is disabled.
//! Build with `-Denable-network=true` for the real implementation.

const std = @import("std");
const config_module = @import("../../core/config/mod.zig");

// ============================================================================
// Local Stubs Imports
// ============================================================================

const types = @import("stubs/types.zig");
const protocol_mod = @import("stubs/protocol.zig");
const scheduler_mod = @import("stubs/scheduler.zig");
const ha_mod = @import("stubs/ha.zig");
const discovery_mod = @import("stubs/discovery.zig");
const loadbalancer_mod = @import("stubs/loadbalancer.zig");
const retry_mod = @import("stubs/retry.zig");
const rate_limiter_mod = @import("stubs/rate_limiter.zig");
const connection_pool_mod = @import("stubs/connection_pool.zig");
const raft_mod = @import("stubs/raft.zig");
const transport_mod = @import("stubs/transport.zig");
const raft_transport_mod = @import("stubs/raft_transport.zig");
const circuit_breaker_mod = @import("stubs/circuit_breaker.zig");
const failover_mod = @import("stubs/failover.zig");
const unified_memory_mod = @import("stubs/unified_memory.zig");
const linking_mod = @import("stubs/linking.zig");

// ============================================================================
// Core Types Re-exports
// ============================================================================

pub const Error = types.Error;
pub const NetworkError = error{ NetworkDisabled, NotInitialized };
pub const NetworkConfig = types.NetworkConfig;
pub const NetworkState = types.NetworkState;
pub const Node = types.Node;
pub const NodeStatus = types.NodeStatus;
pub const NodeInfo = types.NodeInfo;
pub const NodeRegistry = types.NodeRegistry;

// ============================================================================
// Protocol Re-exports
// ============================================================================

pub const TaskEnvelope = protocol_mod.TaskEnvelope;
pub const ResultEnvelope = protocol_mod.ResultEnvelope;
pub const ResultStatus = protocol_mod.ResultStatus;
pub const encodeTask = protocol_mod.encodeTask;
pub const decodeTask = protocol_mod.decodeTask;
pub const encodeResult = protocol_mod.encodeResult;
pub const decodeResult = protocol_mod.decodeResult;

// ============================================================================
// Scheduler Re-exports
// ============================================================================

pub const TaskScheduler = scheduler_mod.TaskScheduler;
pub const SchedulerConfig = scheduler_mod.SchedulerConfig;
pub const SchedulerError = scheduler_mod.SchedulerError;
pub const TaskPriority = scheduler_mod.TaskPriority;
pub const TaskState = scheduler_mod.TaskState;
pub const ComputeNode = scheduler_mod.ComputeNode;
pub const LoadBalancingStrategy = scheduler_mod.LoadBalancingStrategy;
pub const SchedulerStats = scheduler_mod.SchedulerStats;

// ============================================================================
// High Availability Re-exports
// ============================================================================

pub const HealthCheck = ha_mod.HealthCheck;
pub const ClusterConfig = ha_mod.ClusterConfig;
pub const HaError = ha_mod.HaError;
pub const NodeHealth = ha_mod.NodeHealth;
pub const ClusterState = ha_mod.ClusterState;
pub const HealthCheckResult = ha_mod.HealthCheckResult;
pub const FailoverPolicy = ha_mod.FailoverPolicy;

// ============================================================================
// Service Discovery Re-exports
// ============================================================================

pub const ServiceDiscovery = discovery_mod.ServiceDiscovery;
pub const DiscoveryConfig = discovery_mod.DiscoveryConfig;
pub const DiscoveryBackend = discovery_mod.DiscoveryBackend;
pub const ServiceInstance = discovery_mod.ServiceInstance;
pub const ServiceStatus = discovery_mod.ServiceStatus;
pub const DiscoveryError = discovery_mod.DiscoveryError;
pub const generateServiceId = discovery_mod.generateServiceId;
pub const base64Encode = discovery_mod.base64Encode;
pub const base64Decode = discovery_mod.base64Decode;

// ============================================================================
// Load Balancer Re-exports
// ============================================================================

pub const LoadBalancer = loadbalancer_mod.LoadBalancer;
pub const LoadBalancerConfig = loadbalancer_mod.LoadBalancerConfig;
pub const LoadBalancerStrategy = loadbalancer_mod.LoadBalancerStrategy;
pub const LoadBalancerError = loadbalancer_mod.LoadBalancerError;
pub const NodeState = loadbalancer_mod.NodeState;
pub const NodeStats = loadbalancer_mod.NodeStats;

// ============================================================================
// Retry Re-exports
// ============================================================================

pub const RetryConfig = retry_mod.RetryConfig;
pub const RetryResult = retry_mod.RetryResult;
pub const RetryError = retry_mod.RetryError;
pub const RetryStrategy = retry_mod.RetryStrategy;
pub const RetryExecutor = retry_mod.RetryExecutor;
pub const RetryableErrors = retry_mod.RetryableErrors;
pub const BackoffCalculator = retry_mod.BackoffCalculator;
pub const retryOperation = retry_mod.retry;
pub const retryWithStrategy = retry_mod.retryWithStrategy;

// ============================================================================
// Rate Limiter Re-exports
// ============================================================================

pub const RateLimiter = rate_limiter_mod.RateLimiter;
pub const RateLimiterConfig = rate_limiter_mod.RateLimiterConfig;
pub const RateLimitAlgorithm = rate_limiter_mod.RateLimitAlgorithm;
pub const AcquireResult = rate_limiter_mod.AcquireResult;
pub const TokenBucketLimiter = rate_limiter_mod.TokenBucketLimiter;
pub const SlidingWindowLimiter = rate_limiter_mod.SlidingWindowLimiter;
pub const FixedWindowLimiter = rate_limiter_mod.FixedWindowLimiter;
pub const LimiterStats = rate_limiter_mod.LimiterStats;

// ============================================================================
// Connection Pool Re-exports
// ============================================================================

pub const ConnectionPool = connection_pool_mod.ConnectionPool;
pub const ConnectionPoolConfig = connection_pool_mod.ConnectionPoolConfig;
pub const PooledConnection = connection_pool_mod.PooledConnection;
pub const ConnectionState = connection_pool_mod.ConnectionState;
pub const ConnectionStats = connection_pool_mod.ConnectionStats;
pub const HostKey = connection_pool_mod.HostKey;
pub const PoolStats = connection_pool_mod.PoolStats;
pub const PoolBuilder = connection_pool_mod.PoolBuilder;

// ============================================================================
// Raft Consensus Re-exports
// ============================================================================

pub const RaftNode = raft_mod.RaftNode;
pub const RaftState = raft_mod.RaftState;
pub const RaftConfig = raft_mod.RaftConfig;
pub const RaftError = raft_mod.RaftError;
pub const RaftStats = raft_mod.RaftStats;
pub const LogEntry = raft_mod.LogEntry;
pub const RequestVoteRequest = raft_mod.RequestVoteRequest;
pub const RequestVoteResponse = raft_mod.RequestVoteResponse;
pub const AppendEntriesRequest = raft_mod.AppendEntriesRequest;
pub const AppendEntriesResponse = raft_mod.AppendEntriesResponse;
pub const PeerState = raft_mod.PeerState;
pub const createRaftCluster = raft_mod.createCluster;
pub const RaftPersistence = raft_mod.RaftPersistence;
pub const PersistentState = raft_mod.PersistentState;
pub const RaftSnapshotManager = raft_mod.RaftSnapshotManager;
pub const SnapshotConfig = raft_mod.SnapshotConfig;
pub const SnapshotMetadata = raft_mod.SnapshotMetadata;
pub const SnapshotInfo = raft_mod.SnapshotInfo;
pub const InstallSnapshotRequest = raft_mod.InstallSnapshotRequest;
pub const InstallSnapshotResponse = raft_mod.InstallSnapshotResponse;
pub const ConfigChangeType = raft_mod.ConfigChangeType;
pub const ConfigChangeRequest = raft_mod.ConfigChangeRequest;
pub const applyConfigChange = raft_mod.applyConfigChange;

// ============================================================================
// Transport Re-exports
// ============================================================================

pub const TcpTransport = transport_mod.TcpTransport;
pub const TransportConfig = transport_mod.TransportConfig;
pub const TransportError = transport_mod.TransportError;
pub const TransportStats = transport_mod.TcpTransport.TransportStats;
pub const MessageType = transport_mod.MessageType;
pub const MessageHeader = transport_mod.MessageHeader;
pub const PeerConnection = transport_mod.PeerConnection;
pub const RpcSerializer = transport_mod.RpcSerializer;
pub const parseAddress = transport_mod.parseAddress;

// ============================================================================
// Raft Transport Re-exports
// ============================================================================

pub const RaftTransport = raft_transport_mod.RaftTransport;
pub const RaftTransportConfig = raft_transport_mod.RaftTransportConfig;
pub const RaftTransportStats = raft_transport_mod.RaftTransport.RaftTransportStats;
pub const PeerAddress = raft_transport_mod.PeerAddress;

// ============================================================================
// Circuit Breaker Re-exports
// ============================================================================

pub const CircuitBreaker = circuit_breaker_mod.CircuitBreaker;
pub const CircuitConfig = circuit_breaker_mod.CircuitConfig;
pub const CircuitState = circuit_breaker_mod.CircuitState;
pub const CircuitRegistry = circuit_breaker_mod.CircuitRegistry;
pub const CircuitStats = circuit_breaker_mod.CircuitStats;
pub const CircuitMetrics = circuit_breaker_mod.CircuitMetrics;
pub const CircuitMetricEntry = circuit_breaker_mod.CircuitMetricEntry;
pub const NetworkOperationError = circuit_breaker_mod.NetworkOperationError;
pub const AggregateStats = circuit_breaker_mod.AggregateStats;

// ============================================================================
// Failover Re-exports
// ============================================================================

pub const FailoverManager = failover_mod.FailoverManager;
pub const FailoverConfig = failover_mod.FailoverConfig;
pub const FailoverState = failover_mod.FailoverState;
pub const FailoverEvent = failover_mod.FailoverEvent;

// ============================================================================
// Unified Memory Re-exports
// ============================================================================

pub const UnifiedMemoryManager = unified_memory_mod.UnifiedMemoryManager;
pub const UnifiedMemoryConfig = unified_memory_mod.UnifiedMemoryConfig;
pub const UnifiedMemoryError = unified_memory_mod.UnifiedMemoryError;
pub const MemoryRegion = unified_memory_mod.MemoryRegion;
pub const RegionId = unified_memory_mod.RegionId;
pub const RegionFlags = unified_memory_mod.RegionFlags;
pub const RegionState = unified_memory_mod.RegionState;
pub const CoherenceProtocol = unified_memory_mod.CoherenceProtocol;
pub const CoherenceState = unified_memory_mod.CoherenceState;
pub const RemotePtr = unified_memory_mod.RemotePtr;
pub const RemoteSlice = unified_memory_mod.RemoteSlice;
pub const MemoryNode = unified_memory_mod.MemoryNode;

// ============================================================================
// Linking Re-exports
// ============================================================================

pub const LinkManager = linking_mod.LinkManager;
pub const Link = linking_mod.Link;
pub const LinkConfig = linking_mod.LinkConfig;
pub const LinkState = linking_mod.LinkState;
pub const LinkStats = linking_mod.LinkStats;
pub const TransportType = linking_mod.TransportType;
pub const SecureChannel = linking_mod.SecureChannel;
pub const ChannelConfig = linking_mod.ChannelConfig;
pub const EncryptionType = linking_mod.EncryptionType;
pub const ThunderboltTransport = linking_mod.ThunderboltTransport;
pub const ThunderboltConfig = linking_mod.ThunderboltConfig;
pub const ThunderboltDevice = linking_mod.ThunderboltDevice;
pub const InternetTransport = linking_mod.InternetTransport;
pub const InternetConfig = linking_mod.InternetConfig;
pub const NatTraversal = linking_mod.NatTraversal;
pub const QuicConnection = linking_mod.QuicConnection;

// ============================================================================
// Sub-module Namespace Re-exports
// ============================================================================

pub const retry = retry_mod;
pub const rate_limiter = rate_limiter_mod;
pub const connection_pool = connection_pool_mod;
pub const raft = raft_mod;
pub const transport = transport_mod;
pub const raft_transport = raft_transport_mod;
pub const circuit_breaker = circuit_breaker_mod;
pub const unified_memory = unified_memory_mod;
pub const linking = linking_mod;
pub const failover = failover_mod;

// ============================================================================
// Context (Framework integration)
// ============================================================================

pub const Context = struct {
    pub const State = enum { disconnected, connecting, connected, error_state };

    pub fn init(_: std.mem.Allocator, _: config_module.NetworkConfig) Error!*Context {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn connect(_: *Context) Error!void {
        return error.NetworkDisabled;
    }
    pub fn disconnect(_: *Context) void {}
    pub fn getState(_: *Context) State {
        return .disconnected;
    }
    pub fn discoverPeers(_: *Context) Error![]NodeInfo {
        return error.NetworkDisabled;
    }
    pub fn sendTask(_: *Context, _: []const u8, _: anytype) Error!void {
        return error.NetworkDisabled;
    }
};

// ============================================================================
// Module Lifecycle
// ============================================================================

pub fn isEnabled() bool {
    return false;
}
pub fn defaultRegistry() Error!*NodeRegistry {
    return error.NetworkDisabled;
}
pub fn defaultConfig() ?NetworkConfig {
    return null;
}
pub fn isInitialized() bool {
    return false;
}
pub fn init(_: std.mem.Allocator) Error!void {
    return error.NetworkDisabled;
}
pub fn initWithConfig(_: std.mem.Allocator, _: NetworkConfig) Error!void {
    return error.NetworkDisabled;
}
pub fn deinit() void {}
