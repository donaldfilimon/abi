pub const registry = @import("registry.zig");
pub const ha = @import("ha.zig");
pub const discovery = @import("discovery.zig");
pub const heartbeat = @import("heartbeat.zig");
pub const loadbalancer = @import("loadbalancer.zig");

// Re-exports
pub const NodeRegistry = registry.NodeRegistry;
pub const NodeInfo = registry.NodeInfo;
pub const NodeStatus = registry.NodeStatus;
pub const Node = NodeInfo;

pub const HealthCheck = ha.HealthCheck;
pub const ClusterConfig = ha.ClusterConfig;
pub const HaError = ha.HaError;
pub const NodeHealth = ha.NodeHealth;
pub const ClusterState = ha.ClusterState;
pub const HealthCheckResult = ha.HealthCheckResult;
pub const FailoverPolicy = ha.FailoverPolicy;

pub const ServiceDiscovery = discovery.ServiceDiscovery;
pub const DiscoveryConfig = discovery.DiscoveryConfig;
pub const DiscoveryBackend = discovery.DiscoveryBackend;
pub const ServiceInstance = discovery.ServiceInstance;
pub const ServiceStatus = discovery.ServiceStatus;
pub const DiscoveryError = discovery.DiscoveryError;
pub const generateServiceId = discovery.generateServiceId;
pub const base64Encode = discovery.base64Encode;
pub const base64Decode = discovery.base64Decode;

pub const NodeHealthState = heartbeat.NodeHealthState;
pub const ClusterHealthState = heartbeat.ClusterHealthState;
pub const HeartbeatEvent = heartbeat.HeartbeatEvent;
pub const HeartbeatConfig = heartbeat.HeartbeatConfig;
pub const HeartbeatStateMachine = heartbeat.HeartbeatStateMachine;
pub const EventCallback = heartbeat.EventCallback;

pub const LoadBalancer = loadbalancer.LoadBalancer;
pub const LoadBalancerConfig = loadbalancer.LoadBalancerConfig;
pub const LoadBalancerStrategy = loadbalancer.LoadBalancerStrategy;
pub const LoadBalancerError = loadbalancer.LoadBalancerError;
pub const NodeState = loadbalancer.NodeState;
pub const NodeStats = loadbalancer.NodeStats;
