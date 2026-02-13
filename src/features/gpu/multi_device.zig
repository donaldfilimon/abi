//! Multi-GPU device management and coordination.
//!
//! This is a re-export hub. Implementations live in:
//! - `device_group.zig` — Device discovery, load balancing, work distribution, barriers
//! - `gpu_cluster.zig` — Multi-GPU context, AllReduce, scatter/gather, partitioning
//! - `gradient_sync.zig` — Gradient bucketing for efficient AllReduce

const std = @import("std");

// Extracted submodules
pub const device_group_mod = @import("device_group.zig");
pub const gpu_cluster_mod = @import("gpu_cluster.zig");
pub const gradient_sync_mod = @import("gradient_sync.zig");

// Re-export device_group types
pub const DeviceId = device_group_mod.DeviceId;
pub const DeviceType = device_group_mod.DeviceType;
pub const DeviceCapabilities = device_group_mod.DeviceCapabilities;
pub const DeviceInfo = device_group_mod.DeviceInfo;
pub const LoadBalanceStrategy = device_group_mod.LoadBalanceStrategy;
pub const MultiDeviceConfig = device_group_mod.MultiDeviceConfig;
pub const DeviceGroup = device_group_mod.DeviceGroup;
pub const WorkDistribution = device_group_mod.WorkDistribution;
pub const GroupStats = device_group_mod.GroupStats;
pub const PeerTransferConfig = device_group_mod.PeerTransferConfig;
pub const PeerTransfer = device_group_mod.PeerTransfer;
pub const DeviceBarrier = device_group_mod.DeviceBarrier;

// Re-export gpu_cluster types
pub const ReduceOp = gpu_cluster_mod.ReduceOp;
pub const ParallelismStrategy = gpu_cluster_mod.ParallelismStrategy;
pub const ModelPartition = gpu_cluster_mod.ModelPartition;
pub const GPUClusterConfig = gpu_cluster_mod.GPUClusterConfig;
pub const AllReduceAlgorithm = gpu_cluster_mod.AllReduceAlgorithm;
pub const GPUCluster = gpu_cluster_mod.GPUCluster;
pub const TensorPartition = gpu_cluster_mod.TensorPartition;
pub const DeviceChunk = gpu_cluster_mod.DeviceChunk;
pub const ClusterStats = gpu_cluster_mod.ClusterStats;

// Re-export gradient_sync types
pub const GradientBucket = gradient_sync_mod.GradientBucket;
pub const GradientBucketManager = gradient_sync_mod.GradientBucketManager;

// Test discovery for extracted submodules
test {
    _ = @import("device_group.zig");
    _ = @import("gpu_cluster.zig");
    _ = @import("gradient_sync.zig");
    _ = @import("multi_device_test.zig");
}
