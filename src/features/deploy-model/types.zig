// Shared types for the Deploy Model feature (deploy-model)
// This provides a minimal surface for both real and stub implementations.

const std = @import("std");

pub const DeployConfig = struct {
    version: []const u8 = "latest",
    sku: []const u8 = "GlobalStandard",
    capacity: usize = 50,
    rai_policy: []const u8 = "Default",
    dynamic_quota: bool = false,
    priority_processing: bool = false,
    spillover: bool = false,
};

pub const DeployResult = struct {
    region: []const u8,
    deployment_name: []const u8,
    status: []const u8,
};
