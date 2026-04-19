// Deploy Model stub – no-op when the feature is disabled
pub const types = @import("types.zig");

pub const DeployConfig = types.DeployConfig;
pub const DeployResult = types.DeployResult;

pub const isEnabled = false;

pub fn runDeployment(allocator: std.mem.Allocator, cfg: DeployConfig) !DeployResult {
    return error.FeatureDisabled;
}
