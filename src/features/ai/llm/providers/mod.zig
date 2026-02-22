const std = @import("std");

pub const errors = @import("errors.zig");
pub const types = @import("types.zig");
pub const request = @import("request.zig");
pub const response = @import("response.zig");
pub const registry = @import("registry.zig");
pub const model_profiles = @import("model_profiles.zig");
pub const parser = @import("parser.zig");
pub const health = @import("health.zig");
pub const router = @import("router.zig");
pub const plugins = @import("plugins/mod.zig");

pub const ProviderError = errors.ProviderError;
pub const ProviderId = types.ProviderId;
pub const GenerateConfig = types.GenerateConfig;
pub const GenerateResult = types.GenerateResult;
pub const ChatMessage = types.ChatMessage;

pub const ModelProfile = model_profiles.ModelProfile;

pub fn generate(allocator: std.mem.Allocator, cfg: GenerateConfig) !GenerateResult {
    return router.generate(allocator, cfg);
}

pub fn getModelProfile(name: []const u8) ?*const ModelProfile {
    return model_profiles.getProfile(name);
}
