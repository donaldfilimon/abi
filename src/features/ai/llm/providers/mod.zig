const std = @import("std");

pub const errors = @import("errors");
pub const types = @import("types");
pub const request = @import("request");
pub const response = @import("response");
pub const registry = @import("registry");
pub const model_profiles = @import("model_profiles");
pub const parser = @import("parser");
pub const health = @import("health");
pub const router = @import("router");
pub const plugins = @import("plugins");

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

test {
    std.testing.refAllDecls(@This());
}
