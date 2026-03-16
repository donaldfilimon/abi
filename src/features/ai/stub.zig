//! AI feature stub facade used when `feat_ai` is disabled.

const std = @import("std");
const framework_config = @import("../../core/config/mod.zig");
const core_facade = @import("facades/core_stub.zig");

pub const Error = core_facade.Error;

pub const types = @import("types.zig");
pub const config = @import("config.zig");
pub const registry = @import("registry.zig");
pub const profiles = @import("profiles/stub.zig");
pub const personas = profiles;

pub const core = @import("core/stub.zig");
pub const agents = @import("agents/stub.zig");
pub const agent = agents;
pub const llm = @import("llm/stub.zig");
pub const embeddings = @import("embeddings/stub.zig");
pub const training = @import("training/stub.zig");
pub const streaming = @import("streaming/stub.zig");
pub const explore = @import("explore/stub.zig");
pub const abbey = @import("abbey/stub.zig");
pub const tools = @import("tools/stub.zig");
pub const prompts = @import("prompts/stub.zig");
pub const memory = @import("memory/stub.zig");
pub const reasoning = @import("reasoning/stub.zig");
pub const constitution = @import("constitution/stub.zig");
pub const eval = @import("eval/stub.zig");
pub const rag = @import("rag/stub.zig");
pub const templates = @import("templates/stub.zig");
pub const orchestration = @import("orchestration/stub.zig");
pub const documents = @import("documents/stub.zig");
pub const database = @import("database/stub.zig");
pub const vision = @import("vision/stub.zig");
pub const multi_agent = @import("multi_agent/stub.zig");
pub const coordination = @import("coordination/stub.zig");

pub const Context = core_facade.Context;
pub const createRegistry = core_facade.createRegistry;
pub const createAgent = core_facade.createAgent;

pub fn init(_: std.mem.Allocator, _: framework_config.AiConfig) !void {
    return error.AiDisabled;
}

pub fn deinit() void {}

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
