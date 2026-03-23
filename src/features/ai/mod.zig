//! AI feature facade.
//!
//! This top-level module presents the canonical `abi.ai` surface for framework
//! code, tests, and external callers. Compatibility aliases delegate here while
//! the stub-facing contract stays aligned with `stub.zig`.

const std = @import("std");
const build_options = @import("build_options");
const framework_config = @import("../../core/config/mod.zig");
const core_facade = @import("facades/core.zig");

var initialized: bool = false;

pub const Error = core_facade.Error;

pub const types = @import("types.zig");
pub const config = @import("config.zig");
pub const registry = @import("registry.zig");
pub const profiles = if (build_options.feat_ai) @import("profiles/mod.zig") else @import("profiles/stub.zig");

pub const core = if (build_options.feat_ai) @import("core/mod.zig") else @import("core/stub.zig");
pub const agents = if (build_options.feat_ai) @import("agents/mod.zig") else @import("agents/stub.zig");
pub const agent = agents;
pub const llm = if (build_options.feat_llm) @import("llm/mod.zig") else @import("llm/stub.zig");
pub const embeddings = if (build_options.feat_ai) @import("embeddings/mod.zig") else @import("embeddings/stub.zig");
pub const training = if (build_options.feat_training) @import("training/mod.zig") else @import("training/stub.zig");
pub const streaming = if (build_options.feat_ai) @import("streaming/mod.zig") else @import("streaming/stub.zig");
pub const explore = if (build_options.feat_explore) @import("explore/mod.zig") else @import("explore/stub.zig");
pub const abbey = if (build_options.feat_reasoning) @import("abbey/mod.zig") else @import("abbey/stub.zig");
pub const tools = if (build_options.feat_ai) @import("tools/mod.zig") else @import("tools/stub.zig");
pub const prompts = if (build_options.feat_ai) @import("prompts/mod.zig") else @import("prompts/stub.zig");
pub const memory = if (build_options.feat_ai) @import("memory/mod.zig") else @import("memory/stub.zig");
pub const reasoning = if (build_options.feat_reasoning) @import("reasoning/mod.zig") else @import("reasoning/stub.zig");
pub const constitution = if (build_options.feat_reasoning) @import("constitution/mod.zig") else @import("constitution/stub.zig");
pub const eval = if (build_options.feat_reasoning) @import("eval/mod.zig") else @import("eval/stub.zig");
pub const rag = if (build_options.feat_reasoning) @import("rag/mod.zig") else @import("rag/stub.zig");
pub const templates = if (build_options.feat_ai) @import("templates/mod.zig") else @import("templates/stub.zig");
pub const orchestration = if (build_options.feat_ai) @import("orchestration/mod.zig") else @import("orchestration/stub.zig");
pub const documents = if (build_options.feat_ai) @import("documents/mod.zig") else @import("documents/stub.zig");
pub const database = if (build_options.feat_ai) @import("database/mod.zig") else @import("database/stub.zig");
pub const vision = if (build_options.feat_vision) @import("vision/mod.zig") else @import("vision/stub.zig");
pub const multi_agent = if (build_options.feat_ai) @import("multi_agent/mod.zig") else @import("multi_agent/stub.zig");
pub const coordination = if (build_options.feat_ai) @import("coordination/mod.zig") else @import("coordination/stub.zig");
pub const models = if (build_options.feat_ai) @import("models/mod.zig") else @import("models/stub.zig");
pub const transformer = if (build_options.feat_ai) @import("transformer/mod.zig") else @import("transformer/stub.zig");
pub const federated = if (build_options.feat_ai) @import("federated/mod.zig") else @import("federated/stub.zig");

/// Multi-persona orchestration: registry, router, collaboration bus.
pub const persona = if (build_options.feat_ai) @import("persona/mod.zig") else @import("persona/stub.zig");

pub const tool_agent = tools;
pub const discovery = explore;
pub const jumpstart = @import("context_engine/jumpstart.zig");
pub const context_engine = @import("context_engine/mod.zig");
pub const self_improve = @import("self_improve.zig");
pub const deep_research = @import("tools/deep_research.zig");
pub const dynamic_api = @import("tools/dynamic_api.zig");
pub const runtime_bridge = @import("tools/runtime_bridge.zig");
pub const os_control = @import("tools/os_control.zig");

pub const Context = core_facade.Context;
pub const createRegistry = core_facade.createRegistry;
pub const createAgent = core_facade.createAgent;

pub fn init(_: std.mem.Allocator, _: framework_config.AiConfig) !void {
    initialized = true;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return build_options.feat_ai;
}

pub fn isInitialized() bool {
    return initialized;
}

test {
    std.testing.refAllDecls(@This());
}
