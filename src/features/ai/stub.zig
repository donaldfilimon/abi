//! AI feature stub facade used when `feat_ai` is disabled.

const std = @import("std");
const stub_helpers = @import("../core/stub_helpers.zig");
const framework_config = @import("../core/config/mod.zig");
const core_facade = @import("facades/core_stub.zig");

pub const Error = core_facade.Error;

pub const types = @import("types.zig");
pub const config = @import("config.zig");
pub const registry = @import("registry.zig");
pub const profiles = @import("profiles/stub.zig");

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
pub const pipeline = @import("pipeline/stub.zig");
pub const eval = @import("eval/stub.zig");
pub const rag = @import("rag/stub.zig");
pub const templates = @import("templates/stub.zig");
pub const orchestration = @import("orchestration/stub.zig");
pub const documents = @import("documents/stub.zig");
pub const database = @import("database/stub.zig");
pub const vision = @import("vision/stub.zig");
pub const multi_agent = @import("multi_agent/stub.zig");
pub const coordination = @import("coordination/stub.zig");
pub const models = @import("models/stub.zig");
pub const transformer = @import("transformer/stub.zig");
pub const federated = @import("federated/stub.zig");
pub const feedback = @import("feedback/stub.zig");
pub const compliance = @import("compliance/stub.zig");
pub const tool_agent = tools;
pub const discovery = explore;
pub const jumpstart = @import("context_engine/jumpstart.zig");
pub const context_engine = @import("context_engine/stub.zig");
pub const self_improve = @import("self_improve_stub.zig");
pub const profile = @import("profile/stub.zig");
pub const deep_research = struct {
    pub const DeepResearcher = struct {
        allocator: std.mem.Allocator,
        pub fn init(allocator: std.mem.Allocator, _: anytype) DeepResearcher {
            return .{ .allocator = allocator };
        }
        pub fn deinit(_: *DeepResearcher) void {}
        pub fn autonomousSearch(_: *DeepResearcher, _: []const u8) error{AiDisabled}![]const u8 {
            return error.AiDisabled;
        }
    };
};
pub const dynamic_api = struct {
    pub const ApiSchemaType = enum { openapi_v3, graphql, rest_generic, cli_man_page };
    pub const DynamicApiLearner = struct {
        allocator: std.mem.Allocator,
        pub fn init(allocator: std.mem.Allocator) DynamicApiLearner {
            return .{ .allocator = allocator };
        }
        pub fn deinit(_: *DynamicApiLearner) void {}
        pub fn learnNewSystem(_: *DynamicApiLearner, _: ApiSchemaType, _: []const u8) error{AiDisabled}![]const u8 {
            return error.AiDisabled;
        }
    };
};
pub const runtime_bridge = struct {
    pub const RuntimeEnvironment = enum { python3, node, deno };
    pub const RuntimeResult = struct {
        stdout: []const u8 = "",
        stderr: []const u8 = "",
        exit_code: u8 = 1,
        pub fn deinit(_: *RuntimeResult, _: std.mem.Allocator) void {}
    };
    pub const RuntimeBridge = struct {
        allocator: std.mem.Allocator,
        pub fn init(allocator: std.mem.Allocator, _: anytype) RuntimeBridge {
            return .{ .allocator = allocator };
        }
        pub fn deinit(_: *RuntimeBridge) void {}
        pub fn executeScript(_: *RuntimeBridge, _: RuntimeEnvironment, _: []const u8) error{AiDisabled}!RuntimeResult {
            return error.AiDisabled;
        }
    };
};
pub const os_control = struct {
    pub const PermissionLevel = enum { full_control, ask_before_action, read_only };
    pub const OSControlManager = struct {
        allocator: std.mem.Allocator,
        pub fn init(allocator: std.mem.Allocator, _: PermissionLevel) OSControlManager {
            return .{ .allocator = allocator };
        }
        pub fn deinit(_: *OSControlManager) void {}
        pub fn captureScreen(_: *OSControlManager) error{AiDisabled}![]const u8 {
            return error.AiDisabled;
        }
        pub fn typeKeys(_: *OSControlManager, _: []const u8) error{AiDisabled}!void {
            return error.AiDisabled;
        }
    };
};

pub const Context = core_facade.Context;
pub const createRegistry = core_facade.createRegistry;
pub const createAgent = core_facade.createAgent;

const _stub = stub_helpers.StubFeature(framework_config.AiConfig, error{FeatureDisabled});
pub const init = _stub.init;
pub const deinit = _stub.deinit;
pub const isEnabled = _stub.isEnabled;
pub const isInitialized = _stub.isInitialized;

test {
    std.testing.refAllDecls(@This());
}
