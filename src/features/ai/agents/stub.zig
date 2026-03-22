//! Agents Stub Module — disabled at compile time.

const std = @import("std");
const config_module = @import("../../../core/config/mod.zig");
const types = @import("types.zig");

pub const llm_providers = @import("../llm/stub.zig").providers;

// ── Re-exported types ──────────────────────────────────────────────────────

pub const Error = types.Error;
pub const MIN_TEMPERATURE = types.MIN_TEMPERATURE;
pub const MAX_TEMPERATURE = types.MAX_TEMPERATURE;
pub const MIN_TOP_P = types.MIN_TOP_P;
pub const MAX_TOP_P = types.MAX_TOP_P;
pub const MAX_TOKENS_LIMIT = types.MAX_TOKENS_LIMIT;
pub const DEFAULT_TEMPERATURE = types.DEFAULT_TEMPERATURE;
pub const DEFAULT_TOP_P = types.DEFAULT_TOP_P;
pub const DEFAULT_MAX_TOKENS = types.DEFAULT_MAX_TOKENS;
pub const AgentError = types.AgentError;
pub const AgentBackend = types.AgentBackend;
pub const OperationContext = types.OperationContext;
pub const ErrorContext = types.ErrorContext;
pub const AgentConfig = types.AgentConfig;
pub const Message = types.Message;
pub const Agent = types.Agent;
pub const WorkloadType = types.WorkloadType;
pub const Priority = types.Priority;
pub const GpuAwareRequest = types.GpuAwareRequest;
pub const GpuAwareResponse = types.GpuAwareResponse;
pub const GpuAgentStats = types.GpuAgentStats;
pub const BackendInfo = types.BackendInfo;
pub const LearningStatsInfo = types.LearningStatsInfo;
pub const GpuAgent = types.GpuAgent;
pub const Context = types.Context;
pub const ParameterType = types.ParameterType;
pub const Parameter = types.Parameter;
pub const ToolExecutionError = types.ToolExecutionError;
pub const ToolResult = types.ToolResult;
pub const Tool = types.Tool;
pub const ToolContext = types.ToolContext;
pub const ToolRegistry = types.ToolRegistry;

pub fn isEnabled() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
