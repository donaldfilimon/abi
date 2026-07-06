const std = @import("std");

pub const backends_mod = @import("backends.zig");
pub const train_mod = @import("train.zig");
pub const agent_mod = @import("agent.zig");
pub const auth_mod = @import("auth.zig");
pub const plugin_mod = @import("plugin.zig");
pub const twilio_mod = @import("twilio.zig");
pub const dashboard_mod = @import("dashboard.zig");
pub const wdbx_mod = @import("wdbx.zig");
pub const scheduler_mod = @import("scheduler.zig");
pub const nn_mod = @import("nn.zig");

// Re-export all handler functions for backward compatibility
pub const handleBackends = backends_mod.handleBackends;
pub const handleTrain = train_mod.handleTrain;
pub const handleComplete = train_mod.handleComplete;
pub const handleAgent = agent_mod.handleAgent;
pub const handleAgentOs = agent_mod.handleAgentOs;
pub const handleAuth = auth_mod.handleAuth;
pub const handlePlugin = plugin_mod.handlePlugin;
pub const handleTwilio = twilio_mod.handleTwilio;
pub const handleDashboard = dashboard_mod.handleDashboard;
pub const renderTui = dashboard_mod.renderTui;
pub const handleWdbx = wdbx_mod.handleWdbx;
pub const handleScheduler = scheduler_mod.handleScheduler;
pub const handleSchedulerStatus = scheduler_mod.handleSchedulerStatus;
pub const handleNn = nn_mod.handleNn;

test {
    std.testing.refAllDecls(@This());
}
