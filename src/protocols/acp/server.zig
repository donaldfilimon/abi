//! ACP (Agent Communication Protocol) Service
//!
//! Re-exports from the decomposed server/ submodules.
//! See server/mod.zig for the main orchestrator.

const server_mod = @import("server/mod.zig");

pub const AgentCard = server_mod.AgentCard;
pub const TaskStatus = server_mod.TaskStatus;
pub const Task = server_mod.Task;
pub const Session = server_mod.Session;
pub const Server = server_mod.Server;
pub const HttpError = server_mod.HttpError;
pub const TransitionError = server_mod.TransitionError;
pub const serveHttp = server_mod.serveHttp;

// Re-export submodule namespaces for direct access
pub const agent_card = server_mod.agent_card;
pub const tasks = server_mod.tasks;
pub const sessions = server_mod.sessions;
pub const routing = server_mod.routing;
pub const json_utils = server_mod.json_utils;

test {
    @import("std").testing.refAllDecls(@This());
}
