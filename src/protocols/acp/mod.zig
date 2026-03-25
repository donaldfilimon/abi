pub const server = @import("server.zig");
pub const Server = server.Server;
pub const AgentCard = server.AgentCard;
pub const Task = server.Task;
pub const Session = server.Session;
pub const serveHttp = server.serveHttp;
pub const TaskStatus = server.TaskStatus;
pub const HttpError = server.HttpError;
pub const TransitionError = server.TransitionError;

pub fn isEnabled() bool {
    return true;
}

pub fn isInitialized() bool {
    return true;
}

test {
    @import("std").testing.refAllDecls(@This());
}
