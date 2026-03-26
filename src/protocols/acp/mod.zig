pub const server = @import("server.zig");
pub const Server = server.Server;
pub const AgentCard = server.AgentCard;
pub const Task = server.Task;
pub const Session = server.Session;
pub const serveHttp = server.serveHttp;
pub const TaskStatus = server.TaskStatus;
pub const HttpError = server.HttpError;
pub const TransitionError = server.TransitionError;
pub const openapi = server.openapi;

pub fn isEnabled() bool {
    const build_options = @import("build_options");
    return build_options.feat_acp;
}

pub fn isInitialized() bool {
    const build_options = @import("build_options");
    return build_options.feat_acp;
}

test {
    @import("std").testing.refAllDecls(@This());
}
