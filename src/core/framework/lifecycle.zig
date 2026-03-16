//! Framework lifecycle façade over context init/shutdown modules.

const framework_state = @import("state.zig");
const shutdown_mod = @import("shutdown.zig");
const context_init = @import("context_init.zig");

pub const State = framework_state.State;
pub const RegistryError = shutdown_mod.RegistryError;

pub fn init(comptime Framework: type, allocator: anytype, cfg: anytype) Framework.Error!Framework {
    return context_init.init(Framework, allocator, cfg);
}

pub fn initWithIo(comptime Framework: type, allocator: anytype, cfg: anytype, io: anytype) Framework.Error!Framework {
    return context_init.initWithIo(Framework, allocator, cfg, io);
}

pub fn initDefault(comptime Framework: type, allocator: anytype) Framework.Error!Framework {
    return context_init.initDefault(Framework, allocator);
}

pub fn initMinimal(comptime Framework: type, allocator: anytype) Framework.Error!Framework {
    return context_init.initMinimal(Framework, allocator);
}

pub fn deinit(self: anytype) void {
    shutdown_mod.deinit(self);
}

pub fn shutdownWithTimeout(self: anytype, timeout_ms: u64) bool {
    return shutdown_mod.shutdownWithTimeout(self, timeout_ms);
}

pub fn deinitFeatures(self: anytype) void {
    shutdown_mod.deinitFeatures(self);
}

pub fn deinitOptionalContext(comptime Context: type, slot: *?*Context) void {
    shutdown_mod.deinitOptionalContext(Context, slot);
}
