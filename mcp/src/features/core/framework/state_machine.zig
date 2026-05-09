const State = @import("state.zig").State;

/// Set the framework state in one place to keep transition points local.
pub inline fn set(next: *State, value: State) void {
    next.* = value;
}

/// Convenience helpers for common framework transitions.
pub inline fn markInitializing(state: *State) void {
    set(state, .initializing);
}

pub inline fn markRunning(state: *State) void {
    set(state, .running);
}

pub inline fn markStopping(state: *State) void {
    set(state, .stopping);
}

pub inline fn markStopped(state: *State) void {
    set(state, .stopped);
}

/// Mark the framework as failed. Currently unused -- errors propagate via
/// errdefer and the Framework struct is destroyed before the caller sees it.
/// Retained for potential future use in long-lived framework instances.
pub inline fn markFailed(state: *State) void {
    set(state, .failed);
}

/// Compatibility helpers used by split lifecycle modules.
pub inline fn isStopped(state: State) bool {
    return state == .stopped or state == .failed;
}
