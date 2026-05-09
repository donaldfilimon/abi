//! Framework lifecycle state definitions.

pub const State = enum {
    uninitialized,
    initializing,
    running,
    stopping,
    stopped,
    failed,
};
