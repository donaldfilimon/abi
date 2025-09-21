//! Lightweight GPU showcase that forwards to the feature module used in docs
//! and playground builds.

pub const gpu = @import("../features/gpu/mod.zig");
