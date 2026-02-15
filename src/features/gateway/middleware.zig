const types = @import("types.zig");

/// Gateway middleware type aliases are centralized here so future middleware
/// handlers can be split out without changing the public gateway module.
pub const MiddlewareType = types.MiddlewareType;
