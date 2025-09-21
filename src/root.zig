//! Root barrel that exposes the `abi` convenience import along with the public
//! framework surface area for consumers.

pub const abi = @import("mod.zig");
pub const framework = @import("framework/mod.zig");
