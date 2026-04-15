//! Re-export from gpu/backends/

pub const dynlibSupported = @import("../gpu/backends/shared.zig").dynlibSupported;
pub const canUseDynLib = @import("../gpu/backends/shared.zig").canUseDynLib;
pub const isWebTarget = @import("../gpu/backends/shared.zig").isWebTarget;
pub const tryLoadAny = @import("../gpu/backends/shared.zig").tryLoadAny;
pub const openFirst = @import("../gpu/backends/shared.zig").openFirst;
