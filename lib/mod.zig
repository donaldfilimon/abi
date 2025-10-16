//! Compatibility forwarder: `lib/mod.zig` forwards to the canonical `src/mod.zig`.
//! This preserves legacy imports while ensuring a single authoritative API surface.

pub const root = @import("../src/mod.zig");