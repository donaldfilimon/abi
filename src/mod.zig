// Public ABI re‑exports – this is what users import as `@import("abi")`.
pub const ai = @import("features/ai/mod.zig");
pub const database = @import("features/database/mod.zig");
pub const gpu = @import("features/gpu/mod.zig");
pub const web = @import("features/web/mod.zig");
pub const monitoring = @import("features/monitoring/mod.zig");
pub const connectors = @import("features/connectors/mod.zig");
pub const VectorOps = @import("shared/simd.zig");
pub const framework = @import("framework/mod.zig");
pub const cli = @import("cli/mod.zig");
