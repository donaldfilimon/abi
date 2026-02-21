pub const catalog = @import("catalog.zig");
pub const selector = @import("selector.zig");
pub const hints = @import("hints.zig");

pub const PlatformClass = catalog.PlatformClass;
pub const OptimizationHints = hints.OptimizationHints;
pub const SelectionContext = selector.SelectionContext;
pub const BackendNameList = selector.BackendNameList;

pub const classify = catalog.classify;
pub const classifyBuiltin = catalog.classifyBuiltin;
pub const defaultOrder = catalog.defaultOrder;
pub const defaultOrderForTarget = catalog.defaultOrderForTarget;
pub const resolveAutoBackendNames = selector.resolveAutoBackendNames;
pub const optimizationHintsForPlatform = hints.forPlatform;
