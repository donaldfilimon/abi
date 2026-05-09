//! Internal neural-engine facade for vector indexing and persistence.

pub const Metadata = @import("engine.zig").Metadata;
pub const SearchOptions = @import("engine.zig").SearchOptions;
pub const SearchResult = @import("engine.zig").SearchResult;
pub const WritePolicy = @import("engine.zig").WritePolicy;
pub const EngineVector = @import("engine.zig").EngineVector;
pub const Engine = @import("engine.zig").Engine;
pub const save = @import("persistence.zig").save;
pub const load = @import("persistence.zig").load;

test {
    @import("std").testing.refAllDecls(@This());
}
