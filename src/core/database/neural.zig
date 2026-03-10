//! Internal neural-engine facade for vector indexing and persistence.

pub const Metadata = @import("engine").Metadata;
pub const SearchOptions = @import("engine").SearchOptions;
pub const SearchResult = @import("engine").SearchResult;
pub const WritePolicy = @import("engine").WritePolicy;
pub const EngineVector = @import("engine").EngineVector;
pub const Engine = @import("engine").Engine;
pub const save = @import("persistence").save;
pub const load = @import("persistence").load;

test {
    @import("std").testing.refAllDecls(@This());
}
