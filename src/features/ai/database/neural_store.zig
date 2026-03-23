//! AI-local adapter for the legacy string-keyed neural ingestion engine.
//!
//! This keeps neural ingestion/search out of the public `abi.database` surface
//! while avoiding direct `core/database/*` imports throughout the AI feature.

pub const Metadata = @import("../../../core/database/neural.zig").Metadata;
pub const SearchOptions = @import("../../../core/database/neural.zig").SearchOptions;
pub const SearchResult = @import("../../../core/database/neural.zig").SearchResult;
pub const WritePolicy = @import("../../../core/database/neural.zig").WritePolicy;
pub const EngineVector = @import("../../../core/database/neural.zig").EngineVector;
pub const Engine = @import("../../../core/database/neural.zig").Engine;
pub const save = @import("../../../core/database/neural.zig").save;
pub const load = @import("../../../core/database/neural.zig").load;
