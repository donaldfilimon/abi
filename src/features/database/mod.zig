//! Canonical public WDBX surface.
//!
//! Public ABI callers should use this module instead of reaching into
//! `src/core/database/*` directly.

const build_options = @import("build_options");

pub const types = @import("types.zig");
pub const DatabaseFeatureError = types.DatabaseFeatureError;

pub const store = @import("../../core/database/store/mod.zig");
pub const Store = store.Store;
pub const Context = store.Context;
pub const SearchResult = store.SearchResult;
pub const VectorView = store.VectorView;
pub const Stats = store.Stats;
pub const BatchItem = store.BatchItem;
pub const DatabaseConfig = store.DatabaseConfig;
pub const DiagnosticsInfo = store.DiagnosticsInfo;
pub const DatabaseError = store.DatabaseError;

pub const memory = @import("../../core/database/memory/mod.zig");
pub const storage = @import("../../core/database/storage/mod.zig");
pub const distributed = @import("../../core/database/distributed/mod.zig");
pub const retrieval = @import("../../core/database/retrieval/mod.zig");

pub const init = store.init;
pub const deinit = store.deinit;
pub const isInitialized = store.isInitialized;
pub const cli = @import("../../core/database/cli.zig");

pub fn isEnabled() bool {
    return build_options.feat_database;
}

test {
    @import("std").testing.refAllDecls(@This());
}
