//! Disabled WDBX feature facade mirroring `mod.zig`.

pub const types = @import("types.zig");
pub const DatabaseFeatureError = types.DatabaseFeatureError;

pub const store = @import("../../core/database/store/stub.zig");
pub const Store = store.Store;
pub const Context = store.Context;
pub const SearchResult = store.SearchResult;
pub const VectorView = store.VectorView;
pub const Stats = store.Stats;
pub const BatchItem = store.BatchItem;
pub const DatabaseConfig = store.DatabaseConfig;
pub const DiagnosticsInfo = store.DiagnosticsInfo;
pub const DatabaseError = store.DatabaseError;

pub const memory = @import("../../core/database/memory/stub.zig");
pub const storage = @import("../../core/database/storage/stub.zig");
pub const distributed = @import("../../core/database/stub.zig").distributed;
pub const retrieval = @import("../../core/database/retrieval/stub.zig");

pub const init = store.init;
pub const deinit = store.deinit;
pub const isInitialized = store.isInitialized;
pub const cli = @import("../../core/database/stub.zig").cli;

pub fn isEnabled() bool {
    return false;
}

test {
    @import("std").testing.refAllDecls(@This());
}
