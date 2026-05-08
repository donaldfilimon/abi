//! Disabled WDBX feature facade mirroring `mod.zig`.

const stub_helpers = @import("../core/stub_helpers.zig");

pub const types = @import("types.zig");
pub const DatabaseFeatureError = types.DatabaseFeatureError;

pub const store = @import("../core/database/store/stub.zig");
pub const Store = store.Store;
pub const Context = store.Context;
pub const SearchResult = store.SearchResult;
pub const VectorView = store.VectorView;
pub const Stats = store.Stats;
pub const BatchItem = store.BatchItem;
pub const DatabaseConfig = store.DatabaseConfig;
pub const DiagnosticsInfo = store.DiagnosticsInfo;
pub const DatabaseError = store.DatabaseError;

pub const memory = @import("../core/database/memory/stub.zig");
pub const storage = @import("../core/database/storage/stub.zig");
pub const distributed = @import("../core/database/stub.zig").distributed;
pub const retrieval = @import("../core/database/retrieval/stub.zig");

// Lifecycle: init/deinit/isInitialized delegate to the store stub (DatabaseDisabled error).
// isEnabled uses the canonical StubFeatureNoConfig helper.
pub const init = store.init;
pub const deinit = store.deinit;
pub const isInitialized = store.isInitialized;
pub const cli = @import("../core/database/stub.zig").cli;

const Stub = stub_helpers.StubFeatureNoConfig(DatabaseFeatureError);
pub const isEnabled = Stub.isEnabled;

test {
    @import("std").testing.refAllDecls(@This());
}
