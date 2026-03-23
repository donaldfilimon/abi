//! Database Feature Module
//!
//! Canonical public entrypoint for vector database operations.
//! Delegates to the core unified database engine (`core/database/mod.zig`).
//!
//! This module re-exports the full public API surface so that callers get
//! identical types and functions regardless of whether `feat_database` selects
//! `mod.zig` (this file) or `stub.zig` (no-op).

const std = @import("std");
const build_options = @import("build_options");
pub const types = @import("types.zig");

// ── Core database module (all real logic lives here) ───────────────────────
const core_db = @import("../../core/database/mod.zig");
const core_config = @import("../../core/config/database.zig");

// ── Sub-module re-exports (parity with stub.zig) ───────────────────────────
pub const engine = core_db.engine;
pub const hnsw = core_db.hnsw;
pub const distance = @import("../../core/database/distance.zig");
pub const simd = @import("../../core/database/simd.zig");
pub const quantize = @import("../../core/database/quantize.zig");
pub const batch = core_db.batch;
pub const fulltext = core_db.fulltext;
pub const core = core_db.core;
pub const hybrid = core_db.hybrid;
pub const filter = core_db.filter;
pub const clustering = core_db.clustering;
pub const formats = core_db.formats;
pub const index = core_db.index;
pub const quantization = core_db.quantization;
pub const parallel_hnsw = core_db.parallel_hnsw;
pub const parallel_search = core_db.parallel_search;
pub const database = core_db.database;
pub const storage = core_db.storage;
pub const cli = core_db.cli;
pub const neural = core_db.neural;
pub const semantic_store = core_db.semantic_store;
pub const block_chain = core_db.block_chain;

// ── Type re-exports (parity with stub.zig) ─────────────────────────────────
pub const DatabaseConfig = core_config.DatabaseConfig;
pub const DatabaseHandle = core_db.DatabaseHandle;
pub const SearchResult = core_db.SearchResult;
pub const VectorView = core_db.VectorView;
pub const Stats = core_db.Stats;
pub const BatchItem = core_db.BatchItem;
pub const StoreHandle = core_db.StoreHandle;
pub const DiagnosticsInfo = core_db.DiagnosticsInfo;
pub const DatabaseError = core_db.DatabaseError;
pub const KMeans = core_db.KMeans;
pub const ScalarQuantizer = core_db.ScalarQuantizer;
pub const ProductQuantizer = core_db.ProductQuantizer;

pub const DatabaseFeatureError = types.DatabaseFeatureError;

// ── Context (framework integration) ────────────────────────────────────────
pub const Context = core_db.Context;

// ── Module-level functions (parity with stub.zig) ──────────────────────────

pub const init = core_db.init;
pub const deinit = core_db.deinit;
pub const isInitialized = core_db.isInitialized;
pub const open = core_db.open;
pub const connect = core_db.connect;
pub const close = core_db.close;
pub const insert = core_db.insert;
pub const search = core_db.search;
pub const searchInto = core_db.searchInto;
pub const remove = core_db.remove;
pub const update = core_db.update;
pub const get = core_db.get;
pub const list = core_db.list;
pub const stats = core_db.stats;
pub const diagnostics = core_db.diagnostics;
pub const optimize = core_db.optimize;
pub const backup = core_db.backup;
pub const restore = core_db.restore;
pub const backupToPath = core_db.backupToPath;
pub const restoreFromPath = core_db.restoreFromPath;
pub const openFromFile = core_db.openFromFile;
pub const openOrCreate = core_db.openOrCreate;

/// Check if the database module is enabled at compile time.
pub fn isEnabled() bool {
    return build_options.feat_database;
}

test {
    std.testing.refAllDecls(@This());
}
