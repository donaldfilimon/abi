//! WDBX Database Storage Format v2 — re-export proxy.
//!
//! The implementation has been decomposed into focused sub-modules under
//! `storage/`. This file re-exports everything so that existing callers
//! (e.g. `@import("storage.zig")`) continue to work unchanged.

const std = @import("std");
const database = @import("database.zig");
const mod = @import("storage/mod.zig");

// Sub-modules
pub const format = mod.format;
pub const integrity = mod.integrity;
pub const compression = mod.compression;
pub const writer = mod.writer;
pub const reader = mod.reader;
pub const wal = mod.wal;

// --- format.zig ---
pub const MAGIC = mod.MAGIC;
pub const FORMAT_VERSION = mod.FORMAT_VERSION;
pub const MIN_READ_VERSION = mod.MIN_READ_VERSION;
pub const BlockType = mod.BlockType;
pub const CompressionType = mod.CompressionType;
pub const FileHeader = mod.FileHeader;
pub const HeaderFlags = mod.HeaderFlags;
pub const DistanceMetric = mod.DistanceMetric;
pub const FileFooter = mod.FileFooter;

// --- integrity.zig ---
pub const Crc32 = mod.Crc32;
pub const BloomFilter = mod.BloomFilter;

// --- compression.zig ---
pub const deltaEncode = mod.deltaEncode;
pub const deltaDecode = mod.deltaDecode;
pub const quantizeVectors = mod.quantizeVectors;
pub const QuantizedVectors = mod.QuantizedVectors;

// --- wal.zig ---
pub const WalEntry = mod.WalEntry;
pub const WalEntryType = mod.WalEntryType;
pub const WalWriter = mod.WalWriter;

// --- Shared types ---
pub const StorageV2Error = mod.StorageV2Error;
pub const StorageV2Config = mod.StorageV2Config;
pub const HnswGraphData = mod.HnswGraphData;
pub const freeHnswGraphData = mod.freeHnswGraphData;

// --- writer.zig ---
pub const saveDatabaseV2 = mod.saveDatabaseV2;
pub const saveDatabaseWithIndex = mod.saveDatabaseWithIndex;

// --- reader.zig ---
pub const loadDatabaseV2 = mod.loadDatabaseV2;

// --- Unified API ---
pub const saveDatabase = mod.saveDatabase;
pub const saveDatabaseWithConfig = mod.saveDatabaseWithConfig;
pub const loadDatabase = mod.loadDatabase;
pub const loadDatabaseWithConfig = mod.loadDatabaseWithConfig;

test {
    std.testing.refAllDecls(@This());
}
