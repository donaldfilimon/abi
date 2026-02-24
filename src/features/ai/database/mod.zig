//! AI Database Integration Module
//!
//! Provides WDBX-based datasets, conversion utilities, and model export features.
const std = @import("std");

pub const wdbx = @import("wdbx.zig");
pub const convert = @import("convert.zig");
pub const export_mod = @import("export.zig");
pub const brain_export = @import("brain_export.zig");

// Re-exports
pub const WdbxTokenDataset = wdbx.WdbxTokenDataset;
pub const tokenBinToWdbx = convert.tokenBinToWdbx;
pub const wdbxToTokenBin = convert.wdbxToTokenBin;
pub const readTokenBinFile = convert.readTokenBinFile;
pub const writeTokenBinFile = convert.writeTokenBinFile;
pub const exportGguf = export_mod.exportGguf;
pub const BrainExportConfig = brain_export.BrainExportConfig;
pub const TrainingMetadata = brain_export.TrainingMetadata;
pub const ExportResult = brain_export.ExportResult;
pub const exportDual = brain_export.exportDual;

test {
    std.testing.refAllDecls(@This());
}
