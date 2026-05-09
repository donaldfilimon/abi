//! AI Database Integration Module
//!
//! Provides database-backed datasets, conversion utilities, and model export features.
const std = @import("std");

pub const dataset = @import("wdbx.zig");
pub const convert = @import("convert.zig");
pub const export_mod = @import("export.zig");
pub const brain_export = @import("brain_export.zig");

// Re-exports
pub const WdbxTokenDataset = dataset.WdbxTokenDataset;
pub const tokenBinToWdbx = convert.tokenBinToWdbx;
pub const wdbxToTokenBin = convert.wdbxToTokenBin;
pub const readTokenBinFile = convert.readTokenBinFile;
pub const writeTokenBinFile = convert.writeTokenBinFile;
pub const exportGgufFromState = export_mod.exportGgufFromState;
pub const BrainExportConfig = brain_export.BrainExportConfig;
pub const TrainingMetadata = brain_export.TrainingMetadata;
pub const ExportResult = brain_export.ExportResult;
pub const exportDual = brain_export.exportDual;

// Internal exports for mod/stub parity
pub const WdbxTokenDataset_Internal = dataset.WdbxTokenDataset;
pub const tokenBinToWdbx_Internal = convert.tokenBinToWdbx;
pub const wdbxToTokenBin_Internal = convert.wdbxToTokenBin;
pub const readTokenBinFile_Internal = convert.readTokenBinFile;
pub const writeTokenBinFile_Internal = convert.writeTokenBinFile;
pub const exportGgufFromState_Internal = export_mod.exportGgufFromState;
pub const BrainExportConfig_Internal = brain_export.BrainExportConfig;
pub const TrainingMetadata_Internal = brain_export.TrainingMetadata;
pub const ExportResult_Internal = brain_export.ExportResult;
pub const exportDual_Internal = brain_export.exportDual;

test {
    std.testing.refAllDecls(@This());
}
