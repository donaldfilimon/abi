//! AI Database Integration Module
//!
//! Provides database-backed datasets, conversion utilities, and model export features.
const std = @import("std");

pub const dataset = @import("wdbx.zig");
pub const convert = @import("convert");
pub const export_mod = @import("export");
pub const brain_export = @import("brain_export");

// Re-exports
pub const WdbxTokenDataset = dataset.WdbxTokenDataset;
pub const tokenBinToWdbx = convert.tokenBinToWdbx;
pub const wdbxToTokenBin = convert.wdbxToTokenBin;
pub const readTokenBinFile = convert.readTokenBinFile;
pub const writeTokenBinFile = convert.writeTokenBinFile;
pub const exportGguf = export_mod.exportGguf;
pub const BrainExportConfig = brain_export.BrainExportConfig;
pub const TrainingMetadata = brain_export.TrainingMetadata;
pub const ExportResult = brain_export.ExportResult;
pub const exportDual = brain_export.exportDual;
