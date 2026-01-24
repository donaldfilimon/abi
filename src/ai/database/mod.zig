//! AI Database Integration Module
//!
//! Provides WDBX-based datasets, conversion utilities, and model export features.

pub const wdbx = @import("wdbx.zig");
pub const convert = @import("convert.zig");
pub const export_mod = @import("export.zig");

// Re-exports
pub const WdbxTokenDataset = wdbx.WdbxTokenDataset;
pub const tokenBinToWdbx = convert.tokenBinToWdbx;
pub const wdbxToTokenBin = convert.wdbxToTokenBin;
pub const readTokenBinFile = convert.readTokenBinFile;
pub const writeTokenBinFile = convert.writeTokenBinFile;
pub const exportGguf = export_mod.exportGguf;
