//! Stub for AI Database module

const std = @import("std");

pub const WdbxTokenDataset = WdbxTokenDataset_Internal;
pub const tokenBinToWdbx = tokenBinToWdbx_Internal;
pub const wdbxToTokenBin = wdbxToTokenBin_Internal;
pub const readTokenBinFile = readTokenBinFile_Internal;
pub const writeTokenBinFile = writeTokenBinFile_Internal;
pub const exportGgufFromState = exportGgufFromState_Internal;
pub const BrainExportConfig = BrainExportConfig_Internal;
pub const TrainingMetadata = TrainingMetadata_Internal;
pub const ExportResult = ExportResult_Internal;
pub const exportDual = exportDual_Internal;

pub const WdbxTokenDataset_Internal = struct {
    pub fn init(_: std.mem.Allocator, _: []const u8) !WdbxTokenDataset_Internal {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *WdbxTokenDataset_Internal) void {}
    pub fn save(_: *WdbxTokenDataset_Internal) !void {
        return error.FeatureDisabled;
    }
    pub fn appendTokens(_: *WdbxTokenDataset_Internal, _: []const u32, _: ?[]const u8) !void {
        return error.FeatureDisabled;
    }
    pub fn importTokenBin(_: *WdbxTokenDataset_Internal, _: []const u32, _: u32) !void {
        return error.FeatureDisabled;
    }
    pub fn collectTokens(_: *WdbxTokenDataset_Internal, _: usize) ![]u32 {
        return error.FeatureDisabled;
    }
    pub fn exportTokenBinFile(_: *WdbxTokenDataset_Internal, _: std.mem.Allocator, _: []const u8, _: usize) !void {
        return error.FeatureDisabled;
    }
    pub fn ingestText(_: *WdbxTokenDataset_Internal, _: std.mem.Allocator, _: anytype, _: []const u8, _: u32) !void {
        return error.FeatureDisabled;
    }
};

pub fn tokenBinToWdbx_Internal(_: std.mem.Allocator, _: []const u8, _: []const u8, _: usize) !void {
    return error.FeatureDisabled;
}
pub fn wdbxToTokenBin_Internal(_: std.mem.Allocator, _: []const u8, _: []const u8) !void {
    return error.FeatureDisabled;
}
pub fn readTokenBinFile_Internal(_: std.mem.Allocator, _: []const u8) ![]u32 {
    return error.FeatureDisabled;
}
pub fn writeTokenBinFile_Internal(_: std.mem.Allocator, _: []const u8, _: []const u32) !void {
    return error.FeatureDisabled;
}
pub fn exportGgufFromState_Internal(_: std.mem.Allocator, _: anytype, _: []const u8, _: []const u8) !void {
    return error.FeatureDisabled;
}

const shared_types = @import("types.zig");

pub const BrainExportConfig_Internal = shared_types.BrainExportConfig;
pub const TrainingMetadata_Internal = shared_types.TrainingMetadata;
pub const ExportResult_Internal = shared_types.ExportResult;

pub fn exportDual_Internal(_: std.mem.Allocator, _: anytype, _: BrainExportConfig_Internal, _: ?TrainingMetadata_Internal) !ExportResult_Internal {
    return error.FeatureDisabled;
}

// Sub-module namespace stubs
pub const dataset = struct {
    pub const WdbxTokenDataset = WdbxTokenDataset_Internal;
};

pub const convert = struct {
    pub const tokenBinToWdbx = tokenBinToWdbx_Internal;
    pub const wdbxToTokenBin = wdbxToTokenBin_Internal;
    pub const readTokenBinFile = readTokenBinFile_Internal;
    pub const writeTokenBinFile = writeTokenBinFile_Internal;
};

pub const export_mod = struct {
    pub const exportGgufFromState = exportGgufFromState_Internal;
};

pub const brain_export = struct {
    pub const BrainExportConfig = BrainExportConfig_Internal;
    pub const TrainingMetadata = TrainingMetadata_Internal;
    pub const ExportResult = ExportResult_Internal;
    pub const exportDual = exportDual_Internal;
};

test {
    std.testing.refAllDecls(@This());
}
