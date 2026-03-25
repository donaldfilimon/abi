//! Stub for AI Database module

const std = @import("std");

pub const WdbxTokenDataset = struct {
    pub fn init(_: std.mem.Allocator, _: []const u8) !@This() {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *@This()) void {}
    pub fn save(_: *@This()) !void {
        return error.FeatureDisabled;
    }
    pub fn appendTokens(_: *@This(), _: []const u32, _: ?[]const u8) !void {
        return error.FeatureDisabled;
    }
    pub fn importTokenBin(_: *@This(), _: []const u32, _: u32) !void {
        return error.FeatureDisabled;
    }
    pub fn collectTokens(_: *@This(), _: usize) ![]u32 {
        return error.FeatureDisabled;
    }
    pub fn exportTokenBinFile(_: *@This(), _: std.mem.Allocator, _: []const u8, _: usize) !void {
        return error.FeatureDisabled;
    }
    pub fn ingestText(_: *@This(), _: std.mem.Allocator, _: anytype, _: []const u8, _: u32) !void {
        return error.FeatureDisabled;
    }
};

pub fn tokenBinToWdbx(_: std.mem.Allocator, _: []const u8, _: []const u8, _: usize) !void {
    return error.FeatureDisabled;
}
pub fn wdbxToTokenBin(_: std.mem.Allocator, _: []const u8, _: []const u8) !void {
    return error.FeatureDisabled;
}
pub fn readTokenBinFile(_: std.mem.Allocator, _: []const u8) ![]u32 {
    return error.FeatureDisabled;
}
pub fn writeTokenBinFile(_: std.mem.Allocator, _: []const u8, _: []const u32) !void {
    return error.FeatureDisabled;
}
pub fn exportGguf(_: std.mem.Allocator, _: anytype, _: []const u8) !void {
    return error.FeatureDisabled;
}

const shared_types = @import("types.zig");

pub const BrainExportConfig = shared_types.BrainExportConfig;
pub const TrainingMetadata = shared_types.TrainingMetadata;
pub const ExportResult = shared_types.ExportResult;

pub fn exportDual(_: std.mem.Allocator, _: anytype, _: BrainExportConfig, _: ?TrainingMetadata) !ExportResult {
    return error.FeatureDisabled;
}

// Sub-module namespace stubs
pub const dataset = struct {
    pub const WdbxTokenDataset_ = WdbxTokenDataset;
};

pub const convert = struct {
    pub const tokenBinToWdbx_ = tokenBinToWdbx;
    pub const wdbxToTokenBin_ = wdbxToTokenBin;
    pub const readTokenBinFile_ = readTokenBinFile;
    pub const writeTokenBinFile_ = writeTokenBinFile;
};

pub const export_mod = struct {
    pub const exportGguf_ = exportGguf;
};

pub const brain_export = struct {
    pub const BrainExportConfig_ = BrainExportConfig;
    pub const TrainingMetadata_ = TrainingMetadata;
    pub const ExportResult_ = ExportResult;
    pub const exportDual_ = @import("stub.zig").exportDual;
};

test {
    std.testing.refAllDecls(@This());
}
