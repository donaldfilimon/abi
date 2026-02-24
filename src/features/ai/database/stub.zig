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

pub const BrainExportConfig = struct {
    wdbx_path: []const u8,
    gguf_path: ?[]const u8 = null,
    include_training_history: bool = true,
    include_embeddings: bool = true,
};

pub const TrainingMetadata = struct {
    model_name: []const u8 = "unnamed",
    epochs_completed: u32 = 0,
    final_loss: f32 = 0.0,
    learning_rate: f32 = 0.0,
    lora_rank: u32 = 0,
    training_samples: u64 = 0,
    timestamp: i64 = 0,
};

pub const ExportResult = struct {
    wdbx_written: bool = false,
    gguf_written: bool = false,
    wdbx_size_bytes: u64 = 0,
    gguf_size_bytes: u64 = 0,
};

pub fn exportDual(_: std.mem.Allocator, _: anytype, _: BrainExportConfig, _: ?TrainingMetadata) !ExportResult {
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
