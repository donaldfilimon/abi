//! Shared types for the AI Database module.
//!
//! These are used by both the real implementation (`mod.zig` / `brain_export.zig`)
//! and the stub (`stub.zig`), eliminating inline duplication that can drift.

const std = @import("std");

/// Configuration for dual brain export (.wdbx + .gguf).
pub const BrainExportConfig = struct {
    /// Path for native .wdbx brain file (always written).
    wdbx_path: []const u8,
    /// Path for .gguf file (optional — skipped if null).
    gguf_path: ?[]const u8 = null,
    /// Include training history/metadata in WDBX.
    include_training_history: bool = true,
    /// Include embedding table vectors (IDs 1..N) in WDBX.
    include_embeddings: bool = true,
};

/// Training metadata stored in WDBX under tag "ABTR".
pub const TrainingMetadata = struct {
    model_name: []const u8 = "unnamed",
    epochs_completed: u32 = 0,
    final_loss: f32 = 0.0,
    learning_rate: f32 = 0.0,
    lora_rank: u32 = 0,
    training_samples: u64 = 0,
    timestamp: i64 = 0,
};

/// Result of a dual export operation.
pub const ExportResult = struct {
    wdbx_written: bool = false,
    gguf_written: bool = false,
    wdbx_size_bytes: u64 = 0,
    gguf_size_bytes: u64 = 0,
};


test {
    std.testing.refAllDecls(@This());
}
