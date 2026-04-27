//! I/O module for LLM file operations.
//!
//! Provides memory-mapped file access, GGUF format parsing, and GGUF export
//! for efficient model loading and trained model serialization.

const std = @import("std");

pub const mmap = @import("mmap.zig");
pub const gguf = @import("gguf.zig");
pub const gguf_writer = @import("gguf_writer.zig");
pub const tensor_loader = @import("tensor_loader.zig");

// Re-exports for reading
pub const MappedFile = mmap.MappedFile;
pub const MmapError = mmap.MmapError;

pub const GgufFile = gguf.GgufFile;
pub const GgufHeader = gguf.GgufHeader;
pub const GgufMetadata = gguf.GgufMetadata;
pub const GgufMetadataValue = gguf.GgufMetadataValue;
pub const GgufMetadataValueType = gguf.GgufMetadataValueType;
pub const GgufTensorType = gguf.GgufTensorType;
pub const TensorInfo = gguf.TensorInfo;
pub const GgufError = gguf.GgufError;

pub const TensorLoader = tensor_loader.TensorLoader;

// Re-exports for writing
pub const GgufWriter = gguf_writer.GgufWriter;
pub const GgufWriterError = gguf_writer.GgufWriterError;
pub const exportToGguf = gguf_writer.exportToGguf;
pub const ExportConfig = gguf_writer.ExportConfig;
pub const ExportWeights = gguf_writer.ExportWeights;
pub const LayerWeights = gguf_writer.LayerWeights;

test "io module imports" {
    _ = mmap;
    _ = gguf;
    _ = gguf_writer;
    _ = tensor_loader;
}

test {
    std.testing.refAllDecls(@This());
}
