//! I/O module for LLM file operations.
//!
//! Provides memory-mapped file access and GGUF format parsing for efficient
//! model loading without copying data into heap memory.

const std = @import("std");

pub const mmap = @import("mmap.zig");
pub const gguf = @import("gguf.zig");
pub const tensor_loader = @import("tensor_loader.zig");

// Re-exports
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

test "io module imports" {
    _ = mmap;
    _ = gguf;
    _ = tensor_loader;
}
