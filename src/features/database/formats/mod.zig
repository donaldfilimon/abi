//! Unified Storage Format Module
//!
//! High-performance, zero-copy storage format with compression and
//! interoperability with standard model formats (GGUF, SafeTensors, ONNX).
//!
//! Features:
//! - Memory-mapped zero-copy access
//! - Optional LZ4/ZSTD/RLE compression
//! - Streaming incremental writes
//! - Format conversion to/from GGUF, SafeTensors, NPY
//! - Versioned format with backward compatibility
//! - Batch operations for throughput
//! - Vector database integration
//! - Full checksum verification
//!
//! File structure:
//! ```
//! +------------------+
//! | Header (64B)     |  Magic, version, flags, counts, checksum
//! +------------------+
//! | Metadata Section |  Key-value pairs (variable)
//! +------------------+
//! | Name Table       |  Tensor names (aligned)
//! +------------------+
//! | Index Table      |  Fast lookup table (aligned)
//! +------------------+
//! | Data Blocks      |  Compressed or raw tensor data
//! +------------------+
//! | Final Checksum   |  CRC32 of entire file (32B)
//! +------------------+
//! ```
//!
//! ## Usage Examples
//!
//! ### Creating a file with StreamingWriter
//! ```zig
//! var writer = StreamingWriter.init(allocator);
//! defer writer.deinit();
//!
//! _ = writer.setCompression(.lz4);
//! try writer.writeTensorTyped(f32, "weights", &data, &.{1024, 768});
//! try writer.writeMetadata("model", "gpt2");
//!
//! const output = try writer.finalize();
//! defer allocator.free(output);
//! ```
//!
//! ### Loading with memory mapping (zero-copy)
//! ```zig
//! var mapped = try MappedFile.open(allocator, "model.abiu");
//! defer mapped.close();
//!
//! var format = try mapped.asUnifiedFormat(allocator);
//! defer format.deinit();
//!
//! const weights = try format.getTensorSlice(f32, "weights");
//! ```
//!
//! ### Converting from GGUF
//! ```zig
//! // Zig 0.16: Use std.Io.Threaded for file I/O
//! var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
//! defer io_backend.deinit();
//! const io = io_backend.io();
//!
//! var gguf_data = try std.Io.Dir.cwd().readFileAlloc(io, "model.gguf", allocator, .limited(max_size));
//! defer allocator.free(gguf_data);
//!
//! var unified = try gguf_converter.fromGguf(allocator, gguf_data);
//! defer unified.deinit();
//! ```
//!
//! ### Vector database with persistence
//! ```zig
//! var db = VectorDatabase.init(allocator, "vectors", 768);
//! defer db.deinit();
//!
//! try db.insert(1, &embedding, "document 1");
//! const results = try db.search(&query, 10);
//!
//! const saved = try db.save();
//! defer allocator.free(saved);
//! ```

const std = @import("std");

// Core modules
pub const unified = @import("unified.zig");
pub const compression = @import("compression.zig");
pub const converters = @import("converters.zig");
pub const streaming = @import("streaming.zig");
pub const mmap = @import("mmap.zig");
pub const vector_db = @import("vector_db.zig");
pub const gguf_converter = @import("gguf_converter.zig");
pub const zon = @import("zon.zig");

// Unified format types
pub const UnifiedFormat = unified.UnifiedFormat;
pub const UnifiedFormatBuilder = unified.UnifiedFormatBuilder;
pub const FormatHeader = unified.FormatHeader;
pub const FormatFlags = unified.FormatFlags;
pub const DataType = unified.DataType;
pub const TensorDescriptor = unified.TensorDescriptor;
pub const UnifiedError = unified.UnifiedError;

// Compression types
pub const CompressionType = compression.CompressionType;
pub const CompressionError = compression.CompressionError;
pub const compress = compression.compress;
pub const decompress = compression.decompress;
pub const estimateCompressedSize = compression.estimateCompressedSize;

// Converter types
pub const Converter = converters.Converter;
pub const ConversionOptions = converters.ConversionOptions;
pub const TargetFormat = converters.TargetFormat;
pub const ConversionError = converters.ConversionError;

// Streaming types
pub const StreamingWriter = streaming.StreamingWriter;
pub const StreamingReader = streaming.StreamingReader;
pub const StreamingError = streaming.StreamingError;

// Memory-mapped file types
pub const MappedFile = mmap.MappedFile;
pub const MemoryCursor = mmap.MemoryCursor;
pub const MmapError = mmap.MmapError;
pub const createMapped = mmap.createMapped;

// Vector database types
pub const VectorDatabase = vector_db.VectorDatabase;
pub const VectorRecord = vector_db.VectorRecord;
pub const SearchResult = vector_db.SearchResult;
pub const VectorDbError = vector_db.VectorDbError;

// GGUF converter types
pub const fromGguf = gguf_converter.fromGguf;
pub const toGguf = gguf_converter.toGguf;
pub const GgufHeader = gguf_converter.GgufHeader;
pub const GgufTensorType = gguf_converter.GgufTensorType;
pub const GgufConversionError = gguf_converter.GgufConversionError;

// ZON format types (Zig Object Notation for WDBX databases)
pub const ZonFormat = zon.ZonFormat;
pub const ZonDatabase = zon.ZonDatabase;
pub const ZonRecord = zon.ZonRecord;
pub const ZonDatabaseConfig = zon.ZonDatabaseConfig;
pub const ZonFormatError = zon.ZonFormatError;
pub const ZonDistanceMetric = zon.DistanceMetric;
pub const ZON_FORMAT_VERSION = zon.ZON_FORMAT_VERSION;
pub const exportToZon = zon.exportToZon;
pub const importFromZon = zon.importFromZon;

/// Format magic number: "ABIU" (ABI Unified)
pub const FORMAT_MAGIC: u32 = 0x55494241;

/// Current format version
pub const FORMAT_VERSION: u16 = 1;

/// Default alignment for data blocks (64 bytes for cache line optimization)
pub const DEFAULT_ALIGNMENT: usize = 64;

/// Maximum supported tensor dimensions
pub const MAX_DIMS: usize = 4;

/// Feature comparison with other formats
pub const FormatComparison = struct {
    /// GGUF (llama.cpp) - Read/Write support, all quantization types
    pub const gguf = struct {
        pub const zero_copy = true;
        pub const compression = false;
        pub const quantization = true;
        pub const streaming = false;
        pub const checksum = false;
    };

    /// SafeTensors (Hugging Face) - Read/Write support
    pub const safetensors = struct {
        pub const zero_copy = true;
        pub const compression = false;
        pub const quantization = false;
        pub const streaming = false;
        pub const checksum = false;
    };

    /// ABI Unified - Full feature support
    pub const unified_format = struct {
        pub const zero_copy = true;
        pub const compression = true; // LZ4, ZSTD, RLE
        pub const quantization = true; // All GGUF types + more
        pub const streaming = true;
        pub const checksum = true; // CRC32 header + full file
        pub const vector_db = true;
        pub const conversion = true; // To/from all formats
    };
};

test "formats module imports" {
    _ = unified;
    _ = compression;
    _ = converters;
    _ = streaming;
    _ = mmap;
    _ = vector_db;
    _ = gguf_converter;
    _ = zon;
}

test "format magic and version" {
    try std.testing.expectEqual(@as(u32, 0x55494241), FORMAT_MAGIC);
    try std.testing.expectEqual(@as(u16, 1), FORMAT_VERSION);
}

test "format roundtrip integration" {
    const allocator = std.testing.allocator;

    // Create a format with tensors and metadata
    var builder = unified.UnifiedFormatBuilder.init(allocator);
    defer builder.deinit();

    _ = builder.setCompression(.lz4);

    const tensor_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    _ = try builder.addTensor("weights", std.mem.sliceAsBytes(&tensor_data), .f32, &.{8});
    _ = try builder.addMetadata("model", "test-model");

    // Build and serialize
    const serialized = try builder.build();
    defer allocator.free(serialized);

    // Load back and verify
    var loaded = try unified.UnifiedFormat.fromMemory(allocator, serialized);
    defer loaded.deinit();

    // Verify tensor exists and has correct properties
    const desc = loaded.getTensor("weights") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(unified.DataType.f32, desc.data_type);
    try std.testing.expectEqual(@as(u64, 8), desc.dims[0]);

    // Verify metadata
    const model_meta = loaded.getMetadata("model") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("test-model", model_meta);

    // Verify decompression works
    const data = try loaded.getTensorData(allocator, "weights");
    defer allocator.free(data);
    try std.testing.expectEqual(@as(usize, 32), data.len); // 8 floats * 4 bytes
}

test {
    std.testing.refAllDecls(@This());
}
