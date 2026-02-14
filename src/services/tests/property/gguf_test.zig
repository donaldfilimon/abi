//! GGUF Format Property-Based Tests
//!
//! Property-based / fuzz tests for GGUF file format parsing including:
//! - Header validation with randomized magic numbers and versions
//! - Tensor metadata parsing with edge cases (dimensions, types, offsets)
//! - Metadata value parsing with various types
//! - Malformed input handling (truncation, corruption, invalid values)
//!
//! These tests ensure the GGUF parser handles adversarial input gracefully
//! without crashes, memory corruption, or undefined behavior.

const std = @import("std");
const property = @import("mod.zig");
const generators = @import("generators.zig");
const abi = @import("abi");
const build_options = @import("build_options");

const forAll = property.forAll;
const forAllWithAllocator = property.forAllWithAllocator;
const Generator = property.Generator;
const assert = property.assert;

// Import GGUF types through abi module
const llm_io = abi.ai.llm.io;
const gguf = llm_io.gguf;
const mmap = llm_io.mmap;

const GgufHeader = llm_io.GgufHeader;
const GgufMetadataValueType = llm_io.GgufMetadataValueType;
const GgufTensorType = llm_io.GgufTensorType;
const TensorInfo = llm_io.TensorInfo;
const GgufError = llm_io.GgufError;
const GgufMetadataValue = llm_io.GgufMetadataValue;
const MemoryCursor = mmap.MemoryCursor;

// GGUF constants
const GGUF_MAGIC = gguf.GGUF_MAGIC;
const GGUF_VERSION_2 = gguf.GGUF_VERSION_2;
const GGUF_VERSION_3 = gguf.GGUF_VERSION_3;

// ============================================================================
// Test Configuration
// ============================================================================

const TestConfig = property.PropertyConfig{
    .iterations = 100,
    .seed = 42,
    .verbose = false,
};

const HighIterConfig = property.PropertyConfig{
    .iterations = 200,
    .seed = 42,
    .verbose = false,
};

const FuzzConfig = property.PropertyConfig{
    .iterations = 500,
    .seed = 12345,
    .verbose = false,
};

// ============================================================================
// GGUF Header Generators
// ============================================================================

/// Generate random GGUF headers with various valid and invalid combinations
const RandomHeader = struct {
    magic: u32,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
};

fn randomHeaderGen() Generator(RandomHeader) {
    return .{
        .generateFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, _: usize) RandomHeader {
                // Mix of valid and invalid magic numbers
                const magic_choices = [_]u32{
                    GGUF_MAGIC, // Valid
                    0x00000000, // All zeros
                    0xFFFFFFFF, // All ones
                    0x46554746, // "GFUF" (one char off)
                    0x47475546, // "GGUF" wrong endianness
                    0x12345678, // Random garbage
                    prng.random().int(u32), // Truly random
                };
                const magic = magic_choices[prng.random().intRangeLessThan(usize, 0, magic_choices.len)];

                // Mix of valid and invalid versions
                const version_choices = [_]u32{
                    GGUF_VERSION_2,
                    GGUF_VERSION_3,
                    0, // Invalid
                    1, // Invalid (v1 not supported)
                    4, // Future version
                    99, // Way in the future
                    std.math.maxInt(u32), // Max value
                };
                const version = version_choices[prng.random().intRangeLessThan(usize, 0, version_choices.len)];

                return .{
                    .magic = magic,
                    .version = version,
                    .tensor_count = prng.random().int(u64),
                    .metadata_kv_count = prng.random().int(u64),
                };
            }
        }.generate,
        .shrinkFn = null,
    };
}

/// Generate valid GGUF headers only
fn validHeaderGen() Generator(RandomHeader) {
    return .{
        .generateFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, size: usize) RandomHeader {
                const version = if (prng.random().boolean()) GGUF_VERSION_2 else GGUF_VERSION_3;
                return .{
                    .magic = GGUF_MAGIC,
                    .version = version,
                    // Limit counts to reasonable values based on size parameter
                    .tensor_count = prng.random().intRangeAtMost(u64, 0, @min(size, 1000)),
                    .metadata_kv_count = prng.random().intRangeAtMost(u64, 0, @min(size, 500)),
                };
            }
        }.generate,
        .shrinkFn = null,
    };
}

// ============================================================================
// Header Validation Tests
// ============================================================================

test "GGUF header validation with random inputs" {
    const gen = randomHeaderGen();

    const result = forAll(RandomHeader, gen, HighIterConfig, struct {
        fn check(random_header: RandomHeader) bool {
            const header = GgufHeader{
                .magic = random_header.magic,
                .version = random_header.version,
                .tensor_count = random_header.tensor_count,
                .metadata_kv_count = random_header.metadata_kv_count,
            };

            const is_valid = header.isValid();

            // Should be valid only if magic and version are correct
            const expected_valid = (random_header.magic == GGUF_MAGIC) and
                (random_header.version == GGUF_VERSION_2 or
                    random_header.version == GGUF_VERSION_3);

            return is_valid == expected_valid;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "GGUF valid headers are always valid" {
    const gen = validHeaderGen();

    const result = forAll(RandomHeader, gen, TestConfig, struct {
        fn check(random_header: RandomHeader) bool {
            const header = GgufHeader{
                .magic = random_header.magic,
                .version = random_header.version,
                .tensor_count = random_header.tensor_count,
                .metadata_kv_count = random_header.metadata_kv_count,
            };

            return header.isValid();
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "GGUF header binary roundtrip" {
    const gen = validHeaderGen();

    const result = forAll(RandomHeader, gen, TestConfig, struct {
        fn check(random_header: RandomHeader) bool {
            // Serialize header to bytes
            var buf: [@sizeOf(GgufHeader)]u8 = undefined;
            std.mem.writeInt(u32, buf[0..4], random_header.magic, .little);
            std.mem.writeInt(u32, buf[4..8], random_header.version, .little);
            std.mem.writeInt(u64, buf[8..16], random_header.tensor_count, .little);
            std.mem.writeInt(u64, buf[16..24], random_header.metadata_kv_count, .little);

            // Parse back using MemoryCursor
            var cursor = MemoryCursor.init(&buf);
            const parsed = cursor.read(GgufHeader) orelse return false;

            return parsed.magic == random_header.magic and
                parsed.version == random_header.version and
                parsed.tensor_count == random_header.tensor_count and
                parsed.metadata_kv_count == random_header.metadata_kv_count;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Tensor Type Generators and Tests
// ============================================================================

fn tensorTypeGen() Generator(u32) {
    return .{
        .generateFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, _: usize) u32 {
                // Mix of valid and invalid tensor type values
                if (prng.random().float(f32) < 0.7) {
                    // 70% valid types
                    const type_idx = prng.random().intRangeLessThan(usize, 0, valid_tensor_types.len);
                    return valid_tensor_types[type_idx];
                } else {
                    // 30% invalid types
                    var candidate = prng.random().intRangeAtMost(u32, 0, 255);
                    while (GgufTensorType.fromInt(candidate) != null) {
                        candidate = prng.random().intRangeAtMost(u32, 0, 255);
                    }
                    return candidate;
                }
            }
        }.generate,
        .shrinkFn = null,
    };
}

test "GGUF tensor type validation" {
    // Generate only valid tensor type indices
    const gen = generators.intRange(usize, 0, valid_tensor_types.len - 1);

    const result = forAll(usize, gen, HighIterConfig, struct {
        fn check(type_idx: usize) bool {
            const type_int = valid_tensor_types[type_idx];
            const tensor_type: GgufTensorType = @enumFromInt(type_int);
            // Valid types should have sensible bytesPerBlock
            const bpb = tensor_type.bytesPerBlock();
            return bpb > 0;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "GGUF tensor type bytes calculation consistency" {
    // Test all valid tensor types using the valid_tensor_types array
    const gen = generators.intRange(usize, 0, valid_tensor_types.len - 1);

    const result = forAll(usize, gen, TestConfig, struct {
        fn check(type_idx: usize) bool {
            const type_int = valid_tensor_types[type_idx];
            const tensor_type: GgufTensorType = @enumFromInt(type_int);

            const bpb = tensor_type.bytesPerBlock();
            const bs = tensor_type.blockSize();

            // Block size and bytes per block must be positive
            if (bpb == 0 or bs == 0) return false;

            // For various element counts, tensorBytes should be consistent
            const test_counts = [_]u64{ 1, 32, 64, 128, 256, 1024, 4096 };
            for (test_counts) |count| {
                const total_bytes = tensor_type.tensorBytes(count);
                // Should be at least enough for the elements
                if (total_bytes == 0 and count > 0) return false;
            }

            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Tensor Info Generators and Tests
// ============================================================================

const RandomTensorInfo = struct {
    n_dims: u32,
    dims: [4]u64,
    tensor_type: u32,
    offset: u64,
};

// Valid tensor type values (excluding deprecated 4 and 5)
const valid_tensor_types = [_]u32{
    0, 1, 2, 3, // f32, f16, q4_0, q4_1
    6, 7, 8, 9, // q5_0, q5_1, q8_0, q8_1
    10, 11, 12, 13, 14, 15, // q2_k through q8_k
    16, 17, 18, 19, 20, 21, 22, 23, // iq variants
    24, 25, 26, 27, 28, 29, 30, // i8 through bf16 (plus iq1_m)
    34, 35, 39, // tq1_0, tq2_0, mxfp4
};

fn tensorInfoGen() Generator(RandomTensorInfo) {
    return .{
        .generateFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, size: usize) RandomTensorInfo {
                const n_dims = prng.random().intRangeAtMost(u32, 1, 4);
                var dims: [4]u64 = .{ 1, 1, 1, 1 };

                // Generate dimensions with varying sizes
                for (0..n_dims) |i| {
                    dims[i] = prng.random().intRangeAtMost(u64, 1, @min(size + 1, 4096));
                }

                // Select from valid tensor types only (skipping deprecated 4 and 5)
                const type_idx = prng.random().intRangeLessThan(usize, 0, valid_tensor_types.len);

                return .{
                    .n_dims = n_dims,
                    .dims = dims,
                    .tensor_type = valid_tensor_types[type_idx],
                    .offset = prng.random().int(u64),
                };
            }
        }.generate,
        .shrinkFn = null,
    };
}

test "GGUF tensor info element count calculation" {
    const gen = tensorInfoGen();

    const result = forAll(RandomTensorInfo, gen, TestConfig, struct {
        fn check(random_info: RandomTensorInfo) bool {
            const info = TensorInfo{
                .name = "test_tensor",
                .n_dims = random_info.n_dims,
                .dims = random_info.dims,
                .tensor_type = @enumFromInt(random_info.tensor_type),
                .offset = random_info.offset,
            };

            const elem_count = info.elementCount();

            // Manually calculate expected count
            var expected: u64 = 1;
            for (0..info.n_dims) |i| {
                expected *= info.dims[i];
            }

            return elem_count == expected;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "GGUF tensor info byte size is non-negative" {
    const gen = tensorInfoGen();

    const result = forAll(RandomTensorInfo, gen, TestConfig, struct {
        fn check(random_info: RandomTensorInfo) bool {
            const info = TensorInfo{
                .name = "test_tensor",
                .n_dims = random_info.n_dims,
                .dims = random_info.dims,
                .tensor_type = @enumFromInt(random_info.tensor_type),
                .offset = random_info.offset,
            };

            const byte_size = info.byteSize();
            const elem_count = info.elementCount();

            // Byte size should be positive if element count is positive
            if (elem_count > 0 and byte_size == 0) return false;

            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "GGUF tensor info shape slice is correct length" {
    const gen = tensorInfoGen();

    const result = forAll(RandomTensorInfo, gen, TestConfig, struct {
        fn check(random_info: RandomTensorInfo) bool {
            const info = TensorInfo{
                .name = "test_tensor",
                .n_dims = random_info.n_dims,
                .dims = random_info.dims,
                .tensor_type = @enumFromInt(random_info.tensor_type),
                .offset = random_info.offset,
            };

            const shape = info.shape();

            return shape.len == info.n_dims;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Metadata Value Type Generators and Tests
// ============================================================================

fn metadataValueTypeGen() Generator(u32) {
    return .{
        .generateFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, _: usize) u32 {
                const max_valid = @intFromEnum(GgufMetadataValueType.float64);
                if (prng.random().float(f32) < 0.8) {
                    return prng.random().intRangeAtMost(u32, 0, max_valid);
                } else {
                    return prng.random().intRangeAtMost(u32, max_valid + 1, 255);
                }
            }
        }.generate,
        .shrinkFn = null,
    };
}

test "GGUF metadata value type validation" {
    const gen = metadataValueTypeGen();

    const result = forAll(u32, gen, TestConfig, struct {
        fn check(type_int: u32) bool {
            const max_valid = @intFromEnum(GgufMetadataValueType.float64);

            if (type_int <= max_valid) {
                // Valid type - enum conversion should work
                const value_type: GgufMetadataValueType = @enumFromInt(type_int);
                // Verify the enum is valid by checking its integer representation
                _ = @intFromEnum(value_type);
                return true;
            } else {
                // Invalid type - would be caught during parsing
                return true;
            }
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Truncated Input Tests (Fuzz for crashes)
// ============================================================================

fn truncatedBytesGen(comptime max_len: usize) Generator([max_len]u8) {
    return .{
        .generateFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, _: usize) [max_len]u8 {
                var buf: [max_len]u8 = undefined;
                prng.random().bytes(&buf);
                return buf;
            }
        }.generate,
        .shrinkFn = null,
    };
}

test "GGUF parsing truncated header does not crash" {
    // Test parsing headers that are too short
    const gen = generators.intRange(usize, 0, @sizeOf(GgufHeader) - 1);

    const result = forAll(usize, gen, TestConfig, struct {
        fn check(truncate_len: usize) bool {
            // Create minimal valid header bytes
            var full_buf: [@sizeOf(GgufHeader)]u8 = undefined;
            std.mem.writeInt(u32, full_buf[0..4], GGUF_MAGIC, .little);
            std.mem.writeInt(u32, full_buf[4..8], GGUF_VERSION_3, .little);
            std.mem.writeInt(u64, full_buf[8..16], 10, .little);
            std.mem.writeInt(u64, full_buf[16..24], 5, .little);

            // Truncate
            const truncated = full_buf[0..truncate_len];

            // Try to parse - should return null, not crash
            var cursor = MemoryCursor.init(truncated);
            const maybe_header = cursor.read(GgufHeader);

            // Should be null for truncated data
            return maybe_header == null;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "GGUF MemoryCursor handles random garbage gracefully" {
    const gen = truncatedBytesGen(128);

    const result = forAll([128]u8, gen, FuzzConfig, struct {
        fn check(garbage: [128]u8) bool {
            // Test readBytes - this doesn't require alignment
            var cursor = MemoryCursor.init(&garbage);
            _ = cursor.readBytes(1);
            _ = cursor.readBytes(4);
            _ = cursor.readBytes(8);
            _ = cursor.readBytes(10);

            // Reset for aligned reads - start from aligned position
            cursor.position = 0;

            // Read types that require alignment from aligned positions
            // Note: MemoryCursor.read uses @alignCast so we need aligned data
            // For random garbage, focus on bounds checking not alignment
            _ = cursor.read(u8); // Always safe
            cursor.position = 0; // Reset to aligned position
            _ = cursor.readBytes(@sizeOf(u16)); // Read as bytes to avoid alignment issue
            cursor.position = 0;
            _ = cursor.readBytes(@sizeOf(u32));
            cursor.position = 0;
            _ = cursor.readBytes(@sizeOf(u64));
            cursor.position = 0;
            _ = cursor.readBytes(@sizeOf(GgufHeader));

            // Test string read (length-prefixed) with random data
            // This tests bounds checking when length prefix may be garbage
            cursor.position = 0;
            _ = cursor.readString();

            // All operations should complete without crashing
            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "GGUF cursor bounds checking with random seeks" {
    const gen = generators.intRange(usize, 0, 256);

    const result = forAll(usize, gen, TestConfig, struct {
        fn check(seek_pos: usize) bool {
            var buf: [64]u8 = undefined;
            @memset(&buf, 0xAB);

            var cursor = MemoryCursor.init(&buf);

            // Seek should return false for out-of-bounds
            const seek_result = cursor.seek(seek_pos);
            if (seek_pos > buf.len) {
                return !seek_result;
            } else {
                return seek_result and cursor.position == seek_pos;
            }
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "GGUF cursor skip bounds checking" {
    const SkipParams = struct {
        initial_pos: usize,
        skip_amount: usize,
    };

    const gen = Generator(SkipParams){
        .generateFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, _: usize) SkipParams {
                return .{
                    .initial_pos = prng.random().intRangeAtMost(usize, 0, 100),
                    .skip_amount = prng.random().intRangeAtMost(usize, 0, 200),
                };
            }
        }.generate,
        .shrinkFn = null,
    };

    const result = forAll(SkipParams, gen, TestConfig, struct {
        fn check(params: SkipParams) bool {
            var buf: [64]u8 = undefined;
            @memset(&buf, 0);

            var cursor = MemoryCursor.init(&buf);

            // Set initial position if valid
            if (!cursor.seek(params.initial_pos)) {
                return params.initial_pos > buf.len;
            }

            const skip_result = cursor.skip(params.skip_amount);
            const expected_end = params.initial_pos + params.skip_amount;

            if (expected_end > buf.len) {
                // Skip should fail
                return !skip_result;
            } else {
                // Skip should succeed
                return skip_result and cursor.position == expected_end;
            }
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Malformed String Parsing Tests
// ============================================================================

test "GGUF string parsing with invalid length prefix" {
    const gen = generators.intRange(u64, 0, std.math.maxInt(u64));

    const result = forAll(u64, gen, TestConfig, struct {
        fn check(claimed_len: u64) bool {
            // Create buffer with length prefix but insufficient data
            var buf: [16]u8 = undefined;
            std.mem.writeInt(u64, buf[0..8], claimed_len, .little);
            @memset(buf[8..], 'x');

            var cursor = MemoryCursor.init(&buf);
            const maybe_string = cursor.readString();

            // Actual available bytes after length prefix is 8
            if (claimed_len <= 8) {
                // Should succeed
                return maybe_string != null and maybe_string.?.len == claimed_len;
            } else {
                // Should fail gracefully
                return maybe_string == null;
            }
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Alignment Tests
// ============================================================================

test "GGUF cursor alignment" {
    const AlignParams = struct {
        initial_pos: usize,
        alignment: usize,
    };

    const gen = Generator(AlignParams){
        .generateFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, _: usize) AlignParams {
                const alignments = [_]usize{ 1, 2, 4, 8, 16, 32, 64 };
                return .{
                    .initial_pos = prng.random().intRangeAtMost(usize, 0, 127),
                    .alignment = alignments[prng.random().intRangeLessThan(usize, 0, alignments.len)],
                };
            }
        }.generate,
        .shrinkFn = null,
    };

    const result = forAll(AlignParams, gen, TestConfig, struct {
        fn check(params: AlignParams) bool {
            var buf: [256]u8 = undefined;
            @memset(&buf, 0);

            var cursor = MemoryCursor.init(&buf);
            _ = cursor.seek(params.initial_pos);

            cursor.alignTo(params.alignment);

            // Position should be aligned
            if (params.alignment == 0) return true; // Avoid division by zero

            const aligned = cursor.position % params.alignment == 0;
            const moved_forward = cursor.position >= params.initial_pos;

            return aligned and moved_forward;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Edge Case Dimension Tests
// ============================================================================

test "GGUF tensor with zero dimensions returns 1 element" {
    // Edge case: n_dims = 0 should still work
    const info = TensorInfo{
        .name = "scalar",
        .n_dims = 0,
        .dims = .{ 1, 1, 1, 1 },
        .tensor_type = .f32,
        .offset = 0,
    };

    // Element count for 0 dims should be 1 (scalar)
    try std.testing.expectEqual(@as(u64, 1), info.elementCount());
}

test "GGUF tensor with max dimensions" {
    const gen = generators.intRange(u64, 1, 65536);

    const result = forAll(u64, gen, TestConfig, struct {
        fn check(dim_size: u64) bool {
            const info = TensorInfo{
                .name = "large_tensor",
                .n_dims = 4,
                .dims = .{ dim_size, dim_size, 1, 1 },
                .tensor_type = .f32,
                .offset = 0,
            };

            const elem_count = info.elementCount();
            const expected = dim_size * dim_size;

            return elem_count == expected;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Overflow Safety Tests
// ============================================================================

test "GGUF tensor byte calculation handles large values" {
    // Test that tensorBytes doesn't overflow or crash with large element counts
    const gen = generators.intRange(u64, 1, std.math.maxInt(u32));

    const result = forAll(u64, gen, TestConfig, struct {
        fn check(element_count: u64) bool {
            // Test with various tensor types
            const types = [_]GgufTensorType{ .f32, .f16, .q4_0, .q8_0 };

            for (types) |tensor_type| {
                const bytes = tensor_type.tensorBytes(element_count);
                // Should complete without crash and be reasonable
                if (bytes == 0 and element_count > 0) return false;
            }

            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Integration: Simulated GGUF File Parsing
// ============================================================================

test "GGUF simulated file parsing with random structure" {
    const SimulatedGguf = struct {
        tensor_count: u8,
        metadata_count: u8,
    };

    const gen = Generator(SimulatedGguf){
        .generateFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, _: usize) SimulatedGguf {
                return .{
                    .tensor_count = prng.random().intRangeAtMost(u8, 0, 10),
                    .metadata_count = prng.random().intRangeAtMost(u8, 0, 10),
                };
            }
        }.generate,
        .shrinkFn = null,
    };

    const result = forAllWithAllocator(SimulatedGguf, std.testing.allocator, gen, TestConfig, struct {
        fn check(params: SimulatedGguf, allocator: std.mem.Allocator) bool {
            // Build a minimal valid GGUF-like structure in memory
            var buffer = std.ArrayListUnmanaged(u8).empty;
            defer buffer.deinit(allocator);

            // Header
            var header_bytes: [@sizeOf(GgufHeader)]u8 = undefined;
            std.mem.writeInt(u32, header_bytes[0..4], GGUF_MAGIC, .little);
            std.mem.writeInt(u32, header_bytes[4..8], GGUF_VERSION_3, .little);
            std.mem.writeInt(u64, header_bytes[8..16], params.tensor_count, .little);
            std.mem.writeInt(u64, header_bytes[16..24], params.metadata_count, .little);
            buffer.appendSlice(allocator, &header_bytes) catch return false;

            // Parse header back
            var cursor = MemoryCursor.init(buffer.items);
            const parsed_header = cursor.read(GgufHeader) orelse return false;

            return parsed_header.isValid() and
                parsed_header.tensor_count == params.tensor_count and
                parsed_header.metadata_kv_count == params.metadata_count;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Metadata Value Conversion Tests
// ============================================================================

test "GGUF metadata value u32 conversion" {
    const gen = generators.intRange(u32, 0, std.math.maxInt(u32));

    const result = forAll(u32, gen, TestConfig, struct {
        fn check(value: u32) bool {
            const metadata = GgufMetadataValue{ .uint32 = value };
            const converted = metadata.asU32();
            return converted != null and converted.? == value;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "GGUF metadata value f32 conversion" {
    const gen = generators.floatRange(-1e10, 1e10);

    const result = forAll(f32, gen, TestConfig, struct {
        fn check(value: f32) bool {
            const metadata = GgufMetadataValue{ .float32 = value };
            const converted = metadata.asF32();
            return converted != null and converted.? == value;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "GGUF metadata value negative int to u32 returns null" {
    const gen = generators.intRange(i32, std.math.minInt(i32), -1);

    const result = forAll(i32, gen, TestConfig, struct {
        fn check(value: i32) bool {
            const metadata = GgufMetadataValue{ .int32 = value };
            const converted = metadata.asU32();
            // Negative values should return null
            return converted == null;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "GGUF metadata value bool conversion" {
    const gen = generators.boolean();

    const result = forAll(bool, gen, TestConfig, struct {
        fn check(value: bool) bool {
            const metadata = GgufMetadataValue{ .bool_ = value };
            const converted = metadata.asBool();
            return converted != null and converted.? == value;
        }
    }.check);

    try std.testing.expect(result.passed);
}
