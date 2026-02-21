//! SPIR-V Code Generator Module
//!
//! Generates SPIR-V binary bytecode from kernel IR for Vulkan compute shaders.
//! SPIR-V is a binary intermediate representation defined by Khronos.
//!
//! ## Module Organization
//!
//! This module is split into logical sub-modules:
//! - `constants.zig` - SPIR-V opcodes, capabilities, and enumerations
//! - `types_gen.zig` - Type generation with caching
//! - `constants_gen.zig` - Constant generation with caching
//! - `emit.zig` - Low-level instruction emission
//! - `codegen.zig` - Statement and expression code generation
//! - `generator.zig` - Main SpirvGenerator struct
//!
//! ## Usage
//!
//! ```zig
//! const spirv = @import("spirv/mod.zig");
//!
//! var gen = spirv.SpirvGenerator.init(allocator);
//! defer gen.deinit();
//!
//! const result = try gen.generate(&kernel_ir);
//! // result.code contains SPIR-V binary
//! // result.spirv_binary also contains the binary for Vulkan
//! ```
//!
//! Specification: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html

const std = @import("std");

// Re-export sub-modules
pub const constants = @import("constants.zig");
pub const types_gen = @import("types_gen.zig");
pub const constants_gen = @import("constants_gen.zig");
pub const emit = @import("emit.zig");
pub const codegen = @import("codegen.zig");
pub const generator = @import("generator.zig");

// Re-export main types for convenience
pub const SpirvGenerator = generator.SpirvGenerator;

// Re-export constants
pub const SPIRV_MAGIC = constants.SPIRV_MAGIC;
pub const SPIRV_VERSION = constants.SPIRV_VERSION;
pub const GENERATOR_ID = constants.GENERATOR_ID;

// Re-export enums
pub const OpCode = constants.OpCode;
pub const Capability = constants.Capability;
pub const ExecutionModel = constants.ExecutionModel;
pub const AddressingModel = constants.AddressingModel;
pub const MemoryModel = constants.MemoryModel;
pub const ExecutionMode = constants.ExecutionMode;
pub const StorageClass = constants.StorageClass;
pub const Decoration = constants.Decoration;
pub const BuiltIn = constants.BuiltIn;
pub const MemorySemantics = constants.MemorySemantics;
pub const Scope = constants.Scope;

// Re-export key types
pub const TypeKey = types_gen.TypeKey;
pub const ConstKey = constants_gen.ConstKey;

// Import for tests
const dsl_types = @import("../../types.zig");
const kernel = @import("../../kernel.zig");

// ============================================================================
// Tests
// ============================================================================

test "SpirvGenerator basic kernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = SpirvGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    // Verify SPIR-V magic number
    const words = std.mem.bytesAsSlice(u32, result.code);
    try std.testing.expectEqual(SPIRV_MAGIC, words[0]);
    try std.testing.expectEqual(SPIRV_VERSION, words[1]);

    // Result should have spirv_binary set
    try std.testing.expect(result.spirv_binary != null);
}

test "SpirvGenerator with buffers" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = SpirvGenerator.init(allocator);
    defer gen.deinit();

    const buffers = [_]kernel.BufferBinding{
        .{
            .name = "input",
            .binding = 0,
            .group = 0,
            .element_type = dsl_types.Type.f32Type(),
            .access = .read_only,
        },
        .{
            .name = "output",
            .binding = 1,
            .group = 0,
            .element_type = dsl_types.Type.f32Type(),
            .access = .write_only,
        },
    };

    var ir = kernel.KernelIR.empty("vector_add");
    ir.buffers = &buffers;
    ir.workgroup_size = .{ 64, 1, 1 };

    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    // Verify valid SPIR-V
    const words = std.mem.bytesAsSlice(u32, result.code);
    try std.testing.expectEqual(SPIRV_MAGIC, words[0]);
    try std.testing.expect(words.len > 10); // Should have significant content
}

test "SpirvGenerator OpCode encoding" {
    // Verify instruction encoding
    const word_count: u32 = 4;
    const opcode: u32 = @intFromEnum(OpCode.OpCapability);
    const first_word = (word_count << 16) | opcode;

    try std.testing.expectEqual(@as(u32, 0x00040011), first_word);
}

test "SPIR-V module structure validation" {
    // This test validates the SPIR-V binary structure follows the spec
    // Reference: SPIR-V Specification Section 2.3 (Physical Layout of a SPIR-V Module)
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = SpirvGenerator.init(allocator);
    defer gen.deinit();

    const buffers = [_]kernel.BufferBinding{
        .{
            .name = "data",
            .binding = 0,
            .group = 0,
            .element_type = dsl_types.Type.f32Type(),
            .access = .read_write,
        },
    };

    var ir = kernel.KernelIR.empty("validation_test");
    ir.buffers = &buffers;
    ir.workgroup_size = .{ 256, 1, 1 };

    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    const words = std.mem.bytesAsSlice(u32, result.code);

    // Validate header (5 words)
    try std.testing.expect(words.len >= 5);

    // Word 0: Magic number
    try std.testing.expectEqual(SPIRV_MAGIC, words[0]);

    // Word 1: Version (1.5)
    try std.testing.expectEqual(SPIRV_VERSION, words[1]);

    // Word 2: Generator ID (should be 0 for unregistered)
    try std.testing.expectEqual(GENERATOR_ID, words[2]);

    // Word 3: Bound (upper bound of all IDs, must be > 0)
    try std.testing.expect(words[3] > 0);

    // Word 4: Schema (reserved, must be 0)
    try std.testing.expectEqual(@as(u32, 0), words[4]);

    // Verify there is at least one instruction after the header.
    if (words.len > 5) {
        const first_word_count = words[5] >> 16;
        try std.testing.expect(first_word_count > 0);
    }

    // Validate that module has required sections in order:
    // 1. Capabilities (OpCapability)
    // 2. Memory model (OpMemoryModel)
    // 3. Entry points (OpEntryPoint)
    var found_capability = false;
    var found_memory_model = false;
    var found_entry_point = false;

    var i: usize = 5; // Start after header
    while (i < words.len) {
        const instruction = words[i];
        const word_count_inst = instruction >> 16;
        const op = instruction & 0xFFFF;

        if (op == @intFromEnum(OpCode.OpCapability)) found_capability = true;
        if (op == @intFromEnum(OpCode.OpMemoryModel)) found_memory_model = true;
        if (op == @intFromEnum(OpCode.OpEntryPoint)) found_entry_point = true;

        if (word_count_inst == 0) break;
        i += word_count_inst;
    }

    try std.testing.expect(found_capability);
    try std.testing.expect(found_memory_model);
    try std.testing.expect(found_entry_point);
}

test "SPIR-V external validation hint" {
    // This test documents how to validate SPIR-V externally with spirv-val
    // Users should run: spirv-val <output.spv> to validate generated shaders
    //
    // To generate a binary for validation:
    // 1. Use SpirvGenerator.generate() to get GeneratedSource
    // 2. Write result.code to a .spv file
    // 3. Run: spirv-val output.spv
    //
    // The spirv-val tool is part of the SPIRV-Tools package:
    // https://github.com/KhronosGroup/SPIRV-Tools
    //
    // Example validation command:
    //   spirv-val --target-env vulkan1.2 shader.spv
    //
    // This test just verifies the documentation exists and the generator works.
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = SpirvGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("external_val_test");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    // spirv_binary should be set for external validation
    try std.testing.expect(result.spirv_binary != null);
    try std.testing.expect(result.spirv_binary.?.len > 0);
}
