//! SPIR-V Code Generator
//!
//! Generates SPIR-V binary bytecode from kernel IR for Vulkan compute shaders.
//! SPIR-V is a binary intermediate representation defined by Khronos.
//!
//! ## Module Organization
//!
//! This file re-exports from the `spirv/` subdirectory for backward compatibility.
//! The implementation is split into logical modules:
//! - `spirv/constants.zig` - SPIR-V opcodes, capabilities, and enumerations
//! - `spirv/types_gen.zig` - Type generation with caching
//! - `spirv/constants_gen.zig` - Constant generation with caching
//! - `spirv/emit.zig` - Low-level instruction emission
//! - `spirv/codegen.zig` - Statement and expression code generation
//! - `spirv/generator.zig` - Main SpirvGenerator struct
//!
//! Specification: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html

const std = @import("std");
const types = @import("../types.zig");
const expr = @import("../expr.zig");
const stmt = @import("../stmt.zig");
const kernel = @import("../kernel.zig");
const backend = @import("backend.zig");
const gpu_backend = @import("../../backend.zig");

// Import the modular implementation
const spirv = @import("spirv/mod.zig");

// Re-export constants for backward compatibility
pub const SPIRV_MAGIC = spirv.SPIRV_MAGIC;
pub const SPIRV_VERSION = spirv.SPIRV_VERSION;
pub const GENERATOR_ID = spirv.GENERATOR_ID;

// Re-export enums for backward compatibility
pub const OpCode = spirv.OpCode;
pub const Capability = spirv.Capability;
pub const ExecutionModel = spirv.ExecutionModel;
pub const AddressingModel = spirv.AddressingModel;
pub const MemoryModel = spirv.MemoryModel;
pub const ExecutionMode = spirv.ExecutionMode;
pub const StorageClass = spirv.StorageClass;
pub const Decoration = spirv.Decoration;
pub const BuiltIn = spirv.BuiltIn;
pub const MemorySemantics = spirv.MemorySemantics;
pub const Scope = spirv.Scope;

// Re-export the main generator for backward compatibility
pub const SpirvGenerator = spirv.SpirvGenerator;

// Re-export sub-modules
pub const constants = spirv.constants;
pub const types_gen = spirv.types_gen;
pub const constants_gen = spirv.constants_gen;
pub const emit = spirv.emit;
pub const codegen = spirv.codegen;
pub const generator = spirv.generator;

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
            .element_type = types.Type.f32Type(),
            .access = .read_only,
        },
        .{
            .name = "output",
            .binding = 1,
            .group = 0,
            .element_type = types.Type.f32Type(),
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
            .element_type = types.Type.f32Type(),
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

    // Verify first instruction after header is OpCapability (opcode 17)
    if (words.len > 5) {
        const first_op = words[5] & 0xFFFF;
        try std.testing.expectEqual(@as(u32, @intFromEnum(OpCode.OpCapability)), first_op);
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
