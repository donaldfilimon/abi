//! GPU Kernel DSL Module
//!
//! This module provides a portable kernel DSL (Domain-Specific Language) for writing
//! GPU compute kernels that auto-translate to CUDA, Vulkan/GLSL, WebGPU/WGSL, and Metal/MSL.
//!
//! ## Overview
//!
//! The DSL provides:
//! - **Type System**: Scalar, vector, matrix types with address space qualifiers
//! - **Expression AST**: Binary ops, unary ops, function calls, indexing, swizzles
//! - **Statement AST**: Declarations, assignments, control flow, barriers
//! - **Kernel Builder**: Fluent API for constructing kernel IR
//! - **Code Generators**: CUDA, GLSL, WGSL, MSL backends
//! - **Unified Compiler**: Compiles IR to any supported backend
//!
//! ## Quick Start
//!
//! ```zig
//! const dsl = @import("gpu/dsl/mod.zig");
//!
//! // Create a kernel builder
//! var builder = dsl.KernelBuilder.init(allocator, "vector_add");
//! defer builder.deinit();
//!
//! // Add buffer bindings
//! const a = try builder.addBuffer("a", dsl.types.Type.f32Type(), .read_only);
//! const b = try builder.addBuffer("b", dsl.types.Type.f32Type(), .read_only);
//! const c = try builder.addBuffer("c", dsl.types.Type.f32Type(), .write_only);
//! const n = try builder.addUniform("n", dsl.types.Type.u32Type());
//!
//! // Build kernel logic
//! const gid = builder.globalInvocationId();
//! const idx = try gid.x();
//! const condition = try builder.lt(idx, try n.toExpr());
//! const sum = try builder.add(try a.at(idx), try b.at(idx));
//! try builder.ifStmt(condition, &[_]*const dsl.stmt.Stmt{
//!     try builder.assignStmt(try c.at(idx), sum)
//! }, null);
//!
//! // Build IR
//! const ir = try builder.build();
//!
//! // Compile to any backend
//! var cuda_src = try dsl.compiler.compile(allocator, &ir, .cuda, .{});
//! defer cuda_src.deinit(allocator);
//! ```

const std = @import("std");

// ============================================================================
// Core Type System
// ============================================================================

/// Type system for the kernel DSL.
pub const types = @import("types.zig");

/// Scalar types (bool, i8-i64, u8-u64, f16-f64).
pub const ScalarType = types.ScalarType;

/// Vector types (vec2, vec3, vec4).
pub const VectorType = types.VectorType;

/// Matrix types (mat2-mat4).
pub const MatrixType = types.MatrixType;

/// Address spaces (private, workgroup, storage, uniform).
pub const AddressSpace = types.AddressSpace;

/// Complete type representation.
pub const Type = types.Type;

/// Access mode for buffer parameters.
pub const AccessMode = types.AccessMode;

// ============================================================================
// Expression AST
// ============================================================================

/// Expression AST nodes.
pub const expr = @import("expr.zig");

/// Value reference.
pub const ValueRef = expr.ValueRef;

/// Unary operations.
pub const UnaryOp = expr.UnaryOp;

/// Binary operations.
pub const BinaryOp = expr.BinaryOp;

/// Built-in functions.
pub const BuiltinFn = expr.BuiltinFn;

/// Literal values.
pub const Literal = expr.Literal;

/// Expression node.
pub const Expr = expr.Expr;

/// Built-in variables.
pub const BuiltinVar = expr.BuiltinVar;

// ============================================================================
// Statement AST
// ============================================================================

/// Statement AST nodes.
pub const stmt = @import("stmt.zig");

/// Statement node.
pub const Stmt = stmt.Stmt;

// Statement creation helpers
pub const varDecl = stmt.varDecl;
pub const constDecl = stmt.constDecl;
pub const assign = stmt.assign;
pub const compoundAssign = stmt.compoundAssign;
pub const ifStmt = stmt.ifStmt;
pub const forLoop = stmt.forLoop;
pub const whileLoop = stmt.whileLoop;
pub const returnStmt = stmt.returnStmt;
pub const breakStmt = stmt.breakStmt;
pub const continueStmt = stmt.continueStmt;
pub const exprStmt = stmt.exprStmt;
pub const block = stmt.block;

// ============================================================================
// Kernel IR
// ============================================================================

/// Kernel IR definitions.
pub const kernel = @import("kernel.zig");

/// Buffer binding descriptor.
pub const BufferBinding = kernel.BufferBinding;

/// Uniform binding descriptor.
pub const UniformBinding = kernel.UniformBinding;

/// Push constant descriptor.
pub const PushConstant = kernel.PushConstant;

/// Shared memory declaration.
pub const SharedMemory = kernel.SharedMemory;

/// Helper function definition.
pub const HelperFunction = kernel.HelperFunction;

/// Complete kernel IR.
pub const KernelIR = kernel.KernelIR;

/// Feature flags.
pub const FeatureFlags = kernel.FeatureFlags;

/// Validation result.
pub const ValidationResult = kernel.ValidationResult;

/// Portable kernel source.
pub const PortableKernelSource = kernel.PortableKernelSource;

/// Built-in kernel types.
pub const BuiltinKernel = kernel.BuiltinKernel;

// ============================================================================
// Kernel Builder
// ============================================================================

/// Kernel builder module.
pub const builder = @import("builder.zig");

/// Kernel builder for constructing IR.
pub const KernelBuilder = builder.KernelBuilder;

/// Value wrapper for tracked values.
pub const Value = builder.Value;

// ============================================================================
// Code Generators
// ============================================================================

/// Code generator backend interface.
pub const codegen = struct {
    pub const backend = @import("codegen/backend.zig");
    pub const common = @import("codegen/common.zig");
    pub const generic = @import("codegen/generic.zig");
    pub const spirv = @import("codegen/spirv.zig");
    const vision = @import("codegen/vision_kernels.zig");

    // Language-specific namespaces (inlined from former wrapper files)
    pub const cuda = struct {
        pub const Generator = generic.CudaGenerator;
        pub const CudaGenerator = generic.CudaGenerator;
        pub const VisionKernels = vision.VisionKernels;

        test "CudaGenerator availability" {
            const allocator = std.testing.allocator;
            var g = Generator.init(allocator);
            defer g.deinit();
            try std.testing.expect(@TypeOf(g).backend_config.language == .cuda);
        }
    };
    pub const glsl = struct {
        pub const Generator = generic.GlslGenerator;
        pub const GlslGenerator = generic.GlslGenerator;
        pub const VisionKernels = vision.VisionKernels;

        test "GlslGenerator availability" {
            const allocator = std.testing.allocator;
            var g = Generator.init(allocator);
            defer g.deinit();
            try std.testing.expect(@TypeOf(g).backend_config.language == .glsl);
        }
    };
    pub const wgsl = struct {
        pub const Generator = generic.WgslGenerator;
        pub const WgslGenerator = generic.WgslGenerator;
        pub const VisionKernels = vision.VisionKernels;

        test "WgslGenerator availability" {
            const allocator = std.testing.allocator;
            var g = Generator.init(allocator);
            defer g.deinit();
            try std.testing.expect(@TypeOf(g).backend_config.language == .wgsl);
        }
    };
    pub const msl = struct {
        pub const Generator = generic.MslGenerator;
        pub const MslGenerator = generic.MslGenerator;
        pub const VisionKernels = vision.VisionKernels;

        test "MslGenerator availability" {
            const allocator = std.testing.allocator;
            var g = Generator.init(allocator);
            defer g.deinit();
            try std.testing.expect(@TypeOf(g).backend_config.language == .msl);
        }
    };

    // Re-export common types
    pub const CodegenError = backend.CodegenError;
    pub const GeneratedSource = backend.GeneratedSource;
    pub const CodeGenerator = backend.CodeGenerator;
    pub const BackendCapabilities = backend.BackendCapabilities;
    pub const validateForBackend = backend.validateForBackend;

    // SPIR-V types
    pub const SpirVGenerator = spirv.SpirVGenerator;
    pub const ShaderCache = spirv.ShaderCache;
};

/// Codegen error type.
pub const CodegenError = codegen.CodegenError;

/// Generated source code.
pub const GeneratedSource = codegen.GeneratedSource;

// ============================================================================
// Optimizer (MLIR/LLVM-inspired)
// ============================================================================

/// Kernel IR optimizer module.
pub const optimizer = @import("optimizer.zig");

/// Optimizer for kernel IR.
pub const Optimizer = optimizer.Optimizer;

/// Optimization pass types.
pub const OptimizationPass = optimizer.OptimizationPass;

/// Optimization level presets.
pub const OptimizationLevel = optimizer.OptimizationLevel;

/// Optimization statistics.
pub const OptimizationStats = optimizer.OptimizationStats;

// ============================================================================
// Compiler
// ============================================================================

/// Unified compiler module.
pub const compiler = @import("compiler.zig");

/// Compile error type.
pub const CompileError = compiler.CompileError;

/// Compile options.
pub const CompileOptions = compiler.CompileOptions;

/// Compile kernel IR to a specific backend.
pub const compile = compiler.compile;

/// Compile kernel IR to KernelSource.
pub const compileToKernelSource = compiler.compileToKernelSource;

/// Compile to all available backends.
pub const compileAll = compiler.compileAll;

/// Get the best available backend.
pub const getBestBackend = compiler.getBestBackend;

/// Check if backend supports compilation.
pub const backendSupportsCompilation = compiler.backendSupportsCompilation;

/// Zig-to-SPIRV compiler integration.
pub const zig_spirv = @import("spirv.zig");

// ============================================================================
// Tests
// ============================================================================

test {
    // Run all subtests
    _ = types;
    _ = expr;
    _ = stmt;
    _ = kernel;
    _ = builder;
    _ = codegen.backend;
    _ = codegen.common;
    _ = codegen.cuda;
    _ = codegen.glsl;
    _ = codegen.wgsl;
    _ = codegen.msl;
    _ = codegen.spirv;
    _ = compiler;
    _ = zig_spirv;
}

test "DSL module exports" {
    // Verify all exports are accessible
    _ = ScalarType.f32;
    _ = BinaryOp.add;
    _ = UnaryOp.sqrt;
    _ = BuiltinFn.barrier;
    _ = AccessMode.read_only;
    _ = BuiltinKernel.vector_add;
}
