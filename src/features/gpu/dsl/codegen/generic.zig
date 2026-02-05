//! Generic GPU Code Generator
//!
//! A comptime-generic code generator template that uses backend configuration
//! to produce output for any target shading language. This eliminates code
//! duplication across GLSL, WGSL, MSL, CUDA, and other backends.

const std = @import("std");
const types = @import("../types.zig");
const expr = @import("../expr.zig");
const stmt = @import("../stmt.zig");
const kernel = @import("../kernel.zig");
const backend = @import("backend.zig");
const common = @import("common.zig");
const gpu_backend = @import("../../backend.zig");
const configs = @import("configs/mod.zig");

/// Create a code generator type for the given backend configuration.
pub fn CodeGenerator(comptime Config: type) type {
    return struct {
        writer: common.CodeWriter,
        allocator: std.mem.Allocator,
        config: *const configs.BackendConfig,

        const Self = @This();

        /// The backend configuration used by this generator.
        pub const backend_config: configs.BackendConfig = Config.config;

        // Comptime-generated lookup tables for O(1) access
        // These are computed once at compile time for each backend instantiation.
        pub const unary_fn_table: configs.UnaryFunctions.LookupTable = backend_config.unary_fns.buildTable();
        pub const binary_fn_table: configs.BinaryFunctions.LookupTable = backend_config.binary_fns.buildTable();
        pub const type_name_table: configs.TypeNames.LookupTable = backend_config.type_names.buildTable();
        pub const vector_prefix_table: configs.VectorNaming.LookupTable = backend_config.vector_naming.buildTable();

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .writer = common.CodeWriter.init(allocator),
                .allocator = allocator,
                .config = &backend_config,
            };
        }

        pub fn deinit(self: *Self) void {
            self.writer.deinit();
        }

        /// Generate source code from kernel IR.
        pub fn generate(
            self: *Self,
            ir: *const kernel.KernelIR,
        ) backend.CodegenError!backend.GeneratedSource {
            try self.writeHeader(ir);
            try self.writeSharedMemory(ir);
            try self.writeHelpers();
            try self.writeKernelSignature(ir);
            try self.writeBuiltinVars(ir);
            try self.writeBody(ir);
            try self.writeKernelClose();

            const code = try self.writer.getCode();
            const entry_point = try self.allocator.dupe(u8, ir.entry_point);

            return .{
                .code = code,
                .entry_point = entry_point,
                .backend = self.getGpuBackend(),
                .language = self.getLanguage(),
            };
        }

        fn getGpuBackend(self: *const Self) gpu_backend.Backend {
            return switch (self.config.language) {
                .glsl => .vulkan,
                .wgsl => .webgpu,
                .msl => .metal,
                .cuda => .cuda,
                .spirv => .vulkan,
                .hlsl => .vulkan,
            };
        }

        fn getLanguage(self: *const Self) backend.GeneratedSource.Language {
            return switch (self.config.language) {
                .glsl => .glsl,
                .wgsl => .wgsl,
                .msl => .msl,
                .cuda => .cuda,
                .spirv => .spirv,
                .hlsl => .hlsl,
            };
        }

        // ============================================================
        // Header Generation
        // ============================================================

        fn writeHeader(self: *Self, ir: *const kernel.KernelIR) !void {
            try self.writer.writeLine("// Auto-generated compute shader");
            try self.writer.writeFmt("// Kernel: {s}\n", .{ir.name});
            try self.writer.newline();

            // Language-specific headers
            switch (self.config.language) {
                .glsl => try self.writeGlslHeader(),
                .wgsl => {}, // WGSL needs no header
                .msl => try self.writeMslHeader(),
                .cuda => try self.writeCudaHeader(),
                else => {},
            }
        }

        fn writeGlslHeader(self: *Self) !void {
            try self.writer.writeLine("#version 450");
            try self.writer.writeLine("#extension GL_ARB_separate_shader_objects : enable");
            try self.writer.newline();
        }

        fn writeMslHeader(self: *Self) !void {
            try self.writer.writeLine("#include <metal_stdlib>");
            try self.writer.writeLine("using namespace metal;");
            try self.writer.newline();
        }

        fn writeCudaHeader(self: *Self) !void {
            try self.writer.writeLine("#include <cuda_runtime.h>");
            try self.writer.writeLine("#include <stdint.h>");
            try self.writer.newline();
        }

        // ============================================================
        // Helpers
        // ============================================================

        fn writeHelpers(self: *Self) !void {
            // CUDA needs clamp helper
            if (self.config.language == .cuda) {
                try self.writer.writeLine("#ifndef CLAMP_DEFINED");
                try self.writer.writeLine("#define CLAMP_DEFINED");
                try self.writer.writeLine("__device__ __forceinline__ float clamp(float x, float lo, float hi) {");
                try self.writer.writeLine("    return fminf(fmaxf(x, lo), hi);");
                try self.writer.writeLine("}");
                try self.writer.writeLine("#endif");
                try self.writer.newline();
            }
        }

        // ============================================================
        // Shared Memory
        // ============================================================

        fn writeSharedMemory(self: *Self, ir: *const kernel.KernelIR) !void {
            if (ir.shared_memory.len == 0) return;

            for (ir.shared_memory) |shared| {
                try self.writer.writeIndent();
                try self.writeSharedMemoryDecl(shared);
            }
            try self.writer.newline();
        }

        fn writeSharedMemoryDecl(self: *Self, shared: kernel.SharedMemory) !void {
            switch (self.config.language) {
                .glsl => {
                    try self.writer.write("shared ");
                    try self.writeType(shared.element_type);
                    if (shared.size) |size| {
                        try self.writer.writeFmt(" {s}[{d}];\n", .{ shared.name, size });
                    } else {
                        try self.writer.writeFmt(" {s}[];\n", .{shared.name});
                    }
                },
                .wgsl => {
                    try self.writer.write("var<workgroup> ");
                    try self.writer.writeFmt("{s}: array<", .{shared.name});
                    try self.writeType(shared.element_type);
                    if (shared.size) |size| {
                        try self.writer.writeFmt(", {d}>;\n", .{size});
                    } else {
                        try self.writer.write(">;\n");
                    }
                },
                .msl => {
                    try self.writer.write("threadgroup ");
                    try self.writeType(shared.element_type);
                    if (shared.size) |size| {
                        try self.writer.writeFmt(" {s}[{d}];\n", .{ shared.name, size });
                    } else {
                        try self.writer.writeFmt(" {s}[];\n", .{shared.name});
                    }
                },
                .cuda => {
                    if (shared.size) |size| {
                        try self.writer.write("__shared__ ");
                        try self.writeType(shared.element_type);
                        try self.writer.writeFmt(" {s}[{d}];\n", .{ shared.name, size });
                    } else {
                        try self.writer.write("extern __shared__ ");
                        try self.writeType(shared.element_type);
                        try self.writer.writeFmt(" {s}[];\n", .{shared.name});
                    }
                },
                else => {},
            }
        }

        // ============================================================
        // Kernel Signature
        // ============================================================

        fn writeKernelSignature(self: *Self, ir: *const kernel.KernelIR) !void {
            switch (self.config.language) {
                .glsl => try self.writeGlslSignature(ir),
                .wgsl => try self.writeWgslSignature(ir),
                .msl => try self.writeMslSignature(ir),
                .cuda => try self.writeCudaSignature(ir),
                else => {},
            }
        }

        fn writeGlslSignature(self: *Self, ir: *const kernel.KernelIR) !void {
            // Layout declarations
            try self.writer.writeFmt("layout(local_size_x = {d}, local_size_y = {d}, local_size_z = {d}) in;\n", .{
                ir.workgroup_size[0],
                ir.workgroup_size[1],
                ir.workgroup_size[2],
            });
            try self.writer.newline();

            // Buffer declarations
            for (ir.buffers, 0..) |buf, i| {
                try self.writer.writeFmt("layout(set = 0, binding = {d}) buffer Buffer{d} {{\n", .{ buf.binding, i });
                self.writer.indent();
                try self.writer.writeIndent();
                try self.writeType(buf.element_type);
                try self.writer.writeFmt(" {s}[];\n", .{buf.name});
                self.writer.dedent();
                try self.writer.writeLine("};");
            }

            // Uniform declarations
            if (ir.uniforms.len > 0) {
                try self.writer.writeLine("layout(push_constant) uniform PushConstants {");
                self.writer.indent();
                for (ir.uniforms) |uni| {
                    try self.writer.writeIndent();
                    try self.writeType(uni.ty);
                    try self.writer.writeFmt(" {s};\n", .{uni.name});
                }
                self.writer.dedent();
                try self.writer.writeLine("} uniforms;");
            }

            try self.writer.newline();
            try self.writer.writeLine("void main() {");
            self.writer.indent();
        }

        fn writeWgslSignature(self: *Self, ir: *const kernel.KernelIR) !void {
            // Struct for uniforms
            if (ir.uniforms.len > 0) {
                try self.writer.writeLine("struct Uniforms {");
                self.writer.indent();
                for (ir.uniforms) |uni| {
                    try self.writer.writeIndent();
                    try self.writer.writeFmt("{s}: ", .{uni.name});
                    try self.writeType(uni.ty);
                    try self.writer.write(",\n");
                }
                self.writer.dedent();
                try self.writer.writeLine("}");
                try self.writer.writeLine("@group(0) @binding(0) var<uniform> uniforms: Uniforms;");
                try self.writer.newline();
            }

            // Buffer bindings
            for (ir.buffers, 0..) |buf, i| {
                const access = if (buf.access == .read_only) "read" else "read_write";
                try self.writer.writeFmt("@group(0) @binding({d}) var<storage, {s}> {s}: array<", .{ buf.binding, access, buf.name });
                try self.writeType(buf.element_type);
                try self.writer.write(">;\n");
                _ = i;
            }

            try self.writer.newline();
            try self.writer.writeFmt("@compute @workgroup_size({d}, {d}, {d})\n", .{
                ir.workgroup_size[0],
                ir.workgroup_size[1],
                ir.workgroup_size[2],
            });
            try self.writer.writeFmt("fn {s}(\n", .{ir.entry_point});
            self.writer.indent();
            try self.writer.writeLine("@builtin(global_invocation_id) globalInvocationId: vec3<u32>,");
            try self.writer.writeLine("@builtin(local_invocation_id) localInvocationId: vec3<u32>,");
            try self.writer.writeLine("@builtin(workgroup_id) workgroupId: vec3<u32>,");
            try self.writer.writeLine("@builtin(local_invocation_index) localInvocationIndex: u32,");
            try self.writer.writeLine("@builtin(num_workgroups) numWorkgroups: vec3<u32>");
            self.writer.dedent();
            try self.writer.writeLine(") {");
            self.writer.indent();
        }

        fn writeMslSignature(self: *Self, ir: *const kernel.KernelIR) !void {
            // Uniform struct
            if (ir.uniforms.len > 0) {
                try self.writer.writeLine("struct Uniforms {");
                self.writer.indent();
                for (ir.uniforms) |uni| {
                    try self.writer.writeIndent();
                    try self.writeType(uni.ty);
                    try self.writer.writeFmt(" {s};\n", .{uni.name});
                }
                self.writer.dedent();
                try self.writer.writeLine("};");
                try self.writer.newline();
            }

            try self.writer.writeLine("kernel void");
            try self.writer.writeFmt("{s}(\n", .{ir.entry_point});
            self.writer.indent();

            // Buffer parameters
            for (ir.buffers) |buf| {
                try self.writer.writeIndent();
                if (buf.access == .read_only) {
                    try self.writer.write("const ");
                }
                try self.writer.write("device ");
                try self.writeType(buf.element_type);
                try self.writer.writeFmt("* {s} [[buffer({d})]],\n", .{ buf.name, buf.binding });
            }

            // Uniform buffer
            if (ir.uniforms.len > 0) {
                try self.writer.writeLine("constant Uniforms& uniforms [[buffer(0)]],");
            }

            // Built-in parameters
            try self.writer.writeLine("uint3 globalInvocationId [[thread_position_in_grid]],");
            try self.writer.writeLine("uint3 localInvocationId [[thread_position_in_threadgroup]],");
            try self.writer.writeLine("uint3 workgroupId [[threadgroup_position_in_grid]],");
            try self.writer.writeLine("uint localInvocationIndex [[thread_index_in_threadgroup]],");
            try self.writer.writeLine("uint3 numWorkgroups [[threadgroups_per_grid]]");

            self.writer.dedent();
            try self.writer.writeLine(") {");
            self.writer.indent();
        }

        fn writeCudaSignature(self: *Self, ir: *const kernel.KernelIR) !void {
            try self.writer.writeFmt("__global__ void {s}(\n", .{ir.entry_point});
            self.writer.indent();

            // Buffer parameters
            for (ir.buffers, 0..) |buf, i| {
                try self.writer.writeIndent();
                if (buf.access == .read_only) {
                    try self.writer.write("const ");
                }
                try self.writeType(buf.element_type);
                try self.writer.writeFmt("* __restrict__ {s}", .{buf.name});

                if (i < ir.buffers.len - 1 or ir.uniforms.len > 0) {
                    try self.writer.write(",\n");
                } else {
                    try self.writer.newline();
                }
            }

            // Uniform parameters
            for (ir.uniforms, 0..) |uni, i| {
                try self.writer.writeIndent();
                try self.writeType(uni.ty);
                try self.writer.writeFmt(" {s}", .{uni.name});

                if (i < ir.uniforms.len - 1) {
                    try self.writer.write(",\n");
                } else {
                    try self.writer.newline();
                }
            }

            self.writer.dedent();
            try self.writer.writeLine(") {");
            self.writer.indent();
        }

        // ============================================================
        // Built-in Variables
        // ============================================================

        fn writeBuiltinVars(self: *Self, ir: *const kernel.KernelIR) !void {
            // GLSL/WGSL/MSL have built-ins via parameters, CUDA needs computation
            if (self.config.language == .cuda) {
                try self.writer.writeLine("// Built-in variables");
                try self.writer.writeLine("const uint3 globalInvocationId = make_uint3(");
                self.writer.indent();
                try self.writer.writeLine("blockIdx.x * blockDim.x + threadIdx.x,");
                try self.writer.writeLine("blockIdx.y * blockDim.y + threadIdx.y,");
                try self.writer.writeLine("blockIdx.z * blockDim.z + threadIdx.z);");
                self.writer.dedent();
                try self.writer.writeLine("const uint3 localInvocationId = make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);");
                try self.writer.writeLine("const uint3 workgroupId = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z);");
                try self.writer.writeFmt("const uint3 workgroupSize = make_uint3({d}, {d}, {d});\n", .{
                    ir.workgroup_size[0],
                    ir.workgroup_size[1],
                    ir.workgroup_size[2],
                });
                try self.writer.writeLine("const unsigned int localInvocationIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;");
                try self.writer.newline();
            }
        }

        // ============================================================
        // Body & Close
        // ============================================================

        fn writeBody(self: *Self, ir: *const kernel.KernelIR) !void {
            for (ir.body) |s| {
                try self.writeStmt(s);
            }
        }

        fn writeKernelClose(self: *Self) !void {
            self.writer.dedent();
            try self.writer.writeLine("}");
        }

        // ============================================================
        // Type Writing
        // ============================================================

        pub fn writeType(self: *Self, ty: types.Type) backend.CodegenError!void {
            switch (ty) {
                .scalar => |s| try self.writer.write(type_name_table[@intFromEnum(s)]),
                .vector => |v| try self.writeVectorType(v),
                .matrix => |m| try self.writeMatrixType(m),
                .array => |a| try self.writeArrayType(a),
                .ptr => |p| {
                    try self.writeType(p.pointee.*);
                    try self.writer.write("*");
                },
                .void_ => try self.writer.write(self.config.type_names.void_),
            }
        }

        fn writeVectorType(self: *Self, v: types.VectorType) !void {
            switch (self.config.vector_naming.style) {
                .prefix => {
                    const prefix = vector_prefix_table[@intFromEnum(v.element)];
                    try self.writer.writeFmt("{s}{d}", .{ prefix, v.size });
                },
                .type_suffix => {
                    const prefix = vector_prefix_table[@intFromEnum(v.element)];
                    try self.writer.writeFmt("{s}{d}", .{ prefix, v.size });
                },
                .generic => {
                    // WGSL style: vec3<f32>
                    try self.writer.writeFmt("vec{d}<", .{v.size});
                    try self.writer.write(type_name_table[@intFromEnum(v.element)]);
                    try self.writer.write(">");
                },
            }
        }

        fn writeMatrixType(self: *Self, m: types.MatrixType) !void {
            switch (self.config.language) {
                .glsl => try self.writer.writeFmt("mat{d}x{d}", .{ m.cols, m.rows }),
                .wgsl => {
                    try self.writer.writeFmt("mat{d}x{d}<", .{ m.cols, m.rows });
                    try self.writer.write(type_name_table[@intFromEnum(m.element)]);
                    try self.writer.write(">");
                },
                .msl => try self.writer.writeFmt("float{d}x{d}", .{ m.cols, m.rows }),
                .cuda => {
                    // CUDA doesn't have native matrix types
                    const base = type_name_table[@intFromEnum(m.element)];
                    try self.writer.writeFmt("{s}[{d}][{d}]", .{ base, m.rows, m.cols });
                },
                else => try self.writer.writeFmt("mat{d}x{d}", .{ m.cols, m.rows }),
            }
        }

        fn writeArrayType(self: *Self, a: types.ArrayType) !void {
            switch (self.config.language) {
                .wgsl => {
                    try self.writer.write("array<");
                    try self.writeType(a.element.*);
                    if (a.size) |size| {
                        try self.writer.writeFmt(", {d}>", .{size});
                    } else {
                        try self.writer.write(">");
                    }
                },
                else => {
                    try self.writeType(a.element.*);
                    if (a.size) |size| {
                        try self.writer.writeFmt("[{d}]", .{size});
                    } else {
                        try self.writer.write("*");
                    }
                },
            }
        }

        // ============================================================
        // Expression Writing
        // ============================================================

        pub fn writeExpr(self: *Self, e: *const expr.Expr) backend.CodegenError!void {
            switch (e.*) {
                .literal => |lit| try self.writeLiteral(lit),
                .ref => |ref| try self.writeRef(ref),
                .unary => |un| try self.writeUnary(un),
                .binary => |bin| try self.writeBinary(bin),
                .call => |c| try self.writeCall(c),
                .index => |idx| try self.writeIndex(idx),
                .field => |f| try self.writeField(f),
                .cast => |c| try self.writeCast(c),
                .select => |s| try self.writeSelect(s),
                .vector_construct => |vc| try self.writeVectorConstruct(vc),
                .swizzle => |sw| try self.writeSwizzle(sw),
            }
        }

        fn writeRef(self: *Self, ref: expr.ValueRef) !void {
            if (ref.name) |name| {
                try self.writer.write(name);
            } else {
                try self.writer.writeFmt("_v{d}", .{ref.id});
            }
        }

        fn writeLiteral(self: *Self, lit: expr.Literal) !void {
            const fmt = self.config.literal_format;
            switch (lit) {
                .bool_ => |v| try self.writer.write(if (v) fmt.bool_true else fmt.bool_false),
                .i32_ => |v| try self.writer.writeFmt("{d}{s}", .{ v, fmt.i32_suffix }),
                .i64_ => |v| try self.writer.writeFmt("{d}{s}", .{ v, fmt.i64_suffix }),
                .u32_ => |v| try self.writer.writeFmt("{d}{s}", .{ v, fmt.u32_suffix }),
                .u64_ => |v| try self.writer.writeFmt("{d}{s}", .{ v, fmt.u64_suffix }),
                .f32_ => |v| {
                    if (v == @trunc(v)) {
                        try self.writer.writeFmt("{d}{s}", .{ @as(i64, @intFromFloat(v)), fmt.f32_decimal_suffix });
                    } else {
                        try self.writer.writeFmt("{d}{s}", .{ v, fmt.f32_suffix });
                    }
                },
                .f64_ => |v| {
                    if (v == @trunc(v)) {
                        try self.writer.writeFmt("{d}{s}", .{ @as(i64, @intFromFloat(v)), fmt.f64_decimal_suffix });
                    } else {
                        try self.writer.writeFmt("{d}{s}", .{ v, fmt.f64_suffix });
                    }
                },
            }
        }

        fn writeUnary(self: *Self, un: expr.Expr.UnaryExpr) !void {
            if (un.op.isPrefix()) {
                try self.writer.write(common.OperatorSymbols.unaryOp(un.op));
                try self.writer.write("(");
                try self.writeExpr(un.operand);
                try self.writer.write(")");
            } else {
                // Function-style unary op - use comptime lookup table for O(1) access
                const func_name = unary_fn_table[@intFromEnum(un.op)];
                try self.writer.writeFmt("{s}(", .{func_name});
                try self.writeExpr(un.operand);
                try self.writer.write(")");
            }
        }

        fn writeBinary(self: *Self, bin: expr.Expr.BinaryExpr) !void {
            if (bin.op.isInfix()) {
                try self.writer.write("(");
                try self.writeExpr(bin.left);
                try self.writer.write(common.OperatorSymbols.binaryOp(bin.op));
                try self.writeExpr(bin.right);
                try self.writer.write(")");
            } else {
                // Function-style binary op - use comptime lookup table for O(1) access
                const func_name = binary_fn_table[@intFromEnum(bin.op)];
                try self.writer.writeFmt("{s}(", .{func_name});
                try self.writeExpr(bin.left);
                try self.writer.write(", ");
                try self.writeExpr(bin.right);
                try self.writer.write(")");
            }
        }

        fn writeCall(self: *Self, c: expr.Expr.CallExpr) backend.CodegenError!void {
            switch (c.function) {
                .barrier => try self.writer.write(self.config.barriers.barrier),
                .memory_barrier => try self.writer.write(self.config.barriers.memory_barrier),
                .memory_barrier_buffer => try self.writer.write(self.config.barriers.memory_barrier_buffer),
                .memory_barrier_shared => try self.writer.write(self.config.barriers.memory_barrier_shared),
                .atomic_add => try self.writeAtomicOp(self.config.atomics.add_fn, c.args),
                .atomic_sub => try self.writeAtomicOp(self.config.atomics.sub_fn, c.args),
                .atomic_and => try self.writeAtomicOp(self.config.atomics.and_fn, c.args),
                .atomic_or => try self.writeAtomicOp(self.config.atomics.or_fn, c.args),
                .atomic_xor => try self.writeAtomicOp(self.config.atomics.xor_fn, c.args),
                .atomic_min => try self.writeAtomicOp(self.config.atomics.min_fn, c.args),
                .atomic_max => try self.writeAtomicOp(self.config.atomics.max_fn, c.args),
                .atomic_exchange => try self.writeAtomicOp(self.config.atomics.exchange_fn, c.args),
                .atomic_compare_exchange => try self.writeAtomicCompareExchange(c.args),
                .atomic_load => try self.writeAtomicLoad(c.args),
                .atomic_store => try self.writeAtomicStore(c.args),
                .clamp => try self.writeBuiltinCall(self.config.builtin_fns.clamp, c.args),
                .mix => try self.writeBuiltinCall(self.config.builtin_fns.mix, c.args),
                .smoothstep => try self.writeBuiltinCall(self.config.builtin_fns.smoothstep, c.args),
                .fma => try self.writeBuiltinCall(self.config.builtin_fns.fma, c.args),
                .select => try self.writeSelectCall(c.args),
                .all => try self.writeBuiltinCall(self.config.builtin_fns.all, c.args),
                .any => try self.writeBuiltinCall(self.config.builtin_fns.any, c.args),
            }
        }

        fn writeAtomicOp(self: *Self, func_name: []const u8, args: []const *const expr.Expr) !void {
            try self.writer.writeFmt("{s}(", .{func_name});
            if (args.len >= 2) {
                if (self.config.atomics.needs_cast) {
                    try self.writer.write(self.config.atomics.cast_template);
                }
                try self.writer.write(self.config.atomics.ptr_prefix);
                try self.writeExpr(args[0]);
                try self.writer.write(", ");
                try self.writeExpr(args[1]);
            }
            try self.writer.write(self.config.atomics.suffix);
        }

        fn writeAtomicCompareExchange(self: *Self, args: []const *const expr.Expr) !void {
            try self.writer.writeFmt("{s}(", .{self.config.atomics.compare_exchange_fn});
            for (args, 0..) |arg, i| {
                if (i > 0) try self.writer.write(", ");
                if (i == 0 and self.config.atomics.needs_cast) {
                    try self.writer.write(self.config.atomics.cast_template);
                }
                try self.writer.write(self.config.atomics.ptr_prefix);
                try self.writeExpr(arg);
            }
            try self.writer.write(self.config.atomics.suffix);
        }

        fn writeAtomicLoad(self: *Self, args: []const *const expr.Expr) !void {
            if (self.config.atomics.load_fn) |func| {
                try self.writer.writeFmt("{s}(", .{func});
                if (args.len >= 1) {
                    if (self.config.atomics.needs_cast) {
                        try self.writer.write(self.config.atomics.cast_template);
                    }
                    try self.writer.write(self.config.atomics.ptr_prefix);
                    try self.writeExpr(args[0]);
                }
                try self.writer.write(self.config.atomics.suffix);
            } else if (args.len >= 1) {
                try self.writeExpr(args[0]);
            }
        }

        fn writeAtomicStore(self: *Self, args: []const *const expr.Expr) !void {
            if (self.config.atomics.store_fn) |func| {
                try self.writer.writeFmt("{s}(", .{func});
                if (args.len >= 2) {
                    if (self.config.atomics.needs_cast) {
                        try self.writer.write(self.config.atomics.cast_template);
                    }
                    try self.writer.write(self.config.atomics.ptr_prefix);
                    try self.writeExpr(args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(args[1]);
                }
                try self.writer.write(self.config.atomics.suffix);
            }
        }

        fn writeBuiltinCall(self: *Self, func_name: []const u8, args: []const *const expr.Expr) !void {
            try self.writer.writeFmt("{s}(", .{func_name});
            for (args, 0..) |arg, i| {
                if (i > 0) try self.writer.write(", ");
                try self.writeExpr(arg);
            }
            try self.writer.write(")");
        }

        fn writeSelectCall(self: *Self, args: []const *const expr.Expr) !void {
            if (self.config.select_reversed) {
                // WGSL/MSL: select(false_val, true_val, cond)
                try self.writer.write("select(");
                if (args.len >= 3) {
                    try self.writeExpr(args[2]); // false
                    try self.writer.write(", ");
                    try self.writeExpr(args[1]); // true
                    try self.writer.write(", ");
                    try self.writeExpr(args[0]); // condition
                }
                try self.writer.write(")");
            } else {
                // GLSL/CUDA: ternary
                try self.writer.write("(");
                if (args.len >= 3) {
                    try self.writeExpr(args[0]);
                    try self.writer.write(" ? ");
                    try self.writeExpr(args[1]);
                    try self.writer.write(" : ");
                    try self.writeExpr(args[2]);
                }
                try self.writer.write(")");
            }
        }

        fn writeIndex(self: *Self, idx: expr.Expr.IndexExpr) !void {
            try self.writeExpr(idx.base);
            try self.writer.write("[");
            try self.writeExpr(idx.index);
            try self.writer.write("]");
        }

        fn writeField(self: *Self, f: expr.Expr.FieldExpr) !void {
            try self.writeExpr(f.base);
            try self.writer.writeFmt(".{s}", .{f.field});
        }

        fn writeCast(self: *Self, c: expr.Expr.CastExpr) !void {
            switch (self.config.language) {
                .wgsl => {
                    try self.writeType(c.target_type);
                    try self.writer.write("(");
                    try self.writeExpr(c.operand);
                    try self.writer.write(")");
                },
                else => {
                    try self.writer.write("(");
                    try self.writeType(c.target_type);
                    try self.writer.write(")(");
                    try self.writeExpr(c.operand);
                    try self.writer.write(")");
                },
            }
        }

        fn writeSelect(self: *Self, s: expr.Expr.SelectExpr) !void {
            if (self.config.select_reversed) {
                try self.writer.write("select(");
                try self.writeExpr(s.false_value);
                try self.writer.write(", ");
                try self.writeExpr(s.true_value);
                try self.writer.write(", ");
                try self.writeExpr(s.condition);
                try self.writer.write(")");
            } else {
                try self.writer.write("(");
                try self.writeExpr(s.condition);
                try self.writer.write(" ? ");
                try self.writeExpr(s.true_value);
                try self.writer.write(" : ");
                try self.writeExpr(s.false_value);
                try self.writer.write(")");
            }
        }

        fn writeVectorConstruct(self: *Self, vc: expr.Expr.VectorConstruct) !void {
            switch (self.config.language) {
                .cuda => {
                    const base = type_name_table[@intFromEnum(vc.element_type)];
                    try self.writer.writeFmt("make_{s}{d}(", .{ base, vc.size });
                },
                .wgsl => {
                    try self.writer.writeFmt("vec{d}<", .{vc.size});
                    try self.writer.write(type_name_table[@intFromEnum(vc.element_type)]);
                    try self.writer.write(">(");
                },
                else => {
                    const prefix = vector_prefix_table[@intFromEnum(vc.element_type)];
                    try self.writer.writeFmt("{s}{d}(", .{ prefix, vc.size });
                },
            }
            for (vc.components, 0..) |comp, i| {
                if (i > 0) try self.writer.write(", ");
                try self.writeExpr(comp);
            }
            try self.writer.write(")");
        }

        fn writeSwizzle(self: *Self, sw: expr.Expr.SwizzleExpr) !void {
            try self.writeExpr(sw.base);
            try self.writer.write(".");
            for (sw.components) |comp| {
                const c: u8 = switch (comp) {
                    0 => 'x',
                    1 => 'y',
                    2 => 'z',
                    3 => 'w',
                    else => 'x',
                };
                try self.writer.write(&[_]u8{c});
            }
        }

        // ============================================================
        // Statement Writing
        // ============================================================

        pub fn writeStmt(self: *Self, s: *const stmt.Stmt) backend.CodegenError!void {
            switch (s.*) {
                .var_decl => |v| try self.writeVarDecl(v),
                .assign => |a| try self.writeAssign(a),
                .compound_assign => |ca| try self.writeCompoundAssign(ca),
                .if_ => |i| try self.writeIf(i),
                .for_ => |f| try self.writeFor(f),
                .while_ => |w| try self.writeWhile(w),
                .do_while => |dw| try self.writeDoWhile(dw),
                .return_ => |r| try self.writeReturn(r),
                .break_ => try self.writer.writeLine("break;"),
                .continue_ => try self.writeContinue(),
                .discard => try self.writeDiscard(),
                .expr_stmt => |e| try self.writeExprStmt(e),
                .block => |b| try self.writeBlock(b),
                .switch_ => |sw| try self.writeSwitch(sw),
            }
        }

        fn writeVarDecl(self: *Self, v: stmt.Stmt.VarDecl) !void {
            try self.writer.writeIndent();
            switch (self.config.language) {
                .wgsl => {
                    if (v.is_const) {
                        try self.writer.write("let ");
                    } else {
                        try self.writer.write("var ");
                    }
                    try self.writer.writeFmt("{s}: ", .{v.name});
                    try self.writeType(v.ty);
                },
                else => {
                    if (v.is_const) try self.writer.write(self.config.const_keyword);
                    try self.writeType(v.ty);
                    try self.writer.writeFmt(" {s}", .{v.name});
                },
            }
            if (v.init) |init_expr| {
                try self.writer.write(" = ");
                try self.writeExpr(init_expr);
            }
            try self.writer.write(";\n");
        }

        fn writeAssign(self: *Self, a: stmt.Stmt.Assignment) !void {
            try self.writer.writeIndent();
            try self.writeExpr(a.target);
            try self.writer.write(" = ");
            try self.writeExpr(a.value);
            try self.writer.write(";\n");
        }

        fn writeCompoundAssign(self: *Self, ca: stmt.Stmt.CompoundAssign) !void {
            try self.writer.writeIndent();
            try self.writeExpr(ca.target);
            const op_str: []const u8 = switch (ca.op) {
                .add => " += ",
                .sub => " -= ",
                .mul => " *= ",
                .div => " /= ",
                .mod => " %= ",
                .bit_and => " &= ",
                .bit_or => " |= ",
                .bit_xor => " ^= ",
                .shl => " <<= ",
                .shr => " >>= ",
                else => " = ",
            };
            try self.writer.write(op_str);
            try self.writeExpr(ca.value);
            try self.writer.write(";\n");
        }

        fn writeIf(self: *Self, i: stmt.Stmt.IfStmt) !void {
            try self.writer.writeIndent();
            try self.writer.write("if (");
            try self.writeExpr(i.condition);
            try self.writer.write(") {\n");
            self.writer.indent();
            for (i.then_body) |body_stmt| {
                try self.writeStmt(body_stmt);
            }
            self.writer.dedent();
            try self.writer.writeIndent();
            try self.writer.write("}");
            if (i.else_body) |else_body| {
                try self.writer.write(" else {\n");
                self.writer.indent();
                for (else_body) |body_stmt| {
                    try self.writeStmt(body_stmt);
                }
                self.writer.dedent();
                try self.writer.writeIndent();
                try self.writer.write("}");
            }
            try self.writer.newline();
        }

        fn writeFor(self: *Self, f: stmt.Stmt.ForStmt) !void {
            try self.writer.writeIndent();
            try self.writer.write("for (");
            if (f.init) |init_stmt| {
                try self.writeStmtInline(init_stmt);
            }
            try self.writer.write("; ");
            if (f.condition) |cond| {
                try self.writeExpr(cond);
            }
            try self.writer.write("; ");
            if (f.update) |update| {
                try self.writeStmtInline(update);
            }
            try self.writer.write(") {\n");
            self.writer.indent();
            for (f.body) |body_stmt| {
                try self.writeStmt(body_stmt);
            }
            self.writer.dedent();
            try self.writer.writeLine("}");
        }

        fn writeWhile(self: *Self, w: stmt.Stmt.WhileStmt) !void {
            if (self.config.while_style == .loop_break) {
                // WGSL: loop { if (!cond) { break; } ... }
                try self.writer.writeLine("loop {");
                self.writer.indent();
                try self.writer.writeIndent();
                try self.writer.write("if (!(");
                try self.writeExpr(w.condition);
                try self.writer.write(")) { break; }\n");
                for (w.body) |body_stmt| {
                    try self.writeStmt(body_stmt);
                }
                self.writer.dedent();
                try self.writer.writeLine("}");
            } else {
                try self.writer.writeIndent();
                try self.writer.write("while (");
                try self.writeExpr(w.condition);
                try self.writer.write(") {\n");
                self.writer.indent();
                for (w.body) |body_stmt| {
                    try self.writeStmt(body_stmt);
                }
                self.writer.dedent();
                try self.writer.writeLine("}");
            }
        }

        fn writeDoWhile(self: *Self, dw: stmt.Stmt.DoWhileStmt) !void {
            if (self.config.while_style == .loop_break) {
                // WGSL: loop { ... if (!cond) { break; } }
                try self.writer.writeLine("loop {");
                self.writer.indent();
                for (dw.body) |body_stmt| {
                    try self.writeStmt(body_stmt);
                }
                try self.writer.writeIndent();
                try self.writer.write("if (!(");
                try self.writeExpr(dw.condition);
                try self.writer.write(")) { break; }\n");
                self.writer.dedent();
                try self.writer.writeLine("}");
            } else {
                try self.writer.writeLine("do {");
                self.writer.indent();
                for (dw.body) |body_stmt| {
                    try self.writeStmt(body_stmt);
                }
                self.writer.dedent();
                try self.writer.writeIndent();
                try self.writer.write("} while (");
                try self.writeExpr(dw.condition);
                try self.writer.write(");\n");
            }
        }

        fn writeReturn(self: *Self, r: stmt.Stmt.Return) !void {
            try self.writer.writeIndent();
            try self.writer.write("return");
            if (r.value) |val| {
                try self.writer.write(" ");
                try self.writeExpr(val);
            }
            try self.writer.write(";\n");
        }

        fn writeContinue(self: *Self) !void {
            if (self.config.while_style == .loop_break) {
                try self.writer.writeLine("continuing;");
            } else {
                try self.writer.writeLine("continue;");
            }
        }

        fn writeDiscard(self: *Self) !void {
            if (self.config.discard_stmt.len > 0) {
                try self.writer.writeLine(self.config.discard_stmt);
            }
        }

        fn writeExprStmt(self: *Self, e: *const expr.Expr) !void {
            try self.writer.writeIndent();
            try self.writeExpr(e);
            try self.writer.write(";\n");
        }

        fn writeBlock(self: *Self, b: stmt.Stmt.Block) !void {
            try self.writer.writeLine("{");
            self.writer.indent();
            for (b.statements) |body_stmt| {
                try self.writeStmt(body_stmt);
            }
            self.writer.dedent();
            try self.writer.writeLine("}");
        }

        fn writeSwitch(self: *Self, sw: stmt.Stmt.SwitchStmt) !void {
            try self.writer.writeIndent();
            try self.writer.write("switch (");
            try self.writeExpr(sw.selector);
            try self.writer.write(") {\n");
            self.writer.indent();
            for (sw.cases) |case| {
                try self.writer.writeIndent();
                try self.writer.write("case ");
                try self.writeLiteral(case.value);
                try self.writer.write(":\n");
                self.writer.indent();
                for (case.body) |body_stmt| {
                    try self.writeStmt(body_stmt);
                }
                if (!case.fallthrough) {
                    try self.writer.writeLine("break;");
                }
                self.writer.dedent();
            }
            if (sw.default) |default| {
                try self.writer.writeLine("default:");
                self.writer.indent();
                for (default) |body_stmt| {
                    try self.writeStmt(body_stmt);
                }
                try self.writer.writeLine("break;");
                self.writer.dedent();
            }
            self.writer.dedent();
            try self.writer.writeLine("}");
        }

        fn writeStmtInline(self: *Self, s: *const stmt.Stmt) !void {
            switch (s.*) {
                .var_decl => |v| {
                    switch (self.config.language) {
                        .wgsl => {
                            if (v.is_const) {
                                try self.writer.write("let ");
                            } else {
                                try self.writer.write("var ");
                            }
                            try self.writer.writeFmt("{s}: ", .{v.name});
                            try self.writeType(v.ty);
                        },
                        else => {
                            try self.writeType(v.ty);
                            try self.writer.writeFmt(" {s}", .{v.name});
                        },
                    }
                    if (v.init) |init_expr| {
                        try self.writer.write(" = ");
                        try self.writeExpr(init_expr);
                    }
                },
                .assign => |a| {
                    try self.writeExpr(a.target);
                    try self.writer.write(" = ");
                    try self.writeExpr(a.value);
                },
                .compound_assign => |ca| {
                    try self.writeExpr(ca.target);
                    const op_str: []const u8 = switch (ca.op) {
                        .add => "+=",
                        .sub => "-=",
                        else => "=",
                    };
                    try self.writer.write(op_str);
                    try self.writeExpr(ca.value);
                },
                .expr_stmt => |e| try self.writeExpr(e),
                else => {},
            }
        }
    };
}

// ============================================================================
// Pre-instantiated Generators
// ============================================================================

/// GLSL code generator using generic template.
pub const GlslGenerator = CodeGenerator(@import("configs/glsl_config.zig"));

/// WGSL code generator using generic template.
pub const WgslGenerator = CodeGenerator(@import("configs/wgsl_config.zig"));

/// MSL code generator using generic template.
pub const MslGenerator = CodeGenerator(@import("configs/msl_config.zig"));

/// CUDA code generator using generic template.
pub const CudaGenerator = CodeGenerator(@import("configs/cuda_config.zig"));

// ============================================================================
// Tests
// ============================================================================

test "GlslGenerator basic" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = GlslGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "#version 450") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.code, "void main()") != null);
}

test "WgslGenerator basic" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = WgslGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "@compute") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.code, "@workgroup_size") != null);
}

test "MslGenerator basic" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = MslGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "#include <metal_stdlib>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.code, "kernel void") != null);
}

test "CudaGenerator basic" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = CudaGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "#include <cuda_runtime.h>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.code, "__global__") != null);
}
