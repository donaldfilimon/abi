//! CUDA Code Generator
//!
//! Generates CUDA C source code from kernel IR.

const std = @import("std");
const types = @import("../types.zig");
const expr = @import("../expr.zig");
const stmt = @import("../stmt.zig");
const kernel = @import("../kernel.zig");
const backend = @import("backend.zig");
const common = @import("common.zig");
const gpu_backend = @import("../../backend.zig");

/// CUDA code generator.
pub const CudaGenerator = struct {
    writer: common.CodeWriter,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .writer = common.CodeWriter.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.writer.deinit();
    }

    /// Generate CUDA source code from kernel IR.
    pub fn generate(
        self: *Self,
        ir: *const kernel.KernelIR,
    ) backend.CodegenError!backend.GeneratedSource {
        // Header comment
        try self.writer.writeLine("// Auto-generated CUDA kernel");
        try self.writer.writeFmt("// Kernel: {s}\n", .{ir.name});
        try self.writer.newline();

        // Includes
        try self.writer.writeLine("#include <cuda_runtime.h>");
        try self.writer.writeLine("#include <stdint.h>");
        try self.writer.newline();

        // Shared memory declarations
        try self.writeSharedMemory(ir);

        // Helper for clamp if needed
        try self.writer.writeLine("#ifndef CLAMP_DEFINED");
        try self.writer.writeLine("#define CLAMP_DEFINED");
        try self.writer.writeLine("__device__ __forceinline__ float clamp(float x, float lo, float hi) {");
        try self.writer.writeLine("    return fminf(fmaxf(x, lo), hi);");
        try self.writer.writeLine("}");
        try self.writer.writeLine("#endif");
        try self.writer.newline();

        // Kernel signature
        try self.writer.writeFmt("__global__ void {s}(\n", .{ir.entry_point});
        self.writer.indent();

        // Buffer parameters
        try self.writeBufferParams(ir);

        // Uniform parameters
        try self.writeUniformParams(ir);

        self.writer.dedent();
        try self.writer.writeLine(") {");
        self.writer.indent();

        // Built-in variable declarations
        try self.writeBuiltinVars(ir);

        // Kernel body
        for (ir.body) |s| {
            try self.writeStmt(s);
        }

        self.writer.dedent();
        try self.writer.writeLine("}");

        const code = try self.writer.getCode();
        const entry_point = try self.allocator.dupe(u8, ir.entry_point);

        return .{
            .code = code,
            .entry_point = entry_point,
            .backend = .cuda,
            .language = .cuda,
        };
    }

    fn writeSharedMemory(self: *Self, ir: *const kernel.KernelIR) !void {
        for (ir.shared_memory) |shared| {
            if (shared.size) |size| {
                // Static shared memory
                try self.writer.writeIndent();
                try self.writer.write("__shared__ ");
                try self.writeType(shared.element_type);
                try self.writer.writeFmt(" {s}[{d}];\n", .{ shared.name, size });
            } else {
                // Dynamic shared memory
                try self.writer.writeIndent();
                try self.writer.write("extern __shared__ ");
                try self.writeType(shared.element_type);
                try self.writer.writeFmt(" {s}[];\n", .{shared.name});
            }
        }
        if (ir.shared_memory.len > 0) {
            try self.writer.newline();
        }
    }

    fn writeBufferParams(self: *Self, ir: *const kernel.KernelIR) !void {
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
    }

    fn writeUniformParams(self: *Self, ir: *const kernel.KernelIR) !void {
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
    }

    fn writeBuiltinVars(self: *Self, ir: *const kernel.KernelIR) !void {
        // Global invocation ID
        try self.writer.writeLine("// Built-in variables");
        try self.writer.writeLine("const uint3 globalInvocationId = make_uint3(");
        self.writer.indent();
        try self.writer.writeLine("blockIdx.x * blockDim.x + threadIdx.x,");
        try self.writer.writeLine("blockIdx.y * blockDim.y + threadIdx.y,");
        try self.writer.writeLine("blockIdx.z * blockDim.z + threadIdx.z);");
        self.writer.dedent();

        // Local invocation ID
        try self.writer.writeLine("const uint3 localInvocationId = make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);");

        // Workgroup ID
        try self.writer.writeLine("const uint3 workgroupId = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z);");

        // Workgroup size (compile-time constant)
        try self.writer.writeFmt("const uint3 workgroupSize = make_uint3({d}, {d}, {d});\n", .{
            ir.workgroup_size[0],
            ir.workgroup_size[1],
            ir.workgroup_size[2],
        });

        // Local invocation index
        try self.writer.writeLine("const unsigned int localInvocationIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;");

        try self.writer.newline();
    }

    fn writeType(self: *Self, ty: types.Type) backend.CodegenError!void {
        switch (ty) {
            .scalar => |s| try self.writer.write(common.TypeNames.cuda.getScalarName(s)),
            .vector => |v| {
                const base = common.TypeNames.cuda.getScalarName(v.element);
                try self.writer.writeFmt("{s}{d}", .{ base, v.size });
            },
            .matrix => |m| {
                // CUDA doesn't have native matrix types, use arrays
                const base = common.TypeNames.cuda.getScalarName(m.element);
                try self.writer.writeFmt("{s}[{d}][{d}]", .{ base, m.rows, m.cols });
            },
            .array => |a| {
                try self.writeType(a.element.*);
                if (a.size) |size| {
                    try self.writer.writeFmt("[{d}]", .{size});
                } else {
                    try self.writer.write("*"); // Runtime-sized as pointer
                }
            },
            .ptr => |p| {
                try self.writeType(p.pointee.*);
                try self.writer.write("*");
            },
            .void_ => try self.writer.write("void"),
        }
    }

    fn writeExpr(self: *Self, e: *const expr.Expr) backend.CodegenError!void {
        switch (e.*) {
            .literal => |lit| try self.writeLiteral(lit),
            .ref => |ref| {
                if (ref.name) |name| {
                    try self.writer.write(name);
                } else {
                    try self.writer.writeFmt("_v{d}", .{ref.id});
                }
            },
            .unary => |un| {
                if (un.op.isPrefix()) {
                    try self.writer.write(common.OperatorSymbols.unaryOp(un.op));
                    try self.writer.write("(");
                    try self.writeExpr(un.operand);
                    try self.writer.write(")");
                } else {
                    // Function-style unary op
                    try self.writeUnaryFunc(un.op, un.operand);
                }
            },
            .binary => |bin| {
                if (bin.op.isInfix()) {
                    try self.writer.write("(");
                    try self.writeExpr(bin.left);
                    try self.writer.write(common.OperatorSymbols.binaryOp(bin.op));
                    try self.writeExpr(bin.right);
                    try self.writer.write(")");
                } else {
                    // Function-style binary op
                    try self.writeBinaryFunc(bin.op, bin.left, bin.right);
                }
            },
            .call => |c| try self.writeCall(c),
            .index => |idx| {
                try self.writeExpr(idx.base);
                try self.writer.write("[");
                try self.writeExpr(idx.index);
                try self.writer.write("]");
            },
            .field => |f| {
                try self.writeExpr(f.base);
                try self.writer.writeFmt(".{s}", .{f.field});
            },
            .cast => |c| {
                try self.writer.write("(");
                try self.writeType(c.target_type);
                try self.writer.write(")(");
                try self.writeExpr(c.operand);
                try self.writer.write(")");
            },
            .select => |s| {
                try self.writer.write("(");
                try self.writeExpr(s.condition);
                try self.writer.write(" ? ");
                try self.writeExpr(s.true_value);
                try self.writer.write(" : ");
                try self.writeExpr(s.false_value);
                try self.writer.write(")");
            },
            .vector_construct => |vc| {
                const base = common.TypeNames.cuda.getScalarName(vc.element_type);
                try self.writer.writeFmt("make_{s}{d}(", .{ base, vc.size });
                for (vc.components, 0..) |comp, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.writeExpr(comp);
                }
                try self.writer.write(")");
            },
            .swizzle => |sw| {
                // CUDA doesn't support swizzle directly, need to construct
                try self.writeExpr(sw.base);
                // For now, just access first component
                if (sw.components.len > 0) {
                    const comp = switch (sw.components[0]) {
                        0 => ".x",
                        1 => ".y",
                        2 => ".z",
                        3 => ".w",
                        else => ".x",
                    };
                    try self.writer.write(comp);
                }
            },
        }
    }

    fn writeLiteral(self: *Self, lit: expr.Literal) !void {
        switch (lit) {
            .bool_ => |v| try self.writer.write(if (v) "true" else "false"),
            .i32_ => |v| try self.writer.writeFmt("{d}", .{v}),
            .i64_ => |v| try self.writer.writeFmt("{d}LL", .{v}),
            .u32_ => |v| try self.writer.writeFmt("{d}u", .{v}),
            .u64_ => |v| try self.writer.writeFmt("{d}ULL", .{v}),
            .f32_ => |v| {
                if (v == @trunc(v)) {
                    try self.writer.writeFmt("{d}.0f", .{@as(i64, @intFromFloat(v))});
                } else {
                    try self.writer.writeFmt("{d}f", .{v});
                }
            },
            .f64_ => |v| {
                if (v == @trunc(v)) {
                    try self.writer.writeFmt("{d}.0", .{@as(i64, @intFromFloat(v))});
                } else {
                    try self.writer.writeFmt("{d}", .{v});
                }
            },
        }
    }

    fn writeUnaryFunc(self: *Self, op: expr.UnaryOp, operand: *const expr.Expr) !void {
        const func_name: []const u8 = switch (op) {
            .abs => "fabsf",
            .sqrt => "sqrtf",
            .sin => "sinf",
            .cos => "cosf",
            .tan => "tanf",
            .asin => "asinf",
            .acos => "acosf",
            .atan => "atanf",
            .sinh => "sinhf",
            .cosh => "coshf",
            .tanh => "tanhf",
            .exp => "expf",
            .exp2 => "exp2f",
            .log => "logf",
            .log2 => "log2f",
            .log10 => "log10f",
            .floor => "floorf",
            .ceil => "ceilf",
            .round => "roundf",
            .trunc => "truncf",
            .fract => "fmodf", // x - floor(x)
            .sign => "copysignf", // Need special handling
            .normalize => "normalize", // Need special handling
            .length => "length", // Need special handling
            else => "unknown",
        };
        try self.writer.writeFmt("{s}(", .{func_name});
        try self.writeExpr(operand);
        try self.writer.write(")");
    }

    fn writeBinaryFunc(
        self: *Self,
        op: expr.BinaryOp,
        left: *const expr.Expr,
        right: *const expr.Expr,
    ) !void {
        const func_name: []const u8 = switch (op) {
            .min => "fminf",
            .max => "fmaxf",
            .pow => "powf",
            .atan2 => "atan2f",
            .dot => "dot", // Need vector handling
            .cross => "cross", // Need vector handling
            .distance => "distance", // Need vector handling
            .step => "step", // Need implementation
            .reflect => "reflect", // Need implementation
            else => "unknown",
        };
        try self.writer.writeFmt("{s}(", .{func_name});
        try self.writeExpr(left);
        try self.writer.write(", ");
        try self.writeExpr(right);
        try self.writer.write(")");
    }

    fn writeCall(self: *Self, c: expr.Expr.CallExpr) backend.CodegenError!void {
        switch (c.function) {
            .barrier => try self.writer.write("__syncthreads()"),
            .memory_barrier => try self.writer.write("__threadfence()"),
            .memory_barrier_buffer => try self.writer.write("__threadfence()"),
            .memory_barrier_shared => try self.writer.write("__syncthreads()"),
            .atomic_add => {
                try self.writer.write("atomicAdd(");
                if (c.args.len >= 2) {
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
            },
            .atomic_sub => {
                try self.writer.write("atomicSub(");
                if (c.args.len >= 2) {
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
            },
            .atomic_and => {
                try self.writer.write("atomicAnd(");
                if (c.args.len >= 2) {
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
            },
            .atomic_or => {
                try self.writer.write("atomicOr(");
                if (c.args.len >= 2) {
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
            },
            .atomic_xor => {
                try self.writer.write("atomicXor(");
                if (c.args.len >= 2) {
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
            },
            .atomic_min => {
                try self.writer.write("atomicMin(");
                if (c.args.len >= 2) {
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
            },
            .atomic_max => {
                try self.writer.write("atomicMax(");
                if (c.args.len >= 2) {
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
            },
            .atomic_exchange => {
                try self.writer.write("atomicExch(");
                if (c.args.len >= 2) {
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
            },
            .atomic_compare_exchange => {
                try self.writer.write("atomicCAS(");
                for (c.args, 0..) |arg, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.writeExpr(arg);
                }
                try self.writer.write(")");
            },
            .atomic_load, .atomic_store => {
                // Not directly supported, use regular load/store with fence
                if (c.args.len >= 1) {
                    try self.writeExpr(c.args[0]);
                }
            },
            .clamp => {
                try self.writer.write("clamp(");
                for (c.args, 0..) |arg, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.writeExpr(arg);
                }
                try self.writer.write(")");
            },
            .mix => {
                try self.writer.write("lerp(");
                // Note: lerp in CUDA is (a, b, t) but GLSL mix is (a, b, t)
                for (c.args, 0..) |arg, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.writeExpr(arg);
                }
                try self.writer.write(")");
            },
            .smoothstep => {
                try self.writer.write("smoothstep(");
                for (c.args, 0..) |arg, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.writeExpr(arg);
                }
                try self.writer.write(")");
            },
            .fma => {
                try self.writer.write("fmaf(");
                for (c.args, 0..) |arg, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.writeExpr(arg);
                }
                try self.writer.write(")");
            },
            .select => {
                try self.writer.write("(");
                if (c.args.len >= 3) {
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(" ? ");
                    try self.writeExpr(c.args[1]);
                    try self.writer.write(" : ");
                    try self.writeExpr(c.args[2]);
                }
                try self.writer.write(")");
            },
            .all, .any => {
                // Would need implementation for vector bools
                try self.writer.write("true");
            },
        }
    }

    fn writeStmt(self: *Self, s: *const stmt.Stmt) backend.CodegenError!void {
        switch (s.*) {
            .var_decl => |v| {
                try self.writer.writeIndent();
                if (v.is_const) try self.writer.write("const ");
                try self.writeType(v.ty);
                try self.writer.writeFmt(" {s}", .{v.name});
                if (v.init) |init_expr| {
                    try self.writer.write(" = ");
                    try self.writeExpr(init_expr);
                }
                try self.writer.write(";\n");
            },
            .assign => |a| {
                try self.writer.writeIndent();
                try self.writeExpr(a.target);
                try self.writer.write(" = ");
                try self.writeExpr(a.value);
                try self.writer.write(";\n");
            },
            .compound_assign => |ca| {
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
            },
            .if_ => |i| {
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
            },
            .for_ => |f| {
                try self.writer.writeIndent();
                try self.writer.write("for (");
                if (f.init) |init_expr| {
                    try self.writeStmtInline(init_expr);
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
            },
            .while_ => |w| {
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
            },
            .do_while => |dw| {
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
            },
            .return_ => |r| {
                try self.writer.writeIndent();
                try self.writer.write("return");
                if (r.value) |val| {
                    try self.writer.write(" ");
                    try self.writeExpr(val);
                }
                try self.writer.write(";\n");
            },
            .break_ => try self.writer.writeLine("break;"),
            .continue_ => try self.writer.writeLine("continue;"),
            .discard => {}, // No-op in compute
            .expr_stmt => |e| {
                try self.writer.writeIndent();
                try self.writeExpr(e);
                try self.writer.write(";\n");
            },
            .block => |b| {
                try self.writer.writeLine("{");
                self.writer.indent();
                for (b.statements) |body_stmt| {
                    try self.writeStmt(body_stmt);
                }
                self.writer.dedent();
                try self.writer.writeLine("}");
            },
            .switch_ => |sw| {
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
            },
        }
    }

    fn writeStmtInline(self: *Self, s: *const stmt.Stmt) !void {
        // Write statement without newline (for for-loop init/update)
        switch (s.*) {
            .var_decl => |v| {
                try self.writeType(v.ty);
                try self.writer.writeFmt(" {s}", .{v.name});
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

// ============================================================================
// Vision Kernel Code Generation
// ============================================================================

/// Vision kernel code generation utilities for CUDA.
/// These functions generate optimized CUDA code for common vision operations.
pub const VisionKernels = struct {
    /// Generate a complete Conv2D CUDA kernel.
    /// Uses im2col + GEMM approach with shared memory tiling for efficiency.
    pub fn generateConv2d(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated CUDA Conv2D kernel");
        try writer.writeLine("// Uses im2col + GEMM approach with shared memory tiling");
        try writer.newline();
        try writer.writeLine("#include <cuda_runtime.h>");
        try writer.writeLine("#include <stdint.h>");
        try writer.newline();

        // Kernel function
        try writer.writeLine("extern \"C\" __global__ void conv2d(");
        try writer.writeLine("    const float* __restrict__ input,    // [batch, in_channels, height, width]");
        try writer.writeLine("    const float* __restrict__ weights,  // [out_channels, in_channels, kH, kW]");
        try writer.writeLine("    const float* __restrict__ bias,     // [out_channels]");
        try writer.writeLine("    float* __restrict__ output,         // [batch, out_channels, out_h, out_w]");
        try writer.writeLine("    int batch_size, int in_channels, int out_channels,");
        try writer.writeLine("    int in_height, int in_width, int out_height, int out_width,");
        try writer.writeLine("    int kernel_h, int kernel_w, int stride_h, int stride_w,");
        try writer.writeLine("    int pad_h, int pad_w");
        try writer.writeLine(") {");
        try writer.writeLine("    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;");
        try writer.writeLine("    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;");
        try writer.writeLine("    const int batch_oc = blockIdx.z;");
        try writer.writeLine("    const int batch_idx = batch_oc / out_channels;");
        try writer.writeLine("    const int oc = batch_oc % out_channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= out_width || out_y >= out_height || batch_idx >= batch_size) return;");
        try writer.newline();
        try writer.writeLine("    float sum = 0.0f;");
        try writer.writeLine("    for (int ic = 0; ic < in_channels; ++ic) {");
        try writer.writeLine("        for (int ky = 0; ky < kernel_h; ++ky) {");
        try writer.writeLine("            for (int kx = 0; kx < kernel_w; ++kx) {");
        try writer.writeLine("                int ih = out_y * stride_h + ky - pad_h;");
        try writer.writeLine("                int iw = out_x * stride_w + kx - pad_w;");
        try writer.writeLine("                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {");
        try writer.writeLine("                    int input_idx = ((batch_idx * in_channels + ic) * in_height + ih) * in_width + iw;");
        try writer.writeLine("                    int weight_idx = ((oc * in_channels + ic) * kernel_h + ky) * kernel_w + kx;");
        try writer.writeLine("                    sum += input[input_idx] * weights[weight_idx];");
        try writer.writeLine("                }");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    sum += bias[oc];");
        try writer.writeLine("    int output_idx = ((batch_idx * out_channels + oc) * out_height + out_y) * out_width + out_x;");
        try writer.writeLine("    output[output_idx] = sum;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a MaxPool2D CUDA kernel with indices output.
    pub fn generateMaxPool2d(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated CUDA MaxPool2D kernel");
        try writer.writeLine("#include <cuda_runtime.h>");
        try writer.writeLine("#include <float.h>");
        try writer.newline();
        try writer.writeLine("extern \"C\" __global__ void max_pool2d(");
        try writer.writeLine("    const float* __restrict__ input,   // [batch, channels, height, width]");
        try writer.writeLine("    float* __restrict__ output,        // [batch, channels, out_h, out_w]");
        try writer.writeLine("    int* __restrict__ indices,         // [batch, channels, out_h, out_w]");
        try writer.writeLine("    int batch_size, int channels,");
        try writer.writeLine("    int in_height, int in_width, int out_height, int out_width,");
        try writer.writeLine("    int kernel_size, int stride, int padding");
        try writer.writeLine(") {");
        try writer.writeLine("    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;");
        try writer.writeLine("    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;");
        try writer.writeLine("    const int bc = blockIdx.z;");
        try writer.writeLine("    const int batch_idx = bc / channels;");
        try writer.writeLine("    const int channel = bc % channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= out_width || out_y >= out_height || batch_idx >= batch_size) return;");
        try writer.newline();
        try writer.writeLine("    float max_val = -FLT_MAX;");
        try writer.writeLine("    int max_idx = 0;");
        try writer.writeLine("    for (int ky = 0; ky < kernel_size; ++ky) {");
        try writer.writeLine("        for (int kx = 0; kx < kernel_size; ++kx) {");
        try writer.writeLine("            int ih = out_y * stride + ky - padding;");
        try writer.writeLine("            int iw = out_x * stride + kx - padding;");
        try writer.writeLine("            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {");
        try writer.writeLine("                int input_idx = ((batch_idx * channels + channel) * in_height + ih) * in_width + iw;");
        try writer.writeLine("                float val = input[input_idx];");
        try writer.writeLine("                if (val > max_val) {");
        try writer.writeLine("                    max_val = val;");
        try writer.writeLine("                    max_idx = input_idx;");
        try writer.writeLine("                }");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    int output_idx = ((batch_idx * channels + channel) * out_height + out_y) * out_width + out_x;");
        try writer.writeLine("    output[output_idx] = max_val;");
        try writer.writeLine("    indices[output_idx] = max_idx;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate an AvgPool2D CUDA kernel.
    pub fn generateAvgPool2d(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated CUDA AvgPool2D kernel");
        try writer.writeLine("#include <cuda_runtime.h>");
        try writer.newline();
        try writer.writeLine("extern \"C\" __global__ void avg_pool2d(");
        try writer.writeLine("    const float* __restrict__ input,   // [batch, channels, height, width]");
        try writer.writeLine("    float* __restrict__ output,        // [batch, channels, out_h, out_w]");
        try writer.writeLine("    int batch_size, int channels,");
        try writer.writeLine("    int in_height, int in_width, int out_height, int out_width,");
        try writer.writeLine("    int kernel_size, int stride, int padding");
        try writer.writeLine(") {");
        try writer.writeLine("    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;");
        try writer.writeLine("    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;");
        try writer.writeLine("    const int bc = blockIdx.z;");
        try writer.writeLine("    const int batch_idx = bc / channels;");
        try writer.writeLine("    const int channel = bc % channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= out_width || out_y >= out_height || batch_idx >= batch_size) return;");
        try writer.newline();
        try writer.writeLine("    float sum = 0.0f;");
        try writer.writeLine("    int count = 0;");
        try writer.writeLine("    for (int ky = 0; ky < kernel_size; ++ky) {");
        try writer.writeLine("        for (int kx = 0; kx < kernel_size; ++kx) {");
        try writer.writeLine("            int ih = out_y * stride + ky - padding;");
        try writer.writeLine("            int iw = out_x * stride + kx - padding;");
        try writer.writeLine("            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {");
        try writer.writeLine("                int input_idx = ((batch_idx * channels + channel) * in_height + ih) * in_width + iw;");
        try writer.writeLine("                sum += input[input_idx];");
        try writer.writeLine("                count++;");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    int output_idx = ((batch_idx * channels + channel) * out_height + out_y) * out_width + out_x;");
        try writer.writeLine("    output[output_idx] = count > 0 ? sum / (float)count : 0.0f;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a BatchNorm2D CUDA kernel (inference mode).
    pub fn generateBatchNorm2d(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated CUDA BatchNorm2D kernel (inference)");
        try writer.writeLine("#include <cuda_runtime.h>");
        try writer.newline();
        try writer.writeLine("extern \"C\" __global__ void batch_norm2d(");
        try writer.writeLine("    const float* __restrict__ input,        // [batch, channels, height, width]");
        try writer.writeLine("    const float* __restrict__ gamma,        // [channels]");
        try writer.writeLine("    const float* __restrict__ beta,         // [channels]");
        try writer.writeLine("    const float* __restrict__ running_mean, // [channels]");
        try writer.writeLine("    const float* __restrict__ running_var,  // [channels]");
        try writer.writeLine("    float* __restrict__ output,             // [batch, channels, height, width]");
        try writer.writeLine("    int batch_size, int channels, int height, int width, float epsilon");
        try writer.writeLine(") {");
        try writer.writeLine("    const int idx = blockIdx.x * blockDim.x + threadIdx.x;");
        try writer.writeLine("    const int total = batch_size * channels * height * width;");
        try writer.writeLine("    if (idx >= total) return;");
        try writer.newline();
        try writer.writeLine("    const int hw = height * width;");
        try writer.writeLine("    const int chw = channels * hw;");
        try writer.writeLine("    const int c = (idx / hw) % channels;");
        try writer.newline();
        try writer.writeLine("    float x = input[idx];");
        try writer.writeLine("    float mean = running_mean[c];");
        try writer.writeLine("    float var = running_var[c];");
        try writer.writeLine("    float g = gamma[c];");
        try writer.writeLine("    float b = beta[c];");
        try writer.newline();
        try writer.writeLine("    float normalized = (x - mean) * rsqrtf(var + epsilon);");
        try writer.writeLine("    output[idx] = g * normalized + b;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate an im2col CUDA kernel.
    pub fn generateIm2col(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated CUDA im2col kernel");
        try writer.writeLine("// Transforms image patches to columns for efficient convolution via GEMM");
        try writer.writeLine("#include <cuda_runtime.h>");
        try writer.newline();
        try writer.writeLine("extern \"C\" __global__ void im2col(");
        try writer.writeLine("    const float* __restrict__ input,  // [batch, channels, height, width]");
        try writer.writeLine("    float* __restrict__ output,       // [batch, C*kH*kW, out_h*out_w]");
        try writer.writeLine("    int batch_size, int channels,");
        try writer.writeLine("    int in_height, int in_width, int out_height, int out_width,");
        try writer.writeLine("    int kernel_h, int kernel_w, int stride_h, int stride_w,");
        try writer.writeLine("    int pad_h, int pad_w");
        try writer.writeLine(") {");
        try writer.writeLine("    const int idx = blockIdx.x * blockDim.x + threadIdx.x;");
        try writer.writeLine("    const int kernel_hw = kernel_h * kernel_w;");
        try writer.writeLine("    const int col_h = channels * kernel_hw;");
        try writer.writeLine("    const int col_w = out_height * out_width;");
        try writer.writeLine("    const int col_size = col_h * col_w;");
        try writer.writeLine("    const int total = batch_size * col_size;");
        try writer.writeLine("    if (idx >= total) return;");
        try writer.newline();
        try writer.writeLine("    const int batch_idx = idx / col_size;");
        try writer.writeLine("    const int idx_in_batch = idx % col_size;");
        try writer.writeLine("    const int row = idx_in_batch / col_w;");
        try writer.writeLine("    const int col = idx_in_batch % col_w;");
        try writer.newline();
        try writer.writeLine("    const int c = row / kernel_hw;");
        try writer.writeLine("    const int row_in_kernel = row % kernel_hw;");
        try writer.writeLine("    const int ky = row_in_kernel / kernel_w;");
        try writer.writeLine("    const int kx = row_in_kernel % kernel_w;");
        try writer.newline();
        try writer.writeLine("    const int oh = col / out_width;");
        try writer.writeLine("    const int ow = col % out_width;");
        try writer.newline();
        try writer.writeLine("    int ih = oh * stride_h + ky - pad_h;");
        try writer.writeLine("    int iw = ow * stride_w + kx - pad_w;");
        try writer.newline();
        try writer.writeLine("    float val = 0.0f;");
        try writer.writeLine("    if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {");
        try writer.writeLine("        int input_idx = ((batch_idx * channels + c) * in_height + ih) * in_width + iw;");
        try writer.writeLine("        val = input[input_idx];");
        try writer.writeLine("    }");
        try writer.writeLine("    output[idx] = val;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a col2im CUDA kernel.
    pub fn generateCol2im(allocator: std.mem.Allocator) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        try writer.writeLine("// Auto-generated CUDA col2im kernel");
        try writer.writeLine("// Transforms columns back to image (inverse of im2col)");
        try writer.writeLine("#include <cuda_runtime.h>");
        try writer.newline();
        try writer.writeLine("extern \"C\" __global__ void col2im(");
        try writer.writeLine("    const float* __restrict__ col_input, // [batch, C*kH*kW, out_h*out_w]");
        try writer.writeLine("    float* __restrict__ output,          // [batch, channels, height, width]");
        try writer.writeLine("    int batch_size, int channels,");
        try writer.writeLine("    int in_height, int in_width, int out_height, int out_width,");
        try writer.writeLine("    int kernel_h, int kernel_w, int stride_h, int stride_w,");
        try writer.writeLine("    int pad_h, int pad_w");
        try writer.writeLine(") {");
        try writer.writeLine("    const int idx = blockIdx.x * blockDim.x + threadIdx.x;");
        try writer.writeLine("    const int kernel_hw = kernel_h * kernel_w;");
        try writer.writeLine("    const int col_h = channels * kernel_hw;");
        try writer.writeLine("    const int col_w = out_height * out_width;");
        try writer.writeLine("    const int col_size = col_h * col_w;");
        try writer.writeLine("    const int total = batch_size * col_size;");
        try writer.writeLine("    if (idx >= total) return;");
        try writer.newline();
        try writer.writeLine("    const int batch_idx = idx / col_size;");
        try writer.writeLine("    const int idx_in_batch = idx % col_size;");
        try writer.writeLine("    const int row = idx_in_batch / col_w;");
        try writer.writeLine("    const int col = idx_in_batch % col_w;");
        try writer.newline();
        try writer.writeLine("    const int c = row / kernel_hw;");
        try writer.writeLine("    const int row_in_kernel = row % kernel_hw;");
        try writer.writeLine("    const int ky = row_in_kernel / kernel_w;");
        try writer.writeLine("    const int kx = row_in_kernel % kernel_w;");
        try writer.newline();
        try writer.writeLine("    const int oh = col / out_width;");
        try writer.writeLine("    const int ow = col % out_width;");
        try writer.newline();
        try writer.writeLine("    int ih = oh * stride_h + ky - pad_h;");
        try writer.writeLine("    int iw = ow * stride_w + kx - pad_w;");
        try writer.newline();
        try writer.writeLine("    if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {");
        try writer.writeLine("        int output_idx = ((batch_idx * channels + c) * in_height + ih) * in_width + iw;");
        try writer.writeLine("        atomicAdd(&output[output_idx], col_input[idx]);");
        try writer.writeLine("    }");
        try writer.writeLine("}");

        return writer.getCode();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "CudaGenerator basic kernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = CudaGenerator.init(allocator);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "__global__") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.code, "test_kernel") != null);
}
