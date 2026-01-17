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
