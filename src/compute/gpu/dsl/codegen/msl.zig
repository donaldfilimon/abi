//! Metal Shading Language (MSL) Code Generator
//!
//! Generates Metal compute shader source code from kernel IR.

const std = @import("std");
const types = @import("../types.zig");
const expr = @import("../expr.zig");
const stmt = @import("../stmt.zig");
const kernel = @import("../kernel.zig");
const backend = @import("backend.zig");
const common = @import("common.zig");
const gpu_backend = @import("../../backend.zig");

/// MSL code generator.
pub const MslGenerator = struct {
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

    /// Generate MSL source code from kernel IR.
    pub fn generate(
        self: *Self,
        ir: *const kernel.KernelIR,
    ) backend.CodegenError!backend.GeneratedSource {
        // Header
        try self.writeHeader(ir);

        // Uniform struct if needed
        try self.writeUniformStruct(ir);

        // Main kernel function
        try self.writeKernelFunction(ir);

        const code = try self.writer.getCode();
        const entry_point = try self.allocator.dupe(u8, ir.entry_point);

        return .{
            .code = code,
            .entry_point = entry_point,
            .backend = .metal,
            .language = .msl,
        };
    }

    fn writeHeader(self: *Self, ir: *const kernel.KernelIR) !void {
        try self.writer.writeLine("// Auto-generated Metal compute shader");
        try self.writer.writeFmt("// Kernel: {s}\n", .{ir.name});
        try self.writer.newline();

        try self.writer.writeLine("#include <metal_stdlib>");
        try self.writer.writeLine("using namespace metal;");
        try self.writer.newline();
    }

    fn writeUniformStruct(self: *Self, ir: *const kernel.KernelIR) !void {
        if (ir.uniforms.len == 0) return;

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

    fn writeKernelFunction(self: *Self, ir: *const kernel.KernelIR) !void {
        // Kernel attribute
        try self.writer.writeLine("kernel void");

        // Function name
        try self.writer.writeFmt("{s}(\n", .{ir.entry_point});
        self.writer.indent();

        // Buffer parameters
        for (ir.buffers, 0..) |buf, i| {
            try self.writer.writeIndent();
            if (buf.access == .read_only) {
                try self.writer.write("const ");
            }
            try self.writer.write("device ");
            try self.writeType(buf.element_type);
            try self.writer.writeFmt("* {s} [[buffer({d})]],\n", .{ buf.name, buf.binding });
            _ = i;
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

        // Shared memory declarations (threadgroup)
        for (ir.shared_memory) |shared| {
            try self.writer.writeIndent();
            try self.writer.write("threadgroup ");
            try self.writeType(shared.element_type);
            if (shared.size) |size| {
                try self.writer.writeFmt(" {s}[{d}];\n", .{ shared.name, size });
            } else {
                try self.writer.writeFmt(" {s}[];\n", .{shared.name});
            }
        }
        if (ir.shared_memory.len > 0) {
            try self.writer.newline();
        }

        // Kernel body
        for (ir.body) |s| {
            try self.writeStmt(s);
        }

        self.writer.dedent();
        try self.writer.writeLine("}");
    }

    fn writeType(self: *Self, ty: types.Type) backend.CodegenError!void {
        switch (ty) {
            .scalar => |s| {
                const type_name: []const u8 = switch (s) {
                    .bool_ => "bool",
                    .i8 => "int8_t",
                    .i16 => "int16_t",
                    .i32 => "int",
                    .i64 => "int64_t",
                    .u8 => "uint8_t",
                    .u16 => "uint16_t",
                    .u32 => "uint",
                    .u64 => "uint64_t",
                    .f16 => "half",
                    .f32 => "float",
                    .f64 => "double",
                };
                try self.writer.write(type_name);
            },
            .vector => |v| {
                const base: []const u8 = switch (v.element) {
                    .bool_ => "bool",
                    .i8 => "char",
                    .i16 => "short",
                    .i32 => "int",
                    .i64 => "long",
                    .u8 => "uchar",
                    .u16 => "ushort",
                    .u32 => "uint",
                    .u64 => "ulong",
                    .f16 => "half",
                    .f32 => "float",
                    .f64 => "double",
                };
                try self.writer.writeFmt("{s}{d}", .{ base, v.size });
            },
            .matrix => |m| {
                try self.writer.writeFmt("float{d}x{d}", .{ m.cols, m.rows });
            },
            .array => |a| {
                try self.writeType(a.element.*);
                if (a.size) |size| {
                    try self.writer.writeFmt("[{d}]", .{size});
                } else {
                    try self.writer.write("*");
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
                try self.writer.write("select(");
                try self.writeExpr(s.false_value);
                try self.writer.write(", ");
                try self.writeExpr(s.true_value);
                try self.writer.write(", ");
                try self.writeExpr(s.condition);
                try self.writer.write(")");
            },
            .vector_construct => |vc| {
                const base: []const u8 = switch (vc.element_type) {
                    .f32 => "float",
                    .i32 => "int",
                    .u32 => "uint",
                    else => "float",
                };
                try self.writer.writeFmt("{s}{d}(", .{ base, vc.size });
                for (vc.components, 0..) |comp, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.writeExpr(comp);
                }
                try self.writer.write(")");
            },
            .swizzle => |sw| {
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
            .abs => "abs",
            .sqrt => "sqrt",
            .sin => "sin",
            .cos => "cos",
            .tan => "tan",
            .asin => "asin",
            .acos => "acos",
            .atan => "atan",
            .sinh => "sinh",
            .cosh => "cosh",
            .tanh => "tanh",
            .exp => "exp",
            .exp2 => "exp2",
            .log => "log",
            .log2 => "log2",
            .log10 => "log10",
            .floor => "floor",
            .ceil => "ceil",
            .round => "round",
            .trunc => "trunc",
            .fract => "fract",
            .sign => "sign",
            .normalize => "normalize",
            .length => "length",
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
            .min => "min",
            .max => "max",
            .pow => "pow",
            .atan2 => "atan2",
            .dot => "dot",
            .cross => "cross",
            .distance => "distance",
            .step => "step",
            .reflect => "reflect",
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
            .barrier => try self.writer.write("threadgroup_barrier(mem_flags::mem_threadgroup)"),
            .memory_barrier => try self.writer.write("threadgroup_barrier(mem_flags::mem_device)"),
            .memory_barrier_buffer => try self.writer.write("threadgroup_barrier(mem_flags::mem_device)"),
            .memory_barrier_shared => try self.writer.write("threadgroup_barrier(mem_flags::mem_threadgroup)"),
            .atomic_add => {
                try self.writer.write("atomic_fetch_add_explicit(");
                if (c.args.len >= 2) {
                    try self.writer.write("(device atomic_uint*)&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                    try self.writer.write(", memory_order_relaxed)");
                }
            },
            .atomic_sub => {
                try self.writer.write("atomic_fetch_sub_explicit(");
                if (c.args.len >= 2) {
                    try self.writer.write("(device atomic_uint*)&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                    try self.writer.write(", memory_order_relaxed)");
                }
            },
            .atomic_and => {
                try self.writer.write("atomic_fetch_and_explicit(");
                if (c.args.len >= 2) {
                    try self.writer.write("(device atomic_uint*)&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                    try self.writer.write(", memory_order_relaxed)");
                }
            },
            .atomic_or => {
                try self.writer.write("atomic_fetch_or_explicit(");
                if (c.args.len >= 2) {
                    try self.writer.write("(device atomic_uint*)&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                    try self.writer.write(", memory_order_relaxed)");
                }
            },
            .atomic_xor => {
                try self.writer.write("atomic_fetch_xor_explicit(");
                if (c.args.len >= 2) {
                    try self.writer.write("(device atomic_uint*)&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                    try self.writer.write(", memory_order_relaxed)");
                }
            },
            .atomic_min => {
                try self.writer.write("atomic_fetch_min_explicit(");
                if (c.args.len >= 2) {
                    try self.writer.write("(device atomic_uint*)&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                    try self.writer.write(", memory_order_relaxed)");
                }
            },
            .atomic_max => {
                try self.writer.write("atomic_fetch_max_explicit(");
                if (c.args.len >= 2) {
                    try self.writer.write("(device atomic_uint*)&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                    try self.writer.write(", memory_order_relaxed)");
                }
            },
            .atomic_exchange => {
                try self.writer.write("atomic_exchange_explicit(");
                if (c.args.len >= 2) {
                    try self.writer.write("(device atomic_uint*)&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                    try self.writer.write(", memory_order_relaxed)");
                }
            },
            .atomic_compare_exchange => {
                try self.writer.write("atomic_compare_exchange_weak_explicit(");
                if (c.args.len >= 3) {
                    try self.writer.write("(device atomic_uint*)&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", &");
                    try self.writeExpr(c.args[1]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[2]);
                    try self.writer.write(", memory_order_relaxed, memory_order_relaxed)");
                }
            },
            .atomic_load => {
                try self.writer.write("atomic_load_explicit(");
                if (c.args.len >= 1) {
                    try self.writer.write("(device atomic_uint*)&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", memory_order_relaxed)");
                }
            },
            .atomic_store => {
                try self.writer.write("atomic_store_explicit(");
                if (c.args.len >= 2) {
                    try self.writer.write("(device atomic_uint*)&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                    try self.writer.write(", memory_order_relaxed)");
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
                try self.writer.write("mix(");
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
                try self.writer.write("fma(");
                for (c.args, 0..) |arg, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.writeExpr(arg);
                }
                try self.writer.write(")");
            },
            .select => {
                try self.writer.write("select(");
                if (c.args.len >= 3) {
                    try self.writeExpr(c.args[2]); // false
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]); // true
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[0]); // condition
                }
                try self.writer.write(")");
            },
            .all => {
                try self.writer.write("all(");
                if (c.args.len >= 1) try self.writeExpr(c.args[0]);
                try self.writer.write(")");
            },
            .any => {
                try self.writer.write("any(");
                if (c.args.len >= 1) try self.writeExpr(c.args[0]);
                try self.writer.write(")");
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
            .discard => try self.writer.writeLine("discard_fragment();"),
            .expr_stmt => |exp| {
                try self.writer.writeIndent();
                try self.writeExpr(exp);
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
            .expr_stmt => |exp| try self.writeExpr(exp),
            else => {},
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "MslGenerator basic kernel" {
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
