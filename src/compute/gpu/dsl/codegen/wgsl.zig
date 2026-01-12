//! WGSL Code Generator
//!
//! Generates WGSL (WebGPU Shading Language) compute shader source code from kernel IR.

const std = @import("std");
const types = @import("../types.zig");
const expr = @import("../expr.zig");
const stmt = @import("../stmt.zig");
const kernel = @import("../kernel.zig");
const backend = @import("backend.zig");
const common = @import("common.zig");
const gpu_backend = @import("../../backend.zig");

/// WGSL code generator.
pub const WgslGenerator = struct {
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

    /// Generate WGSL source code from kernel IR.
    pub fn generate(
        self: *Self,
        ir: *const kernel.KernelIR,
    ) backend.CodegenError!backend.GeneratedSource {
        // Header comment
        try self.writer.writeLine("// Auto-generated WGSL compute shader");
        try self.writer.writeFmt("// Kernel: {s}\n", .{ir.name});
        try self.writer.newline();

        // Buffer declarations
        try self.writeBufferDeclarations(ir);

        // Uniform declarations
        try self.writeUniformDeclarations(ir);

        // Shared memory (workgroup) declarations
        try self.writeSharedMemory(ir);

        // Main function
        try self.writeMainFunction(ir);

        const code = try self.writer.getCode();
        const entry_point = try self.allocator.dupe(u8, "main");

        return .{
            .code = code,
            .entry_point = entry_point,
            .backend = .webgpu,
            .language = .wgsl,
        };
    }

    fn writeBufferDeclarations(self: *Self, ir: *const kernel.KernelIR) !void {
        for (ir.buffers) |buf| {
            // Access mode
            const access_str: []const u8 = switch (buf.access) {
                .read_only => "read",
                .write_only => "write",
                .read_write => "read_write",
            };

            try self.writer.writeFmt("@group({d}) @binding({d})\n", .{ buf.group, buf.binding });
            try self.writer.writeFmt("var<storage, {s}> {s}: array<", .{ access_str, buf.name });
            try self.writeType(buf.element_type);
            try self.writer.write(">;\n\n");
        }
    }

    fn writeUniformDeclarations(self: *Self, ir: *const kernel.KernelIR) !void {
        if (ir.uniforms.len == 0) return;

        // Group uniforms into a struct
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
        try self.writer.newline();

        // Binding for uniform buffer
        try self.writer.writeLine("@group(0) @binding(0)");
        try self.writer.writeLine("var<uniform> uniforms: Uniforms;");
        try self.writer.newline();
    }

    fn writeSharedMemory(self: *Self, ir: *const kernel.KernelIR) !void {
        for (ir.shared_memory) |shared| {
            try self.writer.write("var<workgroup> ");
            try self.writer.write(shared.name);
            try self.writer.write(": array<");
            try self.writeType(shared.element_type);
            if (shared.size) |size| {
                try self.writer.writeFmt(", {d}>;\n", .{size});
            } else {
                try self.writer.write(">;\n");
            }
        }
        if (ir.shared_memory.len > 0) {
            try self.writer.newline();
        }
    }

    fn writeMainFunction(self: *Self, ir: *const kernel.KernelIR) !void {
        // Workgroup size attribute
        try self.writer.writeFmt("@compute @workgroup_size({d}, {d}, {d})\n", .{
            ir.workgroup_size[0],
            ir.workgroup_size[1],
            ir.workgroup_size[2],
        });

        // Main function with built-in parameters
        try self.writer.writeLine("fn main(");
        self.writer.indent();
        try self.writer.writeLine("@builtin(global_invocation_id) globalInvocationId: vec3<u32>,");
        try self.writer.writeLine("@builtin(local_invocation_id) localInvocationId: vec3<u32>,");
        try self.writer.writeLine("@builtin(workgroup_id) workgroupId: vec3<u32>,");
        try self.writer.writeLine("@builtin(local_invocation_index) localInvocationIndex: u32,");
        try self.writer.writeLine("@builtin(num_workgroups) numWorkgroups: vec3<u32>,");
        self.writer.dedent();
        try self.writer.writeLine(") {");
        self.writer.indent();

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
                    .i8, .i16, .i32 => "i32",
                    .i64 => "i64",
                    .u8, .u16, .u32 => "u32",
                    .u64 => "u64",
                    .f16 => "f16",
                    .f32 => "f32",
                    .f64 => "f64",
                };
                try self.writer.write(type_name);
            },
            .vector => |v| {
                const elem_name: []const u8 = switch (v.element) {
                    .bool_ => "bool",
                    .i8, .i16, .i32 => "i32",
                    .i64 => "i64",
                    .u8, .u16, .u32 => "u32",
                    .u64 => "u64",
                    .f16 => "f16",
                    .f32 => "f32",
                    .f64 => "f64",
                };
                try self.writer.writeFmt("vec{d}<{s}>", .{ v.size, elem_name });
            },
            .matrix => |m| {
                try self.writer.writeFmt("mat{d}x{d}<f32>", .{ m.cols, m.rows });
            },
            .array => |a| {
                try self.writer.write("array<");
                try self.writeType(a.element.*);
                if (a.size) |size| {
                    try self.writer.writeFmt(", {d}>", .{size});
                } else {
                    try self.writer.write(">");
                }
            },
            .ptr => |p| {
                // WGSL uses reference types implicitly
                try self.writeType(p.pointee.*);
            },
            .void_ => {}, // No void type in WGSL
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
                    try self.writeExpr(un.operand);
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
                try self.writeType(c.target_type);
                try self.writer.write("(");
                try self.writeExpr(c.operand);
                try self.writer.write(")");
            },
            .select => |s| {
                try self.writer.write("select(");
                try self.writeExpr(s.false_value); // Note: WGSL select order is (false, true, cond)
                try self.writer.write(", ");
                try self.writeExpr(s.true_value);
                try self.writer.write(", ");
                try self.writeExpr(s.condition);
                try self.writer.write(")");
            },
            .vector_construct => |vc| {
                const elem_name: []const u8 = switch (vc.element_type) {
                    .f32 => "f32",
                    .i32 => "i32",
                    .u32 => "u32",
                    else => "f32",
                };
                try self.writer.writeFmt("vec{d}<{s}>(", .{ vc.size, elem_name });
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
            .i32_ => |v| try self.writer.writeFmt("{d}i", .{v}),
            .i64_ => |v| try self.writer.writeFmt("{d}i", .{v}),
            .u32_ => |v| try self.writer.writeFmt("{d}u", .{v}),
            .u64_ => |v| try self.writer.writeFmt("{d}u", .{v}),
            .f32_ => |v| {
                if (v == @trunc(v)) {
                    try self.writer.writeFmt("{d}.0", .{@as(i64, @intFromFloat(v))});
                } else {
                    try self.writer.writeFmt("{d}", .{v});
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
            .log10 => "log", // Need to compute manually
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
            .barrier => try self.writer.write("workgroupBarrier()"),
            .memory_barrier => try self.writer.write("storageBarrier()"),
            .memory_barrier_buffer => try self.writer.write("storageBarrier()"),
            .memory_barrier_shared => try self.writer.write("workgroupBarrier()"),
            .atomic_add => {
                try self.writer.write("atomicAdd(");
                if (c.args.len >= 2) {
                    try self.writer.write("&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
            },
            .atomic_sub => {
                try self.writer.write("atomicSub(");
                if (c.args.len >= 2) {
                    try self.writer.write("&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
            },
            .atomic_and => {
                try self.writer.write("atomicAnd(");
                if (c.args.len >= 2) {
                    try self.writer.write("&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
            },
            .atomic_or => {
                try self.writer.write("atomicOr(");
                if (c.args.len >= 2) {
                    try self.writer.write("&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
            },
            .atomic_xor => {
                try self.writer.write("atomicXor(");
                if (c.args.len >= 2) {
                    try self.writer.write("&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
            },
            .atomic_min => {
                try self.writer.write("atomicMin(");
                if (c.args.len >= 2) {
                    try self.writer.write("&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
            },
            .atomic_max => {
                try self.writer.write("atomicMax(");
                if (c.args.len >= 2) {
                    try self.writer.write("&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
            },
            .atomic_exchange => {
                try self.writer.write("atomicExchange(");
                if (c.args.len >= 2) {
                    try self.writer.write("&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
            },
            .atomic_compare_exchange => {
                try self.writer.write("atomicCompareExchangeWeak(");
                if (c.args.len >= 3) {
                    try self.writer.write("&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[2]);
                }
                try self.writer.write(")");
            },
            .atomic_load => {
                try self.writer.write("atomicLoad(");
                if (c.args.len >= 1) {
                    try self.writer.write("&");
                    try self.writeExpr(c.args[0]);
                }
                try self.writer.write(")");
            },
            .atomic_store => {
                try self.writer.write("atomicStore(");
                if (c.args.len >= 2) {
                    try self.writer.write("&");
                    try self.writeExpr(c.args[0]);
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]);
                }
                try self.writer.write(")");
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
                    try self.writeExpr(c.args[2]); // false value
                    try self.writer.write(", ");
                    try self.writeExpr(c.args[1]); // true value
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
                if (v.is_const) {
                    try self.writer.write("let ");
                } else {
                    try self.writer.write("var ");
                }
                try self.writer.writeFmt("{s}: ", .{v.name});
                try self.writeType(v.ty);
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
                // WGSL uses loop {} with break condition
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
            },
            .do_while => |dw| {
                // WGSL uses loop {} with break condition at end
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
            .discard => try self.writer.writeLine("discard;"),
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
                    try self.writer.write(": {\n");
                    self.writer.indent();
                    for (case.body) |body_stmt| {
                        try self.writeStmt(body_stmt);
                    }
                    self.writer.dedent();
                    try self.writer.writeLine("}");
                }
                if (sw.default) |default| {
                    try self.writer.writeLine("default: {");
                    self.writer.indent();
                    for (default) |body_stmt| {
                        try self.writeStmt(body_stmt);
                    }
                    self.writer.dedent();
                    try self.writer.writeLine("}");
                }
                self.writer.dedent();
                try self.writer.writeLine("}");
            },
        }
    }

    fn writeStmtInline(self: *Self, s: *const stmt.Stmt) !void {
        switch (s.*) {
            .var_decl => |v| {
                try self.writer.write("var ");
                try self.writer.writeFmt("{s}: ", .{v.name});
                try self.writeType(v.ty);
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

test "WgslGenerator basic kernel" {
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
    try std.testing.expect(std.mem.indexOf(u8, result.code, "fn main") != null);
}
