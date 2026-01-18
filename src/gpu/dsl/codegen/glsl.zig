//! GLSL Code Generator
//!
//! Generates GLSL compute shader source code from kernel IR.
//! Targets Vulkan (GLSL 450) and OpenGL (GLSL 430+).

const std = @import("std");
const types = @import("../types.zig");
const expr = @import("../expr.zig");
const stmt = @import("../stmt.zig");
const kernel = @import("../kernel.zig");
const backend = @import("backend.zig");
const common = @import("common.zig");
const gpu_backend = @import("../../backend.zig");

/// Target GLSL variant.
pub const GlslTarget = enum {
    vulkan, // GLSL 450 with Vulkan extensions
    opengl, // GLSL 430 compute shaders
    opengles, // GLSL ES 310+ compute
};

/// GLSL code generator.
pub const GlslGenerator = struct {
    writer: common.CodeWriter,
    allocator: std.mem.Allocator,
    target: GlslTarget,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, target: GlslTarget) Self {
        return .{
            .writer = common.CodeWriter.init(allocator),
            .allocator = allocator,
            .target = target,
        };
    }

    pub fn deinit(self: *Self) void {
        self.writer.deinit();
    }

    /// Generate GLSL source code from kernel IR.
    pub fn generate(
        self: *Self,
        ir: *const kernel.KernelIR,
    ) backend.CodegenError!backend.GeneratedSource {
        // Version and extensions
        try self.writeHeader(ir);

        // Layout declarations
        try self.writeLayoutDeclarations(ir);

        // Buffer declarations
        try self.writeBufferDeclarations(ir);

        // Uniform declarations
        try self.writeUniformDeclarations(ir);

        // Shared memory declarations
        try self.writeSharedMemory(ir);

        // Main function
        try self.writeMainFunction(ir);

        const code = try self.writer.getCode();
        const entry_point = try self.allocator.dupe(u8, "main");

        const backend_type: gpu_backend.Backend = switch (self.target) {
            .vulkan => .vulkan,
            .opengl => .opengl,
            .opengles => .opengles,
        };

        return .{
            .code = code,
            .entry_point = entry_point,
            .backend = backend_type,
            .language = .glsl,
        };
    }

    fn writeHeader(self: *Self, ir: *const kernel.KernelIR) !void {
        // Version directive
        switch (self.target) {
            .vulkan => try self.writer.writeLine("#version 450"),
            .opengl => try self.writer.writeLine("#version 430"),
            .opengles => try self.writer.writeLine("#version 310 es"),
        }
        try self.writer.newline();

        // Comment
        try self.writer.writeLine("// Auto-generated GLSL compute shader");
        try self.writer.writeFmt("// Kernel: {s}\n", .{ir.name});
        try self.writer.newline();

        // Extensions
        if (self.target == .vulkan) {
            try self.writer.writeLine("#extension GL_ARB_separate_shader_objects : enable");
        }

        // Precision qualifiers for ES
        if (self.target == .opengles) {
            try self.writer.writeLine("precision highp float;");
            try self.writer.writeLine("precision highp int;");
        }

        try self.writer.newline();
    }

    fn writeLayoutDeclarations(self: *Self, ir: *const kernel.KernelIR) !void {
        // Workgroup size
        try self.writer.writeFmt("layout(local_size_x = {d}, local_size_y = {d}, local_size_z = {d}) in;\n", .{
            ir.workgroup_size[0],
            ir.workgroup_size[1],
            ir.workgroup_size[2],
        });
        try self.writer.newline();
    }

    fn writeBufferDeclarations(self: *Self, ir: *const kernel.KernelIR) !void {
        for (ir.buffers) |buf| {
            // Layout qualifier
            if (self.target == .vulkan) {
                try self.writer.writeFmt("layout(set = {d}, binding = {d}) ", .{ buf.group, buf.binding });
            } else {
                try self.writer.writeFmt("layout(binding = {d}) ", .{buf.binding});
            }

            // Access qualifier
            switch (buf.access) {
                .read_only => try self.writer.write("readonly "),
                .write_only => try self.writer.write("writeonly "),
                .read_write => {},
            }

            // Buffer block
            try self.writer.writeFmt("buffer {s}Buffer {{\n", .{buf.name});
            self.writer.indent();
            try self.writer.writeIndent();
            try self.writeType(buf.element_type);
            try self.writer.writeFmt(" {s}[];\n", .{buf.name});
            self.writer.dedent();
            try self.writer.writeLine("};");
            try self.writer.newline();
        }
    }

    fn writeUniformDeclarations(self: *Self, ir: *const kernel.KernelIR) !void {
        if (ir.uniforms.len == 0) return;

        // Uniforms as push constants (Vulkan) or uniform buffer
        if (self.target == .vulkan) {
            try self.writer.writeLine("layout(push_constant) uniform PushConstants {");
        } else {
            try self.writer.writeLine("layout(binding = 0) uniform Uniforms {");
        }

        self.writer.indent();
        for (ir.uniforms) |uni| {
            try self.writer.writeIndent();
            try self.writeType(uni.ty);
            try self.writer.writeFmt(" {s};\n", .{uni.name});
        }
        self.writer.dedent();

        if (self.target == .vulkan) {
            try self.writer.writeLine("} pc;");
        } else {
            try self.writer.writeLine("} uniforms;");
        }
        try self.writer.newline();
    }

    fn writeSharedMemory(self: *Self, ir: *const kernel.KernelIR) !void {
        for (ir.shared_memory) |shared| {
            try self.writer.write("shared ");
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
    }

    fn writeMainFunction(self: *Self, ir: *const kernel.KernelIR) !void {
        try self.writer.writeLine("void main() {");
        self.writer.indent();

        // Built-in variable aliases (for compatibility)
        try self.writer.writeLine("// Built-in variable aliases");
        try self.writer.writeLine("uvec3 globalInvocationId = gl_GlobalInvocationID;");
        try self.writer.writeLine("uvec3 localInvocationId = gl_LocalInvocationID;");
        try self.writer.writeLine("uvec3 workgroupId = gl_WorkGroupID;");
        try self.writer.writeLine("uint localInvocationIndex = gl_LocalInvocationIndex;");
        try self.writer.newline();

        // Kernel body
        for (ir.body) |s| {
            try self.writeStmt(s);
        }

        self.writer.dedent();
        try self.writer.writeLine("}");
    }

    fn writeType(self: *Self, ty: types.Type) backend.CodegenError!void {
        switch (ty) {
            .scalar => |s| try self.writer.write(common.TypeNames.glsl.getScalarName(s)),
            .vector => |v| {
                const prefix: []const u8 = switch (v.element) {
                    .f32, .f64 => "",
                    .i8, .i16, .i32, .i64 => "i",
                    .u8, .u16, .u32, .u64 => "u",
                    .bool_ => "b",
                    else => "",
                };
                try self.writer.writeFmt("{s}vec{d}", .{ prefix, v.size });
            },
            .matrix => |m| {
                if (m.rows == m.cols) {
                    try self.writer.writeFmt("mat{d}", .{m.rows});
                } else {
                    try self.writer.writeFmt("mat{d}x{d}", .{ m.cols, m.rows });
                }
            },
            .array => |a| {
                try self.writeType(a.element.*);
                if (a.size) |size| {
                    try self.writer.writeFmt("[{d}]", .{size});
                } else {
                    try self.writer.write("[]");
                }
            },
            .ptr => |p| {
                // GLSL doesn't have pointers, treat as reference
                try self.writeType(p.pointee.*);
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
                try self.writeType(c.target_type);
                try self.writer.write("(");
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
                const prefix: []const u8 = switch (vc.element_type) {
                    .f32, .f64 => "",
                    .i8, .i16, .i32, .i64 => "i",
                    .u8, .u16, .u32, .u64 => "u",
                    .bool_ => "b",
                    else => "",
                };
                try self.writer.writeFmt("{s}vec{d}(", .{ prefix, vc.size });
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
            .i64_ => |v| try self.writer.writeFmt("{d}L", .{v}),
            .u32_ => |v| try self.writer.writeFmt("{d}u", .{v}),
            .u64_ => |v| try self.writer.writeFmt("{d}UL", .{v}),
            .f32_ => |v| {
                if (v == @trunc(v)) {
                    try self.writer.writeFmt("{d}.0", .{@as(i64, @intFromFloat(v))});
                } else {
                    try self.writer.writeFmt("{d}", .{v});
                }
            },
            .f64_ => |v| {
                if (v == @trunc(v)) {
                    try self.writer.writeFmt("{d}.0lf", .{@as(i64, @intFromFloat(v))});
                } else {
                    try self.writer.writeFmt("{d}lf", .{v});
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
            .log10 => "log", // log10(x) = log(x) / log(10)
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
            .atan2 => "atan",
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
            .barrier => try self.writer.write("barrier()"),
            .memory_barrier => try self.writer.write("memoryBarrier()"),
            .memory_barrier_buffer => try self.writer.write("memoryBarrierBuffer()"),
            .memory_barrier_shared => try self.writer.write("memoryBarrierShared()"),
            .atomic_add, .atomic_sub, .atomic_and, .atomic_or, .atomic_xor, .atomic_min, .atomic_max, .atomic_exchange => {
                const func_name: []const u8 = switch (c.function) {
                    .atomic_add => "atomicAdd",
                    .atomic_sub => "atomicAdd", // Use negative value
                    .atomic_and => "atomicAnd",
                    .atomic_or => "atomicOr",
                    .atomic_xor => "atomicXor",
                    .atomic_min => "atomicMin",
                    .atomic_max => "atomicMax",
                    .atomic_exchange => "atomicExchange",
                    else => "atomic",
                };
                try self.writer.writeFmt("{s}(", .{func_name});
                for (c.args, 0..) |arg, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.writeExpr(arg);
                }
                try self.writer.write(")");
            },
            .atomic_compare_exchange => {
                try self.writer.write("atomicCompSwap(");
                for (c.args, 0..) |arg, i| {
                    if (i > 0) try self.writer.write(", ");
                    try self.writeExpr(arg);
                }
                try self.writer.write(")");
            },
            .atomic_load, .atomic_store => {
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
// Vision Kernel Code Generation
// ============================================================================

/// Vision kernel code generation utilities for GLSL.
/// These functions generate optimized GLSL compute shaders for vision operations.
pub const VisionKernels = struct {
    /// Generate a Conv2D GLSL compute shader.
    pub fn generateConv2d(allocator: std.mem.Allocator, target: GlslTarget) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        // Version header
        switch (target) {
            .vulkan => try writer.writeLine("#version 450"),
            .opengl => try writer.writeLine("#version 430"),
            .opengles => try writer.writeLine("#version 310 es"),
        }
        try writer.writeLine("// Auto-generated GLSL Conv2D compute shader");
        try writer.newline();

        if (target == .opengles) {
            try writer.writeLine("precision highp float;");
            try writer.writeLine("precision highp int;");
        }
        try writer.newline();

        try writer.writeLine("layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;");
        try writer.newline();

        // Buffer bindings
        try writer.writeLine("layout(set = 0, binding = 0) readonly buffer InputBuffer { float input_data[]; };");
        try writer.writeLine("layout(set = 0, binding = 1) readonly buffer WeightsBuffer { float weights[]; };");
        try writer.writeLine("layout(set = 0, binding = 2) readonly buffer BiasBuffer { float bias[]; };");
        try writer.writeLine("layout(set = 0, binding = 3) writeonly buffer OutputBuffer { float output_data[]; };");
        try writer.newline();

        // Push constants
        try writer.writeLine("layout(push_constant) uniform PushConstants {");
        try writer.writeLine("    int batch_size, in_channels, out_channels;");
        try writer.writeLine("    int in_height, in_width, out_height, out_width;");
        try writer.writeLine("    int kernel_h, kernel_w, stride_h, stride_w;");
        try writer.writeLine("    int pad_h, pad_w;");
        try writer.writeLine("} pc;");
        try writer.newline();

        try writer.writeLine("void main() {");
        try writer.writeLine("    int out_x = int(gl_GlobalInvocationID.x);");
        try writer.writeLine("    int out_y = int(gl_GlobalInvocationID.y);");
        try writer.writeLine("    int batch_oc = int(gl_GlobalInvocationID.z);");
        try writer.writeLine("    int batch_idx = batch_oc / pc.out_channels;");
        try writer.writeLine("    int oc = batch_oc % pc.out_channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= pc.out_width || out_y >= pc.out_height || batch_idx >= pc.batch_size) return;");
        try writer.newline();
        try writer.writeLine("    float sum = 0.0;");
        try writer.writeLine("    for (int ic = 0; ic < pc.in_channels; ++ic) {");
        try writer.writeLine("        for (int ky = 0; ky < pc.kernel_h; ++ky) {");
        try writer.writeLine("            for (int kx = 0; kx < pc.kernel_w; ++kx) {");
        try writer.writeLine("                int ih = out_y * pc.stride_h + ky - pc.pad_h;");
        try writer.writeLine("                int iw = out_x * pc.stride_w + kx - pc.pad_w;");
        try writer.writeLine("                if (ih >= 0 && ih < pc.in_height && iw >= 0 && iw < pc.in_width) {");
        try writer.writeLine("                    int input_idx = ((batch_idx * pc.in_channels + ic) * pc.in_height + ih) * pc.in_width + iw;");
        try writer.writeLine("                    int weight_idx = ((oc * pc.in_channels + ic) * pc.kernel_h + ky) * pc.kernel_w + kx;");
        try writer.writeLine("                    sum += input_data[input_idx] * weights[weight_idx];");
        try writer.writeLine("                }");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    sum += bias[oc];");
        try writer.writeLine("    int output_idx = ((batch_idx * pc.out_channels + oc) * pc.out_height + out_y) * pc.out_width + out_x;");
        try writer.writeLine("    output_data[output_idx] = sum;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a MaxPool2D GLSL compute shader.
    pub fn generateMaxPool2d(allocator: std.mem.Allocator, target: GlslTarget) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        switch (target) {
            .vulkan => try writer.writeLine("#version 450"),
            .opengl => try writer.writeLine("#version 430"),
            .opengles => try writer.writeLine("#version 310 es"),
        }
        try writer.writeLine("// Auto-generated GLSL MaxPool2D compute shader");
        try writer.newline();

        if (target == .opengles) {
            try writer.writeLine("precision highp float;");
            try writer.writeLine("precision highp int;");
        }
        try writer.newline();

        try writer.writeLine("layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;");
        try writer.newline();

        try writer.writeLine("layout(set = 0, binding = 0) readonly buffer InputBuffer { float input_data[]; };");
        try writer.writeLine("layout(set = 0, binding = 1) writeonly buffer OutputBuffer { float output_data[]; };");
        try writer.writeLine("layout(set = 0, binding = 2) writeonly buffer IndicesBuffer { int indices[]; };");
        try writer.newline();

        try writer.writeLine("layout(push_constant) uniform PushConstants {");
        try writer.writeLine("    int batch_size, channels;");
        try writer.writeLine("    int in_height, in_width, out_height, out_width;");
        try writer.writeLine("    int kernel_size, stride, padding;");
        try writer.writeLine("} pc;");
        try writer.newline();

        try writer.writeLine("void main() {");
        try writer.writeLine("    int out_x = int(gl_GlobalInvocationID.x);");
        try writer.writeLine("    int out_y = int(gl_GlobalInvocationID.y);");
        try writer.writeLine("    int bc = int(gl_GlobalInvocationID.z);");
        try writer.writeLine("    int batch_idx = bc / pc.channels;");
        try writer.writeLine("    int channel = bc % pc.channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= pc.out_width || out_y >= pc.out_height || batch_idx >= pc.batch_size) return;");
        try writer.newline();
        try writer.writeLine("    float max_val = -3.4028235e+38;");
        try writer.writeLine("    int max_idx = 0;");
        try writer.writeLine("    for (int ky = 0; ky < pc.kernel_size; ++ky) {");
        try writer.writeLine("        for (int kx = 0; kx < pc.kernel_size; ++kx) {");
        try writer.writeLine("            int ih = out_y * pc.stride + ky - pc.padding;");
        try writer.writeLine("            int iw = out_x * pc.stride + kx - pc.padding;");
        try writer.writeLine("            if (ih >= 0 && ih < pc.in_height && iw >= 0 && iw < pc.in_width) {");
        try writer.writeLine("                int input_idx = ((batch_idx * pc.channels + channel) * pc.in_height + ih) * pc.in_width + iw;");
        try writer.writeLine("                float val = input_data[input_idx];");
        try writer.writeLine("                if (val > max_val) { max_val = val; max_idx = input_idx; }");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    int output_idx = ((batch_idx * pc.channels + channel) * pc.out_height + out_y) * pc.out_width + out_x;");
        try writer.writeLine("    output_data[output_idx] = max_val;");
        try writer.writeLine("    indices[output_idx] = max_idx;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate an AvgPool2D GLSL compute shader.
    pub fn generateAvgPool2d(allocator: std.mem.Allocator, target: GlslTarget) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        switch (target) {
            .vulkan => try writer.writeLine("#version 450"),
            .opengl => try writer.writeLine("#version 430"),
            .opengles => try writer.writeLine("#version 310 es"),
        }
        try writer.writeLine("// Auto-generated GLSL AvgPool2D compute shader");
        try writer.newline();

        if (target == .opengles) {
            try writer.writeLine("precision highp float;");
            try writer.writeLine("precision highp int;");
        }
        try writer.newline();

        try writer.writeLine("layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;");
        try writer.newline();

        try writer.writeLine("layout(set = 0, binding = 0) readonly buffer InputBuffer { float input_data[]; };");
        try writer.writeLine("layout(set = 0, binding = 1) writeonly buffer OutputBuffer { float output_data[]; };");
        try writer.newline();

        try writer.writeLine("layout(push_constant) uniform PushConstants {");
        try writer.writeLine("    int batch_size, channels;");
        try writer.writeLine("    int in_height, in_width, out_height, out_width;");
        try writer.writeLine("    int kernel_size, stride, padding;");
        try writer.writeLine("} pc;");
        try writer.newline();

        try writer.writeLine("void main() {");
        try writer.writeLine("    int out_x = int(gl_GlobalInvocationID.x);");
        try writer.writeLine("    int out_y = int(gl_GlobalInvocationID.y);");
        try writer.writeLine("    int bc = int(gl_GlobalInvocationID.z);");
        try writer.writeLine("    int batch_idx = bc / pc.channels;");
        try writer.writeLine("    int channel = bc % pc.channels;");
        try writer.newline();
        try writer.writeLine("    if (out_x >= pc.out_width || out_y >= pc.out_height || batch_idx >= pc.batch_size) return;");
        try writer.newline();
        try writer.writeLine("    float sum = 0.0;");
        try writer.writeLine("    int count = 0;");
        try writer.writeLine("    for (int ky = 0; ky < pc.kernel_size; ++ky) {");
        try writer.writeLine("        for (int kx = 0; kx < pc.kernel_size; ++kx) {");
        try writer.writeLine("            int ih = out_y * pc.stride + ky - pc.padding;");
        try writer.writeLine("            int iw = out_x * pc.stride + kx - pc.padding;");
        try writer.writeLine("            if (ih >= 0 && ih < pc.in_height && iw >= 0 && iw < pc.in_width) {");
        try writer.writeLine("                int input_idx = ((batch_idx * pc.channels + channel) * pc.in_height + ih) * pc.in_width + iw;");
        try writer.writeLine("                sum += input_data[input_idx];");
        try writer.writeLine("                count++;");
        try writer.writeLine("            }");
        try writer.writeLine("        }");
        try writer.writeLine("    }");
        try writer.writeLine("    int output_idx = ((batch_idx * pc.channels + channel) * pc.out_height + out_y) * pc.out_width + out_x;");
        try writer.writeLine("    output_data[output_idx] = count > 0 ? sum / float(count) : 0.0;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a BatchNorm2D GLSL compute shader (inference mode).
    pub fn generateBatchNorm2d(allocator: std.mem.Allocator, target: GlslTarget) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        switch (target) {
            .vulkan => try writer.writeLine("#version 450"),
            .opengl => try writer.writeLine("#version 430"),
            .opengles => try writer.writeLine("#version 310 es"),
        }
        try writer.writeLine("// Auto-generated GLSL BatchNorm2D compute shader (inference)");
        try writer.newline();

        if (target == .opengles) {
            try writer.writeLine("precision highp float;");
            try writer.writeLine("precision highp int;");
        }
        try writer.newline();

        try writer.writeLine("layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;");
        try writer.newline();

        try writer.writeLine("layout(set = 0, binding = 0) readonly buffer InputBuffer { float input_data[]; };");
        try writer.writeLine("layout(set = 0, binding = 1) readonly buffer GammaBuffer { float gamma[]; };");
        try writer.writeLine("layout(set = 0, binding = 2) readonly buffer BetaBuffer { float beta[]; };");
        try writer.writeLine("layout(set = 0, binding = 3) readonly buffer MeanBuffer { float running_mean[]; };");
        try writer.writeLine("layout(set = 0, binding = 4) readonly buffer VarBuffer { float running_var[]; };");
        try writer.writeLine("layout(set = 0, binding = 5) writeonly buffer OutputBuffer { float output_data[]; };");
        try writer.newline();

        try writer.writeLine("layout(push_constant) uniform PushConstants {");
        try writer.writeLine("    int batch_size, channels, height, width;");
        try writer.writeLine("    float epsilon;");
        try writer.writeLine("} pc;");
        try writer.newline();

        try writer.writeLine("void main() {");
        try writer.writeLine("    int idx = int(gl_GlobalInvocationID.x);");
        try writer.writeLine("    int total = pc.batch_size * pc.channels * pc.height * pc.width;");
        try writer.writeLine("    if (idx >= total) return;");
        try writer.newline();
        try writer.writeLine("    int hw = pc.height * pc.width;");
        try writer.writeLine("    int c = (idx / hw) % pc.channels;");
        try writer.newline();
        try writer.writeLine("    float x = input_data[idx];");
        try writer.writeLine("    float mean = running_mean[c];");
        try writer.writeLine("    float var_val = running_var[c];");
        try writer.writeLine("    float g = gamma[c];");
        try writer.writeLine("    float b = beta[c];");
        try writer.newline();
        try writer.writeLine("    float normalized = (x - mean) * inversesqrt(var_val + pc.epsilon);");
        try writer.writeLine("    output_data[idx] = g * normalized + b;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate an im2col GLSL compute shader.
    pub fn generateIm2col(allocator: std.mem.Allocator, target: GlslTarget) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        switch (target) {
            .vulkan => try writer.writeLine("#version 450"),
            .opengl => try writer.writeLine("#version 430"),
            .opengles => try writer.writeLine("#version 310 es"),
        }
        try writer.writeLine("// Auto-generated GLSL im2col compute shader");
        try writer.newline();

        if (target == .opengles) {
            try writer.writeLine("precision highp float;");
            try writer.writeLine("precision highp int;");
        }
        try writer.newline();

        try writer.writeLine("layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;");
        try writer.newline();

        try writer.writeLine("layout(set = 0, binding = 0) readonly buffer InputBuffer { float input_data[]; };");
        try writer.writeLine("layout(set = 0, binding = 1) writeonly buffer OutputBuffer { float output_data[]; };");
        try writer.newline();

        try writer.writeLine("layout(push_constant) uniform PushConstants {");
        try writer.writeLine("    int batch_size, channels;");
        try writer.writeLine("    int in_height, in_width, out_height, out_width;");
        try writer.writeLine("    int kernel_h, kernel_w, stride_h, stride_w;");
        try writer.writeLine("    int pad_h, pad_w;");
        try writer.writeLine("} pc;");
        try writer.newline();

        try writer.writeLine("void main() {");
        try writer.writeLine("    int idx = int(gl_GlobalInvocationID.x);");
        try writer.writeLine("    int kernel_hw = pc.kernel_h * pc.kernel_w;");
        try writer.writeLine("    int col_h = pc.channels * kernel_hw;");
        try writer.writeLine("    int col_w = pc.out_height * pc.out_width;");
        try writer.writeLine("    int col_size = col_h * col_w;");
        try writer.writeLine("    int total = pc.batch_size * col_size;");
        try writer.writeLine("    if (idx >= total) return;");
        try writer.newline();
        try writer.writeLine("    int batch_idx = idx / col_size;");
        try writer.writeLine("    int idx_in_batch = idx % col_size;");
        try writer.writeLine("    int row = idx_in_batch / col_w;");
        try writer.writeLine("    int col = idx_in_batch % col_w;");
        try writer.newline();
        try writer.writeLine("    int c = row / kernel_hw;");
        try writer.writeLine("    int row_in_kernel = row % kernel_hw;");
        try writer.writeLine("    int ky = row_in_kernel / pc.kernel_w;");
        try writer.writeLine("    int kx = row_in_kernel % pc.kernel_w;");
        try writer.newline();
        try writer.writeLine("    int oh = col / pc.out_width;");
        try writer.writeLine("    int ow = col % pc.out_width;");
        try writer.newline();
        try writer.writeLine("    int ih = oh * pc.stride_h + ky - pc.pad_h;");
        try writer.writeLine("    int iw = ow * pc.stride_w + kx - pc.pad_w;");
        try writer.newline();
        try writer.writeLine("    float val = 0.0;");
        try writer.writeLine("    if (ih >= 0 && ih < pc.in_height && iw >= 0 && iw < pc.in_width) {");
        try writer.writeLine("        int input_idx = ((batch_idx * pc.channels + c) * pc.in_height + ih) * pc.in_width + iw;");
        try writer.writeLine("        val = input_data[input_idx];");
        try writer.writeLine("    }");
        try writer.writeLine("    output_data[idx] = val;");
        try writer.writeLine("}");

        return writer.getCode();
    }

    /// Generate a col2im GLSL compute shader.
    pub fn generateCol2im(allocator: std.mem.Allocator, target: GlslTarget) ![]const u8 {
        var writer = common.CodeWriter.init(allocator);
        defer writer.deinit();

        switch (target) {
            .vulkan => try writer.writeLine("#version 450"),
            .opengl => try writer.writeLine("#version 430"),
            .opengles => try writer.writeLine("#version 310 es"),
        }
        try writer.writeLine("// Auto-generated GLSL col2im compute shader");
        try writer.newline();

        if (target == .opengles) {
            try writer.writeLine("precision highp float;");
            try writer.writeLine("precision highp int;");
        }
        try writer.newline();

        try writer.writeLine("layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;");
        try writer.newline();

        try writer.writeLine("layout(set = 0, binding = 0) readonly buffer ColInputBuffer { float col_input[]; };");
        try writer.writeLine("layout(set = 0, binding = 1) buffer OutputBuffer { float output_data[]; };");
        try writer.newline();

        try writer.writeLine("layout(push_constant) uniform PushConstants {");
        try writer.writeLine("    int batch_size, channels;");
        try writer.writeLine("    int in_height, in_width, out_height, out_width;");
        try writer.writeLine("    int kernel_h, kernel_w, stride_h, stride_w;");
        try writer.writeLine("    int pad_h, pad_w;");
        try writer.writeLine("} pc;");
        try writer.newline();

        try writer.writeLine("void main() {");
        try writer.writeLine("    int idx = int(gl_GlobalInvocationID.x);");
        try writer.writeLine("    int kernel_hw = pc.kernel_h * pc.kernel_w;");
        try writer.writeLine("    int col_h = pc.channels * kernel_hw;");
        try writer.writeLine("    int col_w = pc.out_height * pc.out_width;");
        try writer.writeLine("    int col_size = col_h * col_w;");
        try writer.writeLine("    int total = pc.batch_size * col_size;");
        try writer.writeLine("    if (idx >= total) return;");
        try writer.newline();
        try writer.writeLine("    int batch_idx = idx / col_size;");
        try writer.writeLine("    int idx_in_batch = idx % col_size;");
        try writer.writeLine("    int row = idx_in_batch / col_w;");
        try writer.writeLine("    int col = idx_in_batch % col_w;");
        try writer.newline();
        try writer.writeLine("    int c = row / kernel_hw;");
        try writer.writeLine("    int row_in_kernel = row % kernel_hw;");
        try writer.writeLine("    int ky = row_in_kernel / pc.kernel_w;");
        try writer.writeLine("    int kx = row_in_kernel % pc.kernel_w;");
        try writer.newline();
        try writer.writeLine("    int oh = col / pc.out_width;");
        try writer.writeLine("    int ow = col % pc.out_width;");
        try writer.newline();
        try writer.writeLine("    int ih = oh * pc.stride_h + ky - pc.pad_h;");
        try writer.writeLine("    int iw = ow * pc.stride_w + kx - pc.pad_w;");
        try writer.newline();
        try writer.writeLine("    if (ih >= 0 && ih < pc.in_height && iw >= 0 && iw < pc.in_width) {");
        try writer.writeLine("        int output_idx = ((batch_idx * pc.channels + c) * pc.in_height + ih) * pc.in_width + iw;");
        try writer.writeLine("        atomicAdd(output_data[output_idx], col_input[idx]);");
        try writer.writeLine("    }");
        try writer.writeLine("}");

        return writer.getCode();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "GlslGenerator Vulkan basic kernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = GlslGenerator.init(allocator, .vulkan);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "#version 450") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.code, "void main()") != null);
}

test "GlslGenerator OpenGL basic kernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gen = GlslGenerator.init(allocator, .opengl);
    defer gen.deinit();

    const ir = kernel.KernelIR.empty("test_kernel");
    var result = try gen.generate(&ir);
    defer result.deinit(allocator);

    try std.testing.expect(std.mem.indexOf(u8, result.code, "#version 430") != null);
}
