const stmt = @import("../../stmt.zig");
const expr = @import("../../expr.zig");
const backend = @import("../../codegen/backend.zig");

pub fn writeStmt(self: anytype, s: *const stmt.Stmt) backend.CodegenError!void {
    switch (s.*) {
        .var_decl => |v| try writeVarDecl(self, v),
        .assign => |a| try writeAssign(self, a),
        .compound_assign => |ca| try writeCompoundAssign(self, ca),
        .if_ => |i| try writeIf(self, i),
        .for_ => |f| try writeFor(self, f),
        .while_ => |w| try writeWhile(self, w),
        .do_while => |dw| try writeDoWhile(self, dw),
        .return_ => |r| try writeReturn(self, r),
        .break_ => try self.writer.writeLine("break;"),
        .continue_ => try writeContinue(self),
        .discard => try writeDiscard(self),
        .expr_stmt => |e| try writeExprStmt(self, e),
        .block => |b| try writeBlock(self, b),
        .switch_ => |sw| try writeSwitch(self, sw),
    }
}

fn writeVarDecl(self: anytype, v: stmt.Stmt.VarDecl) !void {
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

fn writeAssign(self: anytype, a: stmt.Stmt.Assignment) !void {
    try self.writer.writeIndent();
    try self.writeExpr(a.target);
    try self.writer.write(" = ");
    try self.writeExpr(a.value);
    try self.writer.write(";\n");
}

fn writeCompoundAssign(self: anytype, ca: stmt.Stmt.CompoundAssign) !void {
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

fn writeIf(self: anytype, i: stmt.Stmt.IfStmt) !void {
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

fn writeFor(self: anytype, f: stmt.Stmt.ForStmt) !void {
    try self.writer.writeIndent();
    try self.writer.write("for (");
    if (f.init) |init_stmt| {
        try writeStmtInline(self, init_stmt);
    }
    try self.writer.write("; ");
    if (f.condition) |cond| {
        try self.writeExpr(cond);
    }
    try self.writer.write("; ");
    if (f.update) |update| {
        try writeStmtInline(self, update);
    }
    try self.writer.write(") {\n");
    self.writer.indent();
    for (f.body) |body_stmt| {
        try self.writeStmt(body_stmt);
    }
    self.writer.dedent();
    try self.writer.writeLine("}");
}

fn writeWhile(self: anytype, w: stmt.Stmt.WhileStmt) !void {
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

fn writeDoWhile(self: anytype, dw: stmt.Stmt.DoWhileStmt) !void {
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

fn writeReturn(self: anytype, r: stmt.Stmt.Return) !void {
    try self.writer.writeIndent();
    try self.writer.write("return");
    if (r.value) |val| {
        try self.writer.write(" ");
        try self.writeExpr(val);
    }
    try self.writer.write(";\n");
}

fn writeContinue(self: anytype) !void {
    if (self.config.while_style == .loop_break) {
        try self.writer.writeLine("continuing;");
    } else {
        try self.writer.writeLine("continue;");
    }
}

fn writeDiscard(self: anytype) !void {
    if (self.config.discard_stmt.len > 0) {
        try self.writer.writeLine(self.config.discard_stmt);
    }
}

fn writeExprStmt(self: anytype, e: *const expr.Expr) !void {
    try self.writer.writeIndent();
    try self.writeExpr(e);
    try self.writer.write(";\n");
}

fn writeBlock(self: anytype, b: stmt.Stmt.Block) !void {
    try self.writer.writeLine("{");
    self.writer.indent();
    for (b.statements) |body_stmt| {
        try self.writeStmt(body_stmt);
    }
    self.writer.dedent();
    try self.writer.writeLine("}");
}

fn writeSwitch(self: anytype, sw: stmt.Stmt.SwitchStmt) !void {
    try self.writer.writeIndent();
    try self.writer.write("switch (");
    try self.writeExpr(sw.selector);
    try self.writer.write(") {\n");
    self.writer.indent();
    for (sw.cases) |case| {
        try self.writer.writeIndent();
        try self.writer.write("case ");

        try writeLiteral(self, case.value);
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

fn writeLiteral(self: anytype, lit: expr.Literal) !void {
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

fn writeStmtInline(self: anytype, s: *const stmt.Stmt) !void {
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
