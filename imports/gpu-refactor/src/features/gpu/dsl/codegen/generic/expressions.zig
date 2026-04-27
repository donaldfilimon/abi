const std = @import("std");
const expr = @import("../../expr.zig");
const backend = @import("../../codegen/backend.zig");
const common = @import("../../codegen/common.zig");

pub fn writeExpr(self: anytype, e: *const expr.Expr) backend.CodegenError!void {
    switch (e.*) {
        .literal => |lit| try writeLiteral(self, lit),
        .ref => |ref| try writeRef(self, ref),
        .unary => |un| try writeUnary(self, un),
        .binary => |bin| try writeBinary(self, bin),
        .call => |c| try writeCall(self, c),
        .index => |idx| try writeIndex(self, idx),
        .field => |f| try writeField(self, f),
        .cast => |c| try writeCast(self, c),
        .select => |s| try writeSelect(self, s),
        .vector_construct => |vc| try writeVectorConstruct(self, vc),
        .swizzle => |sw| try writeSwizzle(self, sw),
    }
}

fn writeRef(self: anytype, ref: expr.ValueRef) !void {
    if (ref.name) |name| {
        try self.writer.write(name);
    } else {
        try self.writer.writeFmt("_v{d}", .{ref.id});
    }
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

fn writeUnary(self: anytype, un: expr.Expr.UnaryExpr) !void {
    if (un.op.isPrefix()) {
        try self.writer.write(common.OperatorSymbols.unaryOp(un.op));
        try self.writer.write("(");
        try self.writeExpr(un.operand);
        try self.writer.write(")");
    } else {
        // Function-style unary op
        const func_name = @TypeOf(self.*).unary_fn_table[@intFromEnum(un.op)];
        try self.writer.writeFmt("{s}(", .{func_name});
        try self.writeExpr(un.operand);
        try self.writer.write(")");
    }
}

fn writeBinary(self: anytype, bin: expr.Expr.BinaryExpr) !void {
    if (bin.op.isInfix()) {
        try self.writer.write("(");
        try self.writeExpr(bin.left);
        try self.writer.write(common.OperatorSymbols.binaryOp(bin.op));
        try self.writeExpr(bin.right);
        try self.writer.write(")");
    } else {
        // Function-style binary op
        const func_name = @TypeOf(self.*).binary_fn_table[@intFromEnum(bin.op)];
        try self.writer.writeFmt("{s}(", .{func_name});
        try self.writeExpr(bin.left);
        try self.writer.write(", ");
        try self.writeExpr(bin.right);
        try self.writer.write(")");
    }
}

fn writeCall(self: anytype, c: expr.Expr.CallExpr) backend.CodegenError!void {
    switch (c.function) {
        .barrier => try self.writer.write(self.config.barriers.barrier),
        .memory_barrier => try self.writer.write(self.config.barriers.memory_barrier),
        .memory_barrier_buffer => try self.writer.write(self.config.barriers.memory_barrier_buffer),
        .memory_barrier_shared => try self.writer.write(self.config.barriers.memory_barrier_shared),
        .atomic_add => try writeAtomicOp(self, self.config.atomics.add_fn, c.args),
        .atomic_sub => try writeAtomicSub(self, c.args),
        .atomic_and => try writeAtomicOp(self, self.config.atomics.and_fn, c.args),
        .atomic_or => try writeAtomicOp(self, self.config.atomics.or_fn, c.args),
        .atomic_xor => try writeAtomicOp(self, self.config.atomics.xor_fn, c.args),
        .atomic_min => try writeAtomicOp(self, self.config.atomics.min_fn, c.args),
        .atomic_max => try writeAtomicOp(self, self.config.atomics.max_fn, c.args),
        .atomic_exchange => try writeAtomicOp(self, self.config.atomics.exchange_fn, c.args),
        .atomic_compare_exchange => try writeAtomicCompareExchange(self, c.args),
        .atomic_load => try writeAtomicLoad(self, c.args),
        .atomic_store => try writeAtomicStore(self, c.args),
        .clamp => try writeBuiltinCall(self, self.config.builtin_fns.clamp, c.args),
        .mix => try writeBuiltinCall(self, self.config.builtin_fns.mix, c.args),
        .smoothstep => try writeBuiltinCall(self, self.config.builtin_fns.smoothstep, c.args),
        .fma => try writeBuiltinCall(self, self.config.builtin_fns.fma, c.args),
        .select => try writeSelectCall(self, c.args),
        .all => try writeBuiltinCall(self, self.config.builtin_fns.all, c.args),
        .any => try writeBuiltinCall(self, self.config.builtin_fns.any, c.args),
    }
}

fn writeAtomicSub(self: anytype, args: []const *const expr.Expr) !void {
    const sub_fn = self.config.atomics.sub_fn;
    const add_fn = self.config.atomics.add_fn;
    const negate = std.mem.eql(u8, sub_fn, add_fn);

    try self.writer.writeFmt("{s}(", .{sub_fn});
    if (args.len >= 2) {
        if (self.config.atomics.needs_cast) {
            try self.writer.write(self.config.atomics.cast_template);
        }
        try self.writer.write(self.config.atomics.ptr_prefix);
        try self.writeExpr(args[0]);
        try self.writer.write(", ");
        if (negate) try self.writer.write("-(");
        try self.writeExpr(args[1]);
        if (negate) try self.writer.write(")");
    }
    try self.writer.write(self.config.atomics.suffix);
}

fn writeAtomicOp(self: anytype, func_name: []const u8, args: []const *const expr.Expr) !void {
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

fn writeAtomicCompareExchange(self: anytype, args: []const *const expr.Expr) !void {
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

fn writeAtomicLoad(self: anytype, args: []const *const expr.Expr) !void {
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

fn writeAtomicStore(self: anytype, args: []const *const expr.Expr) !void {
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

fn writeBuiltinCall(self: anytype, func_name: []const u8, args: []const *const expr.Expr) !void {
    try self.writer.writeFmt("{s}(", .{func_name});
    for (args, 0..) |arg, i| {
        if (i > 0) try self.writer.write(", ");
        try self.writeExpr(arg);
    }
    try self.writer.write(")");
}

fn writeSelectCall(self: anytype, args: []const *const expr.Expr) !void {
    if (self.config.select_reversed) {
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

fn writeIndex(self: anytype, idx: expr.Expr.IndexExpr) !void {
    try self.writeExpr(idx.base);
    try self.writer.write("[");
    try self.writeExpr(idx.index);
    try self.writer.write("]");
}

fn writeField(self: anytype, f: expr.Expr.FieldExpr) !void {
    try self.writeExpr(f.base);
    try self.writer.writeFmt(".{s}", .{f.field});
}

fn writeCast(self: anytype, c: expr.Expr.CastExpr) !void {
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

fn writeSelect(self: anytype, s: expr.Expr.SelectExpr) !void {
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

fn writeVectorConstruct(self: anytype, vc: expr.Expr.VectorConstruct) !void {
    switch (self.config.language) {
        .cuda => {
            const base = @TypeOf(self.*).type_name_table[@intFromEnum(vc.element_type)];
            try self.writer.writeFmt("make_{s}{d}(", .{ base, vc.size });
        },
        .wgsl => {
            try self.writer.writeFmt("vec{d}<", .{vc.size});
            try self.writer.write(@TypeOf(self.*).type_name_table[@intFromEnum(vc.element_type)]);
            try self.writer.write(">(");
        },
        else => {
            const prefix = @TypeOf(self.*).vector_prefix_table[@intFromEnum(vc.element_type)];
            try self.writer.writeFmt("{s}{d}(", .{ prefix, vc.size });
        },
    }
    for (vc.components, 0..) |comp, i| {
        if (i > 0) try self.writer.write(", ");
        try self.writeExpr(comp);
    }
    try self.writer.write(")");
}

fn writeSwizzle(self: anytype, sw: expr.Expr.SwizzleExpr) !void {
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
