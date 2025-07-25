const std = @import("std");
const ast = @import("ast.zig");

pub const Interpreter = struct {
    allocator: std.mem.Allocator,
    variables: std.StringHashMap(i64),

    pub fn init(allocator: std.mem.Allocator) Interpreter {
        return Interpreter{ .allocator = allocator, .variables = std.StringHashMap(i64).init(allocator) };
    }

    pub fn evalProgram(self: *Interpreter, nodes: []const ast.Node) void {
        for (nodes) |n| {
            self.evalStmt(n);
        }
    }

    fn evalStmt(self: *Interpreter, node: ast.Node) void {
        switch (node) {
            .var_decl => |v| {
                const val = self.evalExpr(v.value.*);
                self.variables.put(v.name, val) catch {};
            },
            .print_stmt => |expr| {
                const val = self.evalExpr(expr.*);
                std.debug.print("{d}\n", .{val});
            },
            else => {},
        }
    }

    fn evalExpr(self: *Interpreter, node: ast.Node) i64 {
        switch (node) {
            .integer => |i| return i,
            .identifier => |name| return self.variables.get(name) orelse 0,
            .binary => |b| {
                const l = self.evalExpr(b.left.*);
                const r = self.evalExpr(b.right.*);
                return switch (b.op[0]) {
                    '+' => l + r,
                    '-' => l - r,
                    else => 0,
                };
            },
            else => return 0,
        }
    }
};
