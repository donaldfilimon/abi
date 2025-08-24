//! Cell Language - A simple cell-based programming language
//!
//! This module provides an interpreter for a minimal programming language
//! inspired by functional programming and spreadsheet-like cell evaluation.

const std = @import("std");
const build_options = @import("build_options");

/// Cell value types
pub const CellValue = union(enum) {
    number: f64,
    string: []const u8,
    boolean: bool,
    none: void,

    pub fn format(self: CellValue, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        switch (self) {
            .number => |n| try writer.print("{d}", .{n}),
            .string => |s| try writer.print("\"{s}\"", .{s}),
            .boolean => |b| try writer.print("{}", .{b}),
            .none => try writer.print("none"),
        }
    }

    pub fn isTrue(self: CellValue) bool {
        return switch (self) {
            .boolean => |b| b,
            .number => |n| n != 0.0,
            .string => |s| s.len > 0,
            .none => false,
        };
    }
};

/// Cell expression types
pub const Expression = union(enum) {
    literal: CellValue,
    variable: []const u8,
    binary_op: struct {
        op: BinaryOp,
        left: *Expression,
        right: *Expression,
    },
    function_call: struct {
        name: []const u8,
        args: []Expression,
    },

    pub const BinaryOp = enum {
        add,
        subtract,
        multiply,
        divide,
        equal,
        not_equal,
        less_than,
        greater_than,
    };
};

/// Cell environment for variable storage
pub const Environment = struct {
    allocator: std.mem.Allocator,
    variables: std.StringHashMap(CellValue),

    pub fn init(allocator: std.mem.Allocator) Environment {
        return .{
            .allocator = allocator,
            .variables = std.StringHashMap(CellValue).init(allocator),
        };
    }

    pub fn deinit(self: *Environment) void {
        // Free string values
        var iterator = self.variables.iterator();
        while (iterator.next()) |entry| {
            switch (entry.value_ptr.*) {
                .string => |s| self.allocator.free(s),
                else => {},
            }
        }
        self.variables.deinit();
    }

    pub fn set(self: *Environment, name: []const u8, value: CellValue) !void {
        const owned_name = try self.allocator.dupe(u8, name);
        const owned_value = switch (value) {
            .string => |s| CellValue{ .string = try self.allocator.dupe(u8, s) },
            else => value,
        };
        try self.variables.put(owned_name, owned_value);
    }

    pub fn get(self: *Environment, name: []const u8) ?CellValue {
        return self.variables.get(name);
    }
};

/// Cell interpreter
pub const Interpreter = struct {
    allocator: std.mem.Allocator,
    environment: Environment,

    pub fn init(allocator: std.mem.Allocator) Interpreter {
        return .{
            .allocator = allocator,
            .environment = Environment.init(allocator),
        };
    }

    pub fn deinit(self: *Interpreter) void {
        self.environment.deinit();
    }

    pub fn evaluate(self: *Interpreter, expr: Expression) !CellValue {
        switch (expr) {
            .literal => |value| return value,
            .variable => |name| {
                return self.environment.get(name) orelse {
                    std.log.err("Undefined variable: {s}", .{name});
                    return error.UndefinedVariable;
                };
            },
            .binary_op => |op| {
                const left = try self.evaluate(op.left.*);
                const right = try self.evaluate(op.right.*);
                return evaluateBinaryOp(op.op, left, right);
            },
            .function_call => |call| {
                return self.evaluateFunctionCall(call.name, call.args);
            },
        }
    }

    fn evaluateBinaryOp(op: Expression.BinaryOp, left: CellValue, right: CellValue) !CellValue {
        switch (op) {
            .add => {
                if (left == .number and right == .number) {
                    return CellValue{ .number = left.number + right.number };
                }
                return error.InvalidOperation;
            },
            .subtract => {
                if (left == .number and right == .number) {
                    return CellValue{ .number = left.number - right.number };
                }
                return error.InvalidOperation;
            },
            .multiply => {
                if (left == .number and right == .number) {
                    return CellValue{ .number = left.number * right.number };
                }
                return error.InvalidOperation;
            },
            .divide => {
                if (left == .number and right == .number) {
                    if (right.number == 0.0) return error.DivisionByZero;
                    return CellValue{ .number = left.number / right.number };
                }
                return error.InvalidOperation;
            },
            .equal => {
                return CellValue{ .boolean = valuesEqual(left, right) };
            },
            .not_equal => {
                return CellValue{ .boolean = !valuesEqual(left, right) };
            },
            .less_than => {
                if (left == .number and right == .number) {
                    return CellValue{ .boolean = left.number < right.number };
                }
                return error.InvalidOperation;
            },
            .greater_than => {
                if (left == .number and right == .number) {
                    return CellValue{ .boolean = left.number > right.number };
                }
                return error.InvalidOperation;
            },
        }
    }

    fn valuesEqual(left: CellValue, right: CellValue) bool {
        if (@as(std.meta.Tag(CellValue), left) != @as(std.meta.Tag(CellValue), right)) {
            return false;
        }
        return switch (left) {
            .number => left.number == right.number,
            .string => std.mem.eql(u8, left.string, right.string),
            .boolean => left.boolean == right.boolean,
            .none => true,
        };
    }

    fn evaluateFunctionCall(self: *Interpreter, name: []const u8, args: []Expression) !CellValue {
        if (std.mem.eql(u8, name, "print")) {
            if (args.len != 1) return error.InvalidArgumentCount;
            const value = try self.evaluate(args[0]);
            std.debug.print("{}\n", .{value});
            return CellValue{ .none = {} };
        } else if (std.mem.eql(u8, name, "sqrt")) {
            if (args.len != 1) return error.InvalidArgumentCount;
            const value = try self.evaluate(args[0]);
            if (value != .number) return error.InvalidArgument;
            return CellValue{ .number = @sqrt(value.number) };
        } else if (std.mem.eql(u8, name, "abs")) {
            if (args.len != 1) return error.InvalidArgumentCount;
            const value = try self.evaluate(args[0]);
            if (value != .number) return error.InvalidArgument;
            return CellValue{ .number = @abs(value.number) };
        }

        std.log.err("Unknown function: {s}", .{name});
        return error.UnknownFunction;
    }

    pub fn setVariable(self: *Interpreter, name: []const u8, value: CellValue) !void {
        try self.environment.set(name, value);
    }
};

/// Simple REPL for interactive Cell programming
pub const REPL = struct {
    interpreter: Interpreter,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) REPL {
        return .{
            .interpreter = Interpreter.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *REPL) void {
        self.interpreter.deinit();
    }

    pub fn run(self: *REPL) !void {
        const stdout = std.io.getStdOut().writer();
        const stdin = std.io.getStdIn().reader();

        try stdout.writeAll("Cell Language REPL v1.0\n");
        try stdout.writeAll("Type 'exit' to quit.\n\n");

        // Set some example variables
        try self.interpreter.setVariable("pi", CellValue{ .number = std.math.pi });
        try self.interpreter.setVariable("greeting", CellValue{ .string = "Hello, Cell!" });

        while (true) {
            try stdout.writeAll("cell> ");

            const input = try stdin.readUntilDelimiterAlloc(self.allocator, '\n', 1024);
            defer self.allocator.free(input);

            const trimmed = std.mem.trim(u8, input, " \t\r\n");
            if (trimmed.len == 0) continue;
            if (std.mem.eql(u8, trimmed, "exit")) break;

            // Simple demo - just echo back for now
            // In a full implementation, this would parse and evaluate expressions
            if (std.mem.startsWith(u8, trimmed, "set ")) {
                // Simple variable assignment: set x 42
                const parts = std.mem.split(u8, trimmed[4..], " ");
                var part_iter = parts;

                const var_name = part_iter.next() orelse {
                    try stdout.writeAll("Error: Invalid assignment\n");
                    continue;
                };

                const value_str = part_iter.next() orelse {
                    try stdout.writeAll("Error: Missing value\n");
                    continue;
                };

                if (std.fmt.parseFloat(f64, value_str)) |num| {
                    try self.interpreter.setVariable(var_name, CellValue{ .number = num });
                    try stdout.print("Set {s} = {d}\n", .{ var_name, num });
                } else |_| {
                    try self.interpreter.setVariable(var_name, CellValue{ .string = value_str });
                    try stdout.print("Set {s} = \"{s}\"\n", .{ var_name, value_str });
                }
            } else if (std.mem.startsWith(u8, trimmed, "get ")) {
                const var_name = trimmed[4..];
                if (self.interpreter.environment.get(var_name)) |value| {
                    try stdout.print("{s} = {}\n", .{ var_name, value });
                } else {
                    try stdout.print("Undefined variable: {s}\n", .{var_name});
                }
            } else {
                try stdout.print("Unknown command: {s}\n", .{trimmed});
                try stdout.writeAll("Available commands:\n");
                try stdout.writeAll("  set <name> <value> - Set a variable\n");
                try stdout.writeAll("  get <name> - Get a variable value\n");
                try stdout.writeAll("  exit - Quit the REPL\n");
            }
        }
    }
};

/// Main entry point for the Cell language interpreter
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var repl = REPL.init(allocator);
    defer repl.deinit();

    try repl.run();
}

// Tests
test "cell value formatting" {
    const testing = std.testing;

    var buf: [64]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    const writer = fbs.writer();

    const number = CellValue{ .number = 42.5 };
    try number.format("", .{}, writer);
    try testing.expectEqualStrings("42.5", fbs.getWritten());

    fbs.reset();
    const string = CellValue{ .string = "hello" };
    try string.format("", .{}, writer);
    try testing.expectEqualStrings("\"hello\"", fbs.getWritten());
}

test "interpreter basic operations" {
    const testing = std.testing;

    var interpreter = Interpreter.init(testing.allocator);
    defer interpreter.deinit();

    // Test variable assignment and retrieval
    try interpreter.setVariable("x", CellValue{ .number = 10 });
    const value = interpreter.environment.get("x").?;
    try testing.expectEqual(@as(f64, 10), value.number);
}

test "binary operations" {
    const testing = std.testing;

    var interpreter = Interpreter.init(testing.allocator);
    defer interpreter.deinit();

    const left = CellValue{ .number = 10 };
    const right = CellValue{ .number = 5 };

    const add_result = try interpreter.evaluateBinaryOp(.add, left, right);
    try testing.expectEqual(@as(f64, 15), add_result.number);

    const sub_result = try interpreter.evaluateBinaryOp(.subtract, left, right);
    try testing.expectEqual(@as(f64, 5), sub_result.number);

    const mul_result = try interpreter.evaluateBinaryOp(.multiply, left, right);
    try testing.expectEqual(@as(f64, 50), mul_result.number);

    const div_result = try interpreter.evaluateBinaryOp(.divide, left, right);
    try testing.expectEqual(@as(f64, 2), div_result.number);
}
