const std = @import("std");
const Parser = @import("parser.zig").Parser;
const Interpreter = @import("interpreter.zig").Interpreter;

pub fn main() !void {
    const source = @"let x = 1 + 2; print x;";
    var parser = Parser.init(std.heap.page_allocator, source);
    const program = try parser.parseProgram();
    var interp = Interpreter.init(std.heap.page_allocator);
    interp.evalProgram(program);
}

test "parse var decl" {
    const src = "let a = 5;";
    var parser = Parser.init(std.testing.allocator, src);
    const program = try parser.parseProgram();
    try std.testing.expectEqual(@as(usize, 1), program.len);
}
