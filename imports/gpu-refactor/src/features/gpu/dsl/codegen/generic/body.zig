const kernel = @import("../../kernel.zig");

pub fn writeBody(self: anytype, ir: *const kernel.KernelIR) !void {
    for (ir.body) |s| {
        try self.writeStmt(s);
    }
}

pub fn writeKernelClose(self: anytype) !void {
    self.writer.dedent();
    try self.writer.writeLine("}");
}
