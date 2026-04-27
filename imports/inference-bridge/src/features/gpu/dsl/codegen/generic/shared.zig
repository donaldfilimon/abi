const kernel = @import("../../kernel.zig");

pub fn writeSharedMemory(self: anytype, ir: *const kernel.KernelIR) !void {
    if (ir.shared_memory.len == 0) return;

    for (ir.shared_memory) |shared| {
        try self.writer.writeIndent();
        try writeSharedMemoryDecl(self, shared);
    }
    try self.writer.newline();
}

fn writeSharedMemoryDecl(self: anytype, shared: kernel.SharedMemory) !void {
    switch (self.config.language) {
        .glsl => {
            try self.writer.write("shared ");
            try self.writeType(shared.element_type);
            if (shared.size) |size| {
                try self.writer.writeFmt(" {s}[{d}];\n", .{ shared.name, size });
            } else {
                try self.writer.writeFmt(" {s}[];\n", .{shared.name});
            }
        },
        .wgsl => {
            try self.writer.write("var<workgroup> ");
            try self.writer.writeFmt("{s}: array<", .{shared.name});
            try self.writeType(shared.element_type);
            if (shared.size) |size| {
                try self.writer.writeFmt(", {d}>;\n", .{size});
            } else {
                try self.writer.write(">;\n");
            }
        },
        .msl => {
            try self.writer.write("threadgroup ");
            try self.writeType(shared.element_type);
            if (shared.size) |size| {
                try self.writer.writeFmt(" {s}[{d}];\n", .{ shared.name, size });
            } else {
                try self.writer.writeFmt(" {s}[];\n", .{shared.name});
            }
        },
        .cuda => {
            if (shared.size) |size| {
                try self.writer.write("__shared__ ");
                try self.writeType(shared.element_type);
                try self.writer.writeFmt(" {s}[{d}];\n", .{ shared.name, size });
            } else {
                try self.writer.write("extern __shared__ ");
                try self.writeType(shared.element_type);
                try self.writer.writeFmt(" {s}[];\n", .{shared.name});
            }
        },
        else => {},
    }
}
