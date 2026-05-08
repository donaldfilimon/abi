const kernel = @import("../../kernel.zig");
const configs = @import("../configs/mod.zig");

pub fn writeHeader(self: anytype, ir: *const kernel.KernelIR) !void {
    try self.writer.writeLine("// Auto-generated compute shader");
    try self.writer.writeFmt("// Kernel: {s}\n", .{ir.name});
    try self.writer.newline();

    // Language-specific headers
    switch (self.config.language) {
        .glsl => try writeGlslHeader(self),
        .wgsl => {}, // WGSL needs no header
        .msl => try writeMslHeader(self),
        .cuda => try writeCudaHeader(self),
        else => {},
    }
}

fn writeGlslHeader(self: anytype) !void {
    const target = self.config.glsl_target;
    try self.writer.writeLine(configs.glsl.getVersionDirective(target));
    for (configs.glsl.getExtensions(target)) |ext| {
        try self.writer.writeLine(ext);
    }
    try self.writer.newline();
}

fn writeMslHeader(self: anytype) !void {
    try self.writer.writeLine("#include <metal_stdlib>");
    try self.writer.writeLine("using namespace metal;");
    try self.writer.newline();
}

fn writeCudaHeader(self: anytype) !void {
    try self.writer.writeLine("#include <cuda_runtime.h>");
    try self.writer.writeLine("#include <stdint.h>");
    try self.writer.newline();
}
