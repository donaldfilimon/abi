const kernel = @import("../../kernel.zig");

pub fn writeKernelSignature(self: anytype, ir: *const kernel.KernelIR) !void {
    switch (self.config.language) {
        .glsl => try writeGlslSignature(self, ir),
        .wgsl => try writeWgslSignature(self, ir),
        .msl => try writeMslSignature(self, ir),
        .cuda => try writeCudaSignature(self, ir),
        else => {},
    }
}

fn writeGlslSignature(self: anytype, ir: *const kernel.KernelIR) !void {
    // Layout declarations
    try self.writer.writeFmt("layout(local_size_x = {d}, local_size_y = {d}, local_size_z = {d}) in;\n", .{
        ir.workgroup_size[0],
        ir.workgroup_size[1],
        ir.workgroup_size[2],
    });
    try self.writer.newline();

    // Buffer declarations
    for (ir.buffers, 0..) |buf, i| {
        try self.writer.writeFmt("layout(set = 0, binding = {d}) buffer Buffer{d} {{\n", .{ buf.binding, i });
        self.writer.indent();
        try self.writer.writeIndent();
        try self.writeType(buf.element_type);
        try self.writer.writeFmt(" {s}[];\n", .{buf.name});
        self.writer.dedent();
        try self.writer.writeLine("};");
    }

    // Uniform declarations
    if (ir.uniforms.len > 0) {
        try self.writer.writeLine("layout(push_constant) uniform PushConstants {");
        self.writer.indent();
        for (ir.uniforms) |uni| {
            try self.writer.writeIndent();
            try self.writeType(uni.ty);
            try self.writer.writeFmt(" {s};\n", .{uni.name});
        }
        self.writer.dedent();
        try self.writer.writeLine("} uniforms;");
    }

    try self.writer.newline();
    try self.writer.writeLine("void main() {");
    self.writer.indent();
}

fn writeWgslSignature(self: anytype, ir: *const kernel.KernelIR) !void {
    // Struct for uniforms
    if (ir.uniforms.len > 0) {
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
        try self.writer.writeLine("@group(0) @binding(0) var<uniform> uniforms: Uniforms;");
        try self.writer.newline();
    }

    // Buffer bindings
    for (ir.buffers, 0..) |buf, i| {
        const access = if (buf.access == .read_only) "read" else "read_write";
        try self.writer.writeFmt("@group(0) @binding({d}) var<storage, {s}> {s}: array<", .{ buf.binding, access, buf.name });
        try self.writeType(buf.element_type);
        try self.writer.write(">;\n");
        _ = i;
    }

    try self.writer.newline();
    try self.writer.writeFmt("@compute @workgroup_size({d}, {d}, {d})\n", .{
        ir.workgroup_size[0],
        ir.workgroup_size[1],
        ir.workgroup_size[2],
    });
    try self.writer.writeFmt("fn {s}(\n", .{ir.entry_point});
    self.writer.indent();
    try self.writer.writeLine("@builtin(global_invocation_id) globalInvocationId: vec3<u32>,");
    try self.writer.writeLine("@builtin(local_invocation_id) localInvocationId: vec3<u32>,");
    try self.writer.writeLine("@builtin(workgroup_id) workgroupId: vec3<u32>,");
    try self.writer.writeLine("@builtin(local_invocation_index) localInvocationIndex: u32,");
    try self.writer.writeLine("@builtin(num_workgroups) numWorkgroups: vec3<u32>");
    self.writer.dedent();
    try self.writer.writeLine(") {");
    self.writer.indent();
}

fn writeMslSignature(self: anytype, ir: *const kernel.KernelIR) !void {
    // Uniform struct
    if (ir.uniforms.len > 0) {
        try self.writer.writeLine("struct Uniforms {");
        self.writer.indent();
        for (ir.uniforms) |uni| {
            try self.writer.writeIndent();
            try self.writeType(uni.ty);
            try self.writer.writeFmt(" {s};\n", .{uni.name});
        }
        self.writer.dedent();
        try self.writer.writeLine("};");
        try self.writer.newline();
    }

    try self.writer.writeLine("kernel void");
    try self.writer.writeFmt("{s}(\n", .{ir.entry_point});
    self.writer.indent();

    // Buffer parameters
    for (ir.buffers) |buf| {
        try self.writer.writeIndent();
        if (buf.access == .read_only) {
            try self.writer.write("const ");
        }
        try self.writer.write("device ");
        try self.writeType(buf.element_type);
        try self.writer.writeFmt("* {s} [[buffer({d})]],\n", .{ buf.name, buf.binding });
    }

    // Uniform buffer
    if (ir.uniforms.len > 0) {
        try self.writer.writeLine("constant Uniforms& uniforms [[buffer(0)]],");
    }

    // Built-in parameters
    try self.writer.writeLine("uint3 globalInvocationId [[thread_position_in_grid]],");
    try self.writer.writeLine("uint3 localInvocationId [[thread_position_in_threadgroup]],");
    try self.writer.writeLine("uint3 workgroupId [[threadgroup_position_in_grid]],");
    try self.writer.writeLine("uint localInvocationIndex [[thread_index_in_threadgroup]],");
    try self.writer.writeLine("uint3 numWorkgroups [[threadgroups_per_grid]]");

    self.writer.dedent();
    try self.writer.writeLine(") {");
    self.writer.indent();
}

fn writeCudaSignature(self: anytype, ir: *const kernel.KernelIR) !void {
    try self.writer.writeFmt("__global__ void {s}(\n", .{ir.entry_point});
    self.writer.indent();

    // Buffer parameters
    for (ir.buffers, 0..) |buf, i| {
        try self.writer.writeIndent();
        if (buf.access == .read_only) {
            try self.writer.write("const ");
        }
        try self.writeType(buf.element_type);
        try self.writer.writeFmt("* __restrict__ {s}", .{buf.name});

        if (i < ir.buffers.len - 1 or ir.uniforms.len > 0) {
            try self.writer.write(",\n");
        } else {
            try self.writer.newline();
        }
    }

    // Uniform parameters
    for (ir.uniforms, 0..) |uni, i| {
        try self.writer.writeIndent();
        try self.writeType(uni.ty);
        try self.writer.writeFmt(" {s}", .{uni.name});

        if (i < ir.uniforms.len - 1) {
            try self.writer.write(",\n");
        } else {
            try self.writer.newline();
        }
    }

    self.writer.dedent();
    try self.writer.writeLine(") {");
    self.writer.indent();
}

pub fn writeBuiltinVars(self: anytype, ir: *const kernel.KernelIR) !void {
    // GLSL/WGSL/MSL have built-ins via parameters, CUDA needs computation
    if (self.config.language == .cuda) {
        try self.writer.writeLine("// Built-in variables");
        try self.writer.writeLine("const uint3 globalInvocationId = make_uint3(");
        self.writer.indent();
        try self.writer.writeLine("blockIdx.x * blockDim.x + threadIdx.x,");
        try self.writer.writeLine("blockIdx.y * blockDim.y + threadIdx.y,");
        try self.writer.writeLine("blockIdx.z * blockDim.z + threadIdx.z);");
        self.writer.dedent();
        try self.writer.writeLine("const uint3 localInvocationId = make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);");
        try self.writer.writeLine("const uint3 workgroupId = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z);");
        try self.writer.writeFmt("const uint3 workgroupSize = make_uint3({d}, {d}, {d});\n", .{
            ir.workgroup_size[0],
            ir.workgroup_size[1],
            ir.workgroup_size[2],
        });
        try self.writer.writeLine("const unsigned int localInvocationIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;");
        try self.writer.newline();
    }
}
