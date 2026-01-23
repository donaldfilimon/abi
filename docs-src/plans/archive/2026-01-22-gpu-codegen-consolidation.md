---
title: "2026-01-23-gpu-codegen-consolidation"
tags: []
---
# GPU Codegen Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete Phase 1 of the ABI refactoring plan - refactor GLSL codegen to use the generic module, reducing ~1,145 lines to ~100 lines.

**Architecture:** The generic codegen template (`generic.zig`) uses comptime configuration structs to generate backend-specific shader code. GLSL has 3 target variants (Vulkan, OpenGL, OpenGL ES) requiring special handling.

**Tech Stack:** Zig 0.16, comptime generics, GPU shader codegen (GLSL 430/450/310es)

---

## Status

**Completed:**
- `src/gpu/dsl/codegen/generic.zig` (1,175 lines) - Core generic template
- `src/gpu/dsl/codegen/configs/mod.zig` - Config type definitions
- `src/gpu/dsl/codegen/configs/glsl_config.zig` - GLSL config
- `src/gpu/dsl/codegen/configs/wgsl_config.zig` - WGSL config
- `src/gpu/dsl/codegen/configs/msl_config.zig` - MSL config
- `src/gpu/dsl/codegen/configs/cuda_config.zig` - CUDA config
- `src/gpu/dsl/codegen/wgsl.zig` refactored (1,091 → 365 lines)
- `src/gpu/dsl/codegen/cuda.zig` refactored (1,032 → 332 lines)
- `src/gpu/dsl/codegen/msl.zig` refactored (1,097 → 374 lines)

**Remaining:**
- `src/gpu/dsl/codegen/glsl.zig` (1,145 lines → ~100 lines target)

---

## Task 1: Add GLSL Target Support to Generic Module

**Files:**
- Modify: `src/gpu/dsl/codegen/generic.zig`
- Modify: `src/gpu/dsl/codegen/configs/glsl_config.zig`

**Step 1: Read current GLSL config**

Run: `cat src/gpu/dsl/codegen/configs/glsl_config.zig`

Verify: Config has `type_names`, `vector_naming`, `literal_format`, `atomics`, `barriers`, `builtins`

**Step 2: Add GlslTarget enum to configs/mod.zig**

Edit `src/gpu/dsl/codegen/configs/mod.zig` to add after `Language` enum:

```zig
/// GLSL target variant.
pub const GlslTarget = enum {
    vulkan,  // GLSL 450 with Vulkan extensions
    opengl,  // GLSL 430 compute shaders
    opengles, // GLSL ES 310+ compute
};
```

**Step 3: Run build to verify no errors**

Run: `zig build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add src/gpu/dsl/codegen/configs/mod.zig
git commit -m "feat(gpu): add GlslTarget enum to codegen configs"
```

---

## Task 2: Create GLSL Generator in Generic Module

**Files:**
- Modify: `src/gpu/dsl/codegen/generic.zig`

**Step 1: Add GLSL-specific header generation**

Add to `generic.zig` after the existing `writeHeader` function, a new function for GLSL:

```zig
fn writeGlslHeader(self: *Self, ir: *const kernel.KernelIR, target: configs.GlslTarget) !void {
    // Version directive
    switch (target) {
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
    if (target == .vulkan) {
        try self.writer.writeLine("#extension GL_ARB_separate_shader_objects : enable");
    }
    try self.writer.newline();

    // Precision for ES
    if (target == .opengles) {
        try self.writer.writeLine("precision highp float;");
        try self.writer.writeLine("precision highp int;");
        try self.writer.newline();
    }
}
```

**Step 2: Run build to verify**

Run: `zig build`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add src/gpu/dsl/codegen/generic.zig
git commit -m "feat(gpu): add GLSL header generation to generic codegen"
```

---

## Task 3: Add GLSL Layout Declarations

**Files:**
- Modify: `src/gpu/dsl/codegen/generic.zig`

**Step 1: Add GLSL layout declaration function**

Add to `generic.zig`:

```zig
fn writeGlslLayoutDeclarations(self: *Self, ir: *const kernel.KernelIR) !void {
    // Local size layout
    try self.writer.writeFmt("layout(local_size_x = {d}, local_size_y = {d}, local_size_z = {d}) in;\n", .{
        ir.workgroup_size[0],
        ir.workgroup_size[1],
        ir.workgroup_size[2],
    });
    try self.writer.newline();
}
```

**Step 2: Run build to verify**

Run: `zig build`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add src/gpu/dsl/codegen/generic.zig
git commit -m "feat(gpu): add GLSL layout declarations to generic codegen"
```

---

## Task 4: Add GLSL Buffer Declarations

**Files:**
- Modify: `src/gpu/dsl/codegen/generic.zig`

**Step 1: Add GLSL buffer declaration function**

Add to `generic.zig`:

```zig
fn writeGlslBufferDeclarations(self: *Self, ir: *const kernel.KernelIR, target: configs.GlslTarget) !void {
    for (ir.buffers, 0..) |buffer, i| {
        // Layout with binding
        try self.writer.writeFmt("layout(std430, binding = {d}) ", .{i});

        // Access qualifier
        if (buffer.is_readonly) {
            try self.writer.write("readonly ");
        } else if (buffer.is_writeonly) {
            try self.writer.write("writeonly ");
        }

        try self.writer.write("buffer ");

        // Buffer block name
        try self.writer.writeFmt("Buffer{d} {{\n", .{i});
        self.writer.indent();

        // Array member
        try self.writer.writeIndent();
        try self.writeType(buffer.element_type);
        try self.writer.writeFmt(" {s}[];\n", .{buffer.name});

        self.writer.dedent();
        try self.writer.writeLine("};");
        try self.writer.newline();
    }
}
```

**Step 2: Run build to verify**

Run: `zig build`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add src/gpu/dsl/codegen/generic.zig
git commit -m "feat(gpu): add GLSL buffer declarations to generic codegen"
```

---

## Task 5: Create Pre-instantiated GLSL Generators

**Files:**
- Modify: `src/gpu/dsl/codegen/generic.zig`

**Step 1: Add GLSL generator instantiation**

Add at the bottom of `generic.zig` after the other generator instantiations:

```zig
/// Pre-instantiated GLSL generator (Vulkan target).
pub const GlslVulkanGenerator = CodeGenerator(glsl_config.config);

/// Pre-instantiated GLSL generator (OpenGL target).
pub const GlslOpenGLGenerator = CodeGenerator(glsl_config.config);

/// Pre-instantiated GLSL generator (OpenGL ES target).
pub const GlslOpenGLESGenerator = CodeGenerator(glsl_config.config);
```

**Step 2: Run build to verify**

Run: `zig build`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add src/gpu/dsl/codegen/generic.zig
git commit -m "feat(gpu): add pre-instantiated GLSL generators"
```

---

## Task 6: Refactor glsl.zig to Use Generic Module

**Files:**
- Modify: `src/gpu/dsl/codegen/glsl.zig`

**Step 1: Create backup of current implementation**

Run: `cp src/gpu/dsl/codegen/glsl.zig src/gpu/dsl/codegen/glsl.zig.bak`

**Step 2: Refactor glsl.zig**

Replace the entire file with:

```zig
//! GLSL Code Generator
//!
//! Generates GLSL compute shader source code from kernel IR.
//! Targets Vulkan (GLSL 450) and OpenGL (GLSL 430+).

const std = @import("std");
const types = @import("../types.zig");
const kernel = @import("../kernel.zig");
const backend = @import("backend.zig");
const generic = @import("generic.zig");
const gpu_backend = @import("../../backend.zig");

/// Target GLSL variant.
pub const GlslTarget = enum {
    vulkan,  // GLSL 450 with Vulkan extensions
    opengl,  // GLSL 430 compute shaders
    opengles, // GLSL ES 310+ compute
};

/// GLSL code generator using generic backend.
pub const GlslGenerator = struct {
    inner: generic.GlslGenerator,
    target: GlslTarget,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, target: GlslTarget) Self {
        return .{
            .inner = generic.GlslGenerator.init(allocator),
            .target = target,
        };
    }

    pub fn deinit(self: *Self) void {
        self.inner.deinit();
    }

    /// Generate GLSL source code from kernel IR.
    pub fn generate(
        self: *Self,
        ir: *const kernel.KernelIR,
    ) backend.CodegenError!backend.GeneratedSource {
        var source = try self.inner.generate(ir);

        // Update backend based on target
        source.backend = switch (self.target) {
            .vulkan => .vulkan,
            .opengl => .opengl,
            .opengles => .opengles,
        };

        return source;
    }
};

// =============================================================================
// Vision Kernels - Specialized GLSL shaders for vision operations
// =============================================================================

pub const VisionKernels = struct {
    // Keep existing VisionKernels implementation unchanged
    // (This generates complete shader source directly, not using IR)

    pub fn generateConv2D(
        allocator: std.mem.Allocator,
        in_channels: u32,
        out_channels: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
        use_bias: bool,
        activation: ?types.ActivationType,
    ) ![]const u8 {
        _ = activation;
        var writer = std.ArrayList(u8).init(allocator);
        const w = writer.writer();

        try w.writeAll("#version 450\n\n");
        try w.writeAll("layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n\n");

        // Input/output buffers
        try w.writeAll("layout(std430, binding = 0) readonly buffer Input { float input_data[]; };\n");
        try w.writeAll("layout(std430, binding = 1) readonly buffer Weights { float weights[]; };\n");
        if (use_bias) {
            try w.writeAll("layout(std430, binding = 2) readonly buffer Bias { float bias[]; };\n");
            try w.writeAll("layout(std430, binding = 3) writeonly buffer Output { float output_data[]; };\n");
        } else {
            try w.writeAll("layout(std430, binding = 2) writeonly buffer Output { float output_data[]; };\n");
        }
        try w.writeAll("\n");

        // Uniforms
        try w.writeAll("layout(push_constant) uniform Params {\n");
        try w.writeAll("    uint batch_size;\n");
        try w.writeAll("    uint in_height;\n");
        try w.writeAll("    uint in_width;\n");
        try w.writeAll("    uint out_height;\n");
        try w.writeAll("    uint out_width;\n");
        try w.writeAll("} params;\n\n");

        // Main function
        try w.writeAll("void main() {\n");
        try w.writeAll("    uint gx = gl_GlobalInvocationID.x;\n");
        try w.writeAll("    uint gy = gl_GlobalInvocationID.y;\n");
        try w.writeAll("    uint gz = gl_GlobalInvocationID.z;\n\n");

        try w.writeAll("    if (gx >= params.out_width || gy >= params.out_height) return;\n\n");

        try w.print("    uint out_c = gz % {d}u;\n", .{out_channels});
        try w.print("    uint batch = gz / {d}u;\n\n", .{out_channels});

        try w.writeAll("    float sum = 0.0;\n");
        try w.print("    for (uint ic = 0u; ic < {d}u; ic++) {{\n", .{in_channels});
        try w.print("        for (uint kh = 0u; kh < {d}u; kh++) {{\n", .{kernel_h});
        try w.print("            for (uint kw = 0u; kw < {d}u; kw++) {{\n", .{kernel_w});

        try w.print("                int ih = int(gy * {d}u + kh) - int({d}u);\n", .{ stride_h, pad_h });
        try w.print("                int iw = int(gx * {d}u + kw) - int({d}u);\n", .{ stride_w, pad_w });

        try w.writeAll("                if (ih >= 0 && ih < int(params.in_height) && iw >= 0 && iw < int(params.in_width)) {\n");
        try w.print("                    uint in_idx = batch * {d}u * params.in_height * params.in_width + ", .{in_channels});
        try w.writeAll("ic * params.in_height * params.in_width + uint(ih) * params.in_width + uint(iw);\n");
        try w.print("                    uint w_idx = out_c * {d}u * {d}u * {d}u + ic * {d}u * {d}u + kh * {d}u + kw;\n", .{ in_channels, kernel_h, kernel_w, kernel_h, kernel_w, kernel_w });
        try w.writeAll("                    sum += input_data[in_idx] * weights[w_idx];\n");
        try w.writeAll("                }\n");
        try w.writeAll("            }\n");
        try w.writeAll("        }\n");
        try w.writeAll("    }\n\n");

        if (use_bias) {
            try w.writeAll("    sum += bias[out_c];\n");
        }

        try w.print("    uint out_idx = batch * {d}u * params.out_height * params.out_width + ", .{out_channels});
        try w.writeAll("out_c * params.out_height * params.out_width + gy * params.out_width + gx;\n");
        try w.writeAll("    output_data[out_idx] = sum;\n");
        try w.writeAll("}\n");

        return writer.toOwnedSlice();
    }

    pub fn generateMaxPool2D(
        allocator: std.mem.Allocator,
        channels: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
    ) ![]const u8 {
        var writer = std.ArrayList(u8).init(allocator);
        const w = writer.writer();

        try w.writeAll("#version 450\n\n");
        try w.writeAll("layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n\n");

        try w.writeAll("layout(std430, binding = 0) readonly buffer Input { float input_data[]; };\n");
        try w.writeAll("layout(std430, binding = 1) writeonly buffer Output { float output_data[]; };\n\n");

        try w.writeAll("layout(push_constant) uniform Params {\n");
        try w.writeAll("    uint batch_size;\n");
        try w.writeAll("    uint in_height;\n");
        try w.writeAll("    uint in_width;\n");
        try w.writeAll("    uint out_height;\n");
        try w.writeAll("    uint out_width;\n");
        try w.writeAll("} params;\n\n");

        try w.writeAll("void main() {\n");
        try w.writeAll("    uint gx = gl_GlobalInvocationID.x;\n");
        try w.writeAll("    uint gy = gl_GlobalInvocationID.y;\n");
        try w.writeAll("    uint gz = gl_GlobalInvocationID.z;\n\n");

        try w.writeAll("    if (gx >= params.out_width || gy >= params.out_height) return;\n\n");

        try w.print("    uint c = gz % {d}u;\n", .{channels});
        try w.print("    uint batch = gz / {d}u;\n\n", .{channels});

        try w.writeAll("    float max_val = -1e38;\n");
        try w.print("    for (uint kh = 0u; kh < {d}u; kh++) {{\n", .{kernel_h});
        try w.print("        for (uint kw = 0u; kw < {d}u; kw++) {{\n", .{kernel_w});

        try w.print("            int ih = int(gy * {d}u + kh) - int({d}u);\n", .{ stride_h, pad_h });
        try w.print("            int iw = int(gx * {d}u + kw) - int({d}u);\n", .{ stride_w, pad_w });

        try w.writeAll("            if (ih >= 0 && ih < int(params.in_height) && iw >= 0 && iw < int(params.in_width)) {\n");
        try w.print("                uint in_idx = batch * {d}u * params.in_height * params.in_width + ", .{channels});
        try w.writeAll("c * params.in_height * params.in_width + uint(ih) * params.in_width + uint(iw);\n");
        try w.writeAll("                max_val = max(max_val, input_data[in_idx]);\n");
        try w.writeAll("            }\n");
        try w.writeAll("        }\n");
        try w.writeAll("    }\n\n");

        try w.print("    uint out_idx = batch * {d}u * params.out_height * params.out_width + ", .{channels});
        try w.writeAll("c * params.out_height * params.out_width + gy * params.out_width + gx;\n");
        try w.writeAll("    output_data[out_idx] = max_val;\n");
        try w.writeAll("}\n");

        return writer.toOwnedSlice();
    }

    pub fn generateBatchNorm(
        allocator: std.mem.Allocator,
        channels: u32,
        epsilon: f32,
    ) ![]const u8 {
        var writer = std.ArrayList(u8).init(allocator);
        const w = writer.writer();

        try w.writeAll("#version 450\n\n");
        try w.writeAll("layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n");

        try w.writeAll("layout(std430, binding = 0) buffer Data { float data[]; };\n");
        try w.writeAll("layout(std430, binding = 1) readonly buffer Mean { float mean[]; };\n");
        try w.writeAll("layout(std430, binding = 2) readonly buffer Variance { float variance[]; };\n");
        try w.writeAll("layout(std430, binding = 3) readonly buffer Gamma { float gamma[]; };\n");
        try w.writeAll("layout(std430, binding = 4) readonly buffer Beta { float beta[]; };\n\n");

        try w.writeAll("layout(push_constant) uniform Params {\n");
        try w.writeAll("    uint batch_size;\n");
        try w.writeAll("    uint spatial_size;\n");
        try w.writeAll("} params;\n\n");

        try w.writeAll("void main() {\n");
        try w.writeAll("    uint idx = gl_GlobalInvocationID.x;\n");
        try w.writeAll("    uint total = params.batch_size * params.spatial_size;\n\n");

        try w.print("    if (idx >= total * {d}u) return;\n\n", .{channels});

        try w.print("    uint c = (idx / params.spatial_size) % {d}u;\n", .{channels});
        try w.print("    float inv_std = 1.0 / sqrt(variance[c] + {e});\n", .{epsilon});
        try w.writeAll("    data[idx] = (data[idx] - mean[c]) * inv_std * gamma[c] + beta[c];\n");
        try w.writeAll("}\n");

        return writer.toOwnedSlice();
    }

    pub fn generateSoftmax(
        allocator: std.mem.Allocator,
        axis_size: u32,
    ) ![]const u8 {
        var writer = std.ArrayList(u8).init(allocator);
        const w = writer.writer();

        try w.writeAll("#version 450\n\n");
        try w.writeAll("layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n");

        try w.writeAll("layout(std430, binding = 0) buffer Data { float data[]; };\n\n");

        try w.writeAll("layout(push_constant) uniform Params {\n");
        try w.writeAll("    uint outer_size;\n");
        try w.writeAll("} params;\n\n");

        try w.writeAll("shared float shared_max;\n");
        try w.writeAll("shared float shared_sum;\n\n");

        try w.writeAll("void main() {\n");
        try w.writeAll("    uint outer_idx = gl_WorkGroupID.x;\n");
        try w.writeAll("    uint tid = gl_LocalInvocationID.x;\n\n");

        try w.writeAll("    if (outer_idx >= params.outer_size) return;\n\n");

        try w.print("    uint base = outer_idx * {d}u;\n\n", .{axis_size});

        // Find max
        try w.writeAll("    float local_max = -1e38;\n");
        try w.print("    for (uint i = tid; i < {d}u; i += 256u) {{\n", .{axis_size});
        try w.writeAll("        local_max = max(local_max, data[base + i]);\n");
        try w.writeAll("    }\n\n");

        try w.writeAll("    if (tid == 0u) shared_max = -1e38;\n");
        try w.writeAll("    barrier();\n");
        try w.writeAll("    atomicMax(shared_max, local_max);\n");
        try w.writeAll("    barrier();\n\n");

        // Compute exp and sum
        try w.writeAll("    float local_sum = 0.0;\n");
        try w.print("    for (uint i = tid; i < {d}u; i += 256u) {{\n", .{axis_size});
        try w.writeAll("        float val = exp(data[base + i] - shared_max);\n");
        try w.writeAll("        data[base + i] = val;\n");
        try w.writeAll("        local_sum += val;\n");
        try w.writeAll("    }\n\n");

        try w.writeAll("    if (tid == 0u) shared_sum = 0.0;\n");
        try w.writeAll("    barrier();\n");
        try w.writeAll("    atomicAdd(shared_sum, local_sum);\n");
        try w.writeAll("    barrier();\n\n");

        // Normalize
        try w.print("    for (uint i = tid; i < {d}u; i += 256u) {{\n", .{axis_size});
        try w.writeAll("        data[base + i] /= shared_sum;\n");
        try w.writeAll("    }\n");
        try w.writeAll("}\n");

        return writer.toOwnedSlice();
    }

    pub fn generateDepthwiseConv2D(
        allocator: std.mem.Allocator,
        channels: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride_h: u32,
        stride_w: u32,
        pad_h: u32,
        pad_w: u32,
        depth_multiplier: u32,
    ) ![]const u8 {
        var writer = std.ArrayList(u8).init(allocator);
        const w = writer.writer();

        try w.writeAll("#version 450\n\n");
        try w.writeAll("layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n\n");

        try w.writeAll("layout(std430, binding = 0) readonly buffer Input { float input_data[]; };\n");
        try w.writeAll("layout(std430, binding = 1) readonly buffer Weights { float weights[]; };\n");
        try w.writeAll("layout(std430, binding = 2) writeonly buffer Output { float output_data[]; };\n\n");

        try w.writeAll("layout(push_constant) uniform Params {\n");
        try w.writeAll("    uint batch_size;\n");
        try w.writeAll("    uint in_height;\n");
        try w.writeAll("    uint in_width;\n");
        try w.writeAll("    uint out_height;\n");
        try w.writeAll("    uint out_width;\n");
        try w.writeAll("} params;\n\n");

        try w.writeAll("void main() {\n");
        try w.writeAll("    uint gx = gl_GlobalInvocationID.x;\n");
        try w.writeAll("    uint gy = gl_GlobalInvocationID.y;\n");
        try w.writeAll("    uint gz = gl_GlobalInvocationID.z;\n\n");

        try w.writeAll("    if (gx >= params.out_width || gy >= params.out_height) return;\n\n");

        const out_channels = channels * depth_multiplier;
        try w.print("    uint out_c = gz % {d}u;\n", .{out_channels});
        try w.print("    uint batch = gz / {d}u;\n", .{out_channels});
        try w.print("    uint in_c = out_c / {d}u;\n\n", .{depth_multiplier});

        try w.writeAll("    float sum = 0.0;\n");
        try w.print("    for (uint kh = 0u; kh < {d}u; kh++) {{\n", .{kernel_h});
        try w.print("        for (uint kw = 0u; kw < {d}u; kw++) {{\n", .{kernel_w});

        try w.print("            int ih = int(gy * {d}u + kh) - int({d}u);\n", .{ stride_h, pad_h });
        try w.print("            int iw = int(gx * {d}u + kw) - int({d}u);\n", .{ stride_w, pad_w });

        try w.writeAll("            if (ih >= 0 && ih < int(params.in_height) && iw >= 0 && iw < int(params.in_width)) {\n");
        try w.print("                uint in_idx = batch * {d}u * params.in_height * params.in_width + ", .{channels});
        try w.writeAll("in_c * params.in_height * params.in_width + uint(ih) * params.in_width + uint(iw);\n");
        try w.print("                uint w_idx = out_c * {d}u * {d}u + kh * {d}u + kw;\n", .{ kernel_h, kernel_w, kernel_w });
        try w.writeAll("                sum += input_data[in_idx] * weights[w_idx];\n");
        try w.writeAll("            }\n");
        try w.writeAll("        }\n");
        try w.writeAll("    }\n\n");

        try w.print("    uint out_idx = batch * {d}u * params.out_height * params.out_width + ", .{out_channels});
        try w.writeAll("out_c * params.out_height * params.out_width + gy * params.out_width + gx;\n");
        try w.writeAll("    output_data[out_idx] = sum;\n");
        try w.writeAll("}\n");

        return writer.toOwnedSlice();
    }
};
```

**Step 3: Run build to verify**

Run: `zig build`
Expected: Build succeeds

**Step 4: Run tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 5: Format code**

Run: `zig fmt src/gpu/dsl/codegen/glsl.zig`

**Step 6: Commit**

```bash
git add src/gpu/dsl/codegen/glsl.zig
git commit -m "refactor(gpu): consolidate GLSL codegen using generic module

- Reduce glsl.zig from 1,145 to ~200 lines
- Use generic.GlslGenerator for core codegen
- Preserve VisionKernels unchanged (generate complete shaders)
- Maintain backward compatibility with GlslTarget enum"
```

---

## Task 7: Verify Build and Run Full Test Suite

**Files:**
- All codegen files

**Step 1: Full build**

Run: `zig build`
Expected: Build succeeds with no errors

**Step 2: Run all tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 3: Verify line count reduction**

Run: `wc -l src/gpu/dsl/codegen/glsl.zig src/gpu/dsl/codegen/wgsl.zig src/gpu/dsl/codegen/cuda.zig src/gpu/dsl/codegen/msl.zig src/gpu/dsl/codegen/generic.zig`

Expected output (approximate):
```
   200 src/gpu/dsl/codegen/glsl.zig
   365 src/gpu/dsl/codegen/wgsl.zig
   332 src/gpu/dsl/codegen/cuda.zig
   374 src/gpu/dsl/codegen/msl.zig
  1175 src/gpu/dsl/codegen/generic.zig
  2446 total
```

**Step 4: Delete backup**

Run: `rm src/gpu/dsl/codegen/glsl.zig.bak`

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore(gpu): complete Phase 1 GPU codegen consolidation

Phase 1 complete:
- generic.zig: 1,175 lines (shared codegen logic)
- glsl.zig: ~200 lines (was 1,145)
- wgsl.zig: 365 lines (was 1,091)
- cuda.zig: 332 lines (was 1,032)
- msl.zig: 374 lines (was 1,097)

Total savings: ~2,500 lines"
```

---

## Summary

| File | Before | After | Savings |
|------|--------|-------|---------|
| glsl.zig | 1,145 | ~200 | ~945 lines |
| wgsl.zig | 1,091 | 365 | 726 lines |
| cuda.zig | 1,032 | 332 | 700 lines |
| msl.zig | 1,097 | 374 | 723 lines |
| generic.zig | 0 | 1,175 | (new) |
| configs/* | 0 | ~600 | (new) |
| **Total** | 4,365 | 3,046 | **~1,319 lines** |

Note: The actual savings are lower than originally estimated because the generic module and configs add ~1,775 lines of shared infrastructure. However, the code is now DRY and any future backends will only need ~50-100 lines of config.

