//! Shared WebGPU helpers used by tutorials and smoke tests to exercise ABI's
//! compute pipelines without duplicating setup boilerplate.

const std = @import("std");
const gpu = std.gpu;
const math = std.math;
const print = std.debug.print;

/// GPU context wrapper for managing WebGPU resources
const GPUContext = struct {
    instance: *gpu.Instance,
    adapter: *gpu.Adapter,
    device: *gpu.Device,
    queue: *gpu.Queue,

    /// Initialize GPU context with error handling
    pub fn init() !GPUContext {
        const instance = gpu.Instance.create(.{
            .backends = gpu.Instance.Backends.primary,
            .dx12_shader_compiler = .fxc,
        }) orelse return error.GpuInstanceCreationFailed;

        const adapter = instance.requestAdapter(.{
            .power_preference = .high_performance,
            .force_fallback_adapter = false,
        }) orelse return error.NoSuitableAdapter;

        const device = adapter.requestDevice(.{
            .label = "Main Device",
            .required_features = .{},
            .required_limits = .{},
        }) orelse return error.DeviceCreationFailed;

        const queue = device.queue;

        return .{
            .instance = instance,
            .adapter = adapter,
            .device = device,
            .queue = queue,
        };
    }

    /// Clean up GPU resources
    pub fn deinit(self: *GPUContext) void {
        self.device.deinit();
        self.adapter.deinit();
        self.instance.deinit();
    }

    /// Get device info for debugging
    pub fn printDeviceInfo(self: *GPUContext) void {
        print("GPU Device: {s}\n", .{self.adapter.getName()});
        print("Device Features: {any}\n", .{self.device.features});
    }
};

/// Compute pipeline wrapper with automatic resource management
const ComputePipeline = struct {
    pipeline: *gpu.ComputePipeline,
    bind_group_layout: *gpu.BindGroupLayout,

    /// Initialize compute pipeline with shader
    pub fn init(device: *gpu.Device, shader_source: []const u8) !ComputePipeline {
        const shader = device.createShaderModule(.{
            .label = "Compute Shader",
            .code = .{ .wgsl = shader_source },
        }) orelse return error.ShaderCreationFailed;
        defer shader.deinit();

        const pipeline = device.createComputePipeline(.{
            .label = "Compute Pipeline",
            .compute = .{
                .module = shader,
                .entry_point = "main",
            },
        }) orelse return error.PipelineCreationFailed;

        const bind_group_layout = pipeline.getBindGroupLayout(0);

        return .{
            .pipeline = pipeline,
            .bind_group_layout = bind_group_layout,
        };
    }

    /// Clean up pipeline resources
    pub fn deinit(self: *ComputePipeline) void {
        self.bind_group_layout.deinit();
        self.pipeline.deinit();
    }
};

/// Buffer manager for simplified GPU buffer operations
const BufferManager = struct {
    device: *gpu.Device,

    /// Create a GPU buffer with specified type and usage
    pub fn createBuffer(self: BufferManager, comptime T: type, size: u64, usage: gpu.Buffer.Usage) !*gpu.Buffer {
        return self.device.createBuffer(.{
            .label = @typeName(T) ++ " Buffer",
            .size = size * @sizeOf(T),
            .usage = usage,
            .mapped_at_creation = false,
        }) orelse return error.BufferCreationFailed;
    }

    /// Write data to GPU buffer
    pub fn writeBuffer(self: BufferManager, buffer: *gpu.Buffer, data: anytype) void {
        self.device.queue.writeBuffer(buffer, 0, std.mem.sliceAsBytes(data));
    }

    /// Read data from GPU buffer with staging buffer
    pub fn readBuffer(self: BufferManager, comptime T: type, buffer: *gpu.Buffer, size: u64, allocator: std.mem.Allocator) ![]T {
        const staging_buffer = try self.createBuffer(T, size, .{ .copy_dst = true, .map_read = true });
        defer staging_buffer.deinit();

        const encoder = self.device.createCommandEncoder(.{
            .label = "Copy Commands",
        }) orelse return error.CommandEncoderCreationFailed;
        defer encoder.deinit();

        encoder.copyBufferToBuffer(buffer, 0, staging_buffer, 0, size * @sizeOf(T));

        const command = encoder.finish(.{
            .label = "Copy Command",
        }) orelse return error.CommandCreationFailed;
        defer command.deinit();

        self.device.queue.submit(&[_]*gpu.CommandBuffer{command});

        // Wait for GPU operations to complete
        self.device.queue.onSubmittedWorkDone(null, null);

        const mapped_range = staging_buffer.getMappedRange(T, 0, size) orelse return error.BufferMappingFailed;
        const result = try allocator.dupe(T, mapped_range);
        staging_buffer.unmap();

        return result;
    }

    /// Create a buffer with initial data
    pub fn createBufferWithData(self: BufferManager, comptime T: type, data: []const T, usage: gpu.Buffer.Usage) !*gpu.Buffer {
        const buffer = try self.createBuffer(T, data.len, usage);
        self.writeBuffer(buffer, data);
        return buffer;
    }
};

/// Vector addition example using GPU compute
fn vectorAdd(ctx: *GPUContext, allocator: std.mem.Allocator) !void {
    const size = 1024;
    const workgroup_size = 64;
    const workgroup_count = (size + workgroup_size - 1) / workgroup_size;

    const compute_shader =
        \\@group(0) @binding(0) var<storage, read_write> a: array<f32>;
        \\@group(0) @binding(1) var<storage, read_write> b: array<f32>;
        \\@group(0) @binding(2) var<storage, read_write> result: array<f32>;
        \\
        \\@compute @workgroup_size(64)
        \\fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        \\    let index = global_id.x;
        \\    if (index >= arrayLength(&a)) {
        \\        return;
        \\    }
        \\    result[index] = a[index] + b[index];
        \\}
    ;

    var pipeline = try ComputePipeline.init(ctx.device, compute_shader);
    defer pipeline.deinit();

    const buffer_manager = BufferManager{ .device = ctx.device };

    // Create buffers
    const buffer_a = try buffer_manager.createBuffer(f32, size, .{ .storage = true, .copy_dst = true });
    defer buffer_a.deinit();

    const buffer_b = try buffer_manager.createBuffer(f32, size, .{ .storage = true, .copy_dst = true });
    defer buffer_b.deinit();

    const buffer_result = try buffer_manager.createBuffer(f32, size, .{ .storage = true, .copy_src = true });
    defer buffer_result.deinit();

    // Initialize data
    const data_a = try allocator.alloc(f32, size);
    defer allocator.free(data_a);

    const data_b = try allocator.alloc(f32, size);
    defer allocator.free(data_b);

    for (data_a, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    for (data_b, 0..) |*val, i| {
        val.* = @floatFromInt(i * 2);
    }

    // Upload data
    buffer_manager.writeBuffer(buffer_a, data_a);
    buffer_manager.writeBuffer(buffer_b, data_b);

    // Create bind group
    const bind_group = ctx.device.createBindGroup(.{
        .label = "Compute Bind Group",
        .layout = pipeline.bind_group_layout,
        .entries = &[_]gpu.BindGroup.Entry{
            .{
                .binding = 0,
                .resource = .{ .buffer = .{
                    .buffer = buffer_a,
                    .offset = 0,
                    .size = size * @sizeOf(f32),
                } },
            },
            .{
                .binding = 1,
                .resource = .{ .buffer = .{
                    .buffer = buffer_b,
                    .offset = 0,
                    .size = size * @sizeOf(f32),
                } },
            },
            .{
                .binding = 2,
                .resource = .{ .buffer = .{
                    .buffer = buffer_result,
                    .offset = 0,
                    .size = size * @sizeOf(f32),
                } },
            },
        },
    }) orelse return error.BindGroupCreationFailed;
    defer bind_group.deinit();

    // Dispatch compute
    const encoder = ctx.device.createCommandEncoder(.{
        .label = "Compute Commands",
    }) orelse return error.CommandEncoderCreationFailed;
    defer encoder.deinit();

    const compute_pass = encoder.beginComputePass(.{
        .label = "Vector Add Pass",
    });
    defer compute_pass.deinit();

    compute_pass.setPipeline(pipeline.pipeline);
    compute_pass.setBindGroup(0, bind_group);
    compute_pass.dispatchWorkgroups(workgroup_count, 1, 1);
    compute_pass.end();

    const command = encoder.finish(.{
        .label = "Compute Command",
    }) orelse return error.CommandCreationFailed;
    defer command.deinit();

    ctx.queue.submit(&[_]*gpu.CommandBuffer{command});

    // Wait for completion
    ctx.queue.onSubmittedWorkDone(null, null);

    // Read results
    const result = try buffer_manager.readBuffer(f32, buffer_result, size, allocator);
    defer allocator.free(result);

    print("Vector Addition Results:\n");
    print("A[0]={d:.2}, B[0]={d:.2}, Result[0]={d:.2}\n", .{ data_a[0], data_b[0], result[0] });
    print("A[1]={d:.2}, B[1]={d:.2}, Result[1]={d:.2}\n", .{ data_a[1], data_b[1], result[1] });
    print("A[2]={d:.2}, B[2]={d:.2}, Result[2]={d:.2}\n", .{ data_a[2], data_b[2], result[2] });

    // Verify results
    for (0..@min(10, size)) |i| {
        const expected = data_a[i] + data_b[i];
        if (math.approxEqAbs(f32, result[i], expected, 0.001)) {
            print("✓ Result[{d}] = {d:.2} (expected {d:.2})\n", .{ i, result[i], expected });
        } else {
            print("✗ Result[{d}] = {d:.2} (expected {d:.2})\n", .{ i, result[i], expected });
        }
    }
}

/// Matrix multiplication example using GPU compute
fn matrixMultiply(ctx: *GPUContext, allocator: std.mem.Allocator) !void {
    const size = 256;
    const workgroup_size = 16;
    const workgroup_count_x = (size + workgroup_size - 1) / workgroup_size;
    const workgroup_count_y = workgroup_count_x;

    const compute_shader =
        \\@group(0) @binding(0) var<storage, read> a: array<f32>;
        \\@group(0) @binding(1) var<storage, read> b: array<f32>;
        \\@group(0) @binding(2) var<storage, read_write> result: array<f32>;
        \\
        \\@compute @workgroup_size(16, 16)
        \\fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        \\    let row = global_id.x;
        \\    let col = global_id.y;
        \\    let size = 256u;
        \\    
        \\    if (row >= size || col >= size) {
        \\        return;
        \\    }
        \\    
        \\    var sum: f32 = 0.0;
        \\    for (var k = 0u; k < size; k++) {
        \\        sum += a[row * size + k] * b[k * size + col];
        \\    }
        \\    
        \\    result[row * size + col] = sum;
        \\}
    ;

    const pipeline = try ComputePipeline.init(ctx.device, compute_shader);
    defer pipeline.deinit();

    const buffer_manager = BufferManager{ .device = ctx.device };

    // Create buffers
    const buffer_a = try buffer_manager.createBuffer(f32, size * size, .{ .storage = true, .copy_dst = true });
    defer buffer_a.deinit();

    const buffer_b = try buffer_manager.createBuffer(f32, size * size, .{ .storage = true, .copy_dst = true });
    defer buffer_b.deinit();

    const buffer_result = try buffer_manager.createBuffer(f32, size * size, .{ .storage = true, .copy_src = true });
    defer buffer_result.deinit();

    // Initialize matrices
    const matrix_a = try allocator.alloc(f32, size * size);
    defer allocator.free(matrix_a);

    const matrix_b = try allocator.alloc(f32, size * size);
    defer allocator.free(matrix_b);

    // Fill with test data
    for (matrix_a, 0..) |*val, i| {
        val.* = @floatFromInt(i % 10);
    }

    for (matrix_b, 0..) |*val, i| {
        val.* = @floatFromInt((i * 2) % 10);
    }

    // Upload data
    buffer_manager.writeBuffer(buffer_a, matrix_a);
    buffer_manager.writeBuffer(buffer_b, matrix_b);

    // Create bind group
    const bind_group = ctx.device.createBindGroup(.{
        .label = "Matrix Multiply Bind Group",
        .layout = pipeline.bind_group_layout,
        .entries = &[_]gpu.BindGroup.Entry{
            .{
                .binding = 0,
                .resource = .{ .buffer = .{
                    .buffer = buffer_a,
                    .offset = 0,
                    .size = size * size * @sizeOf(f32),
                } },
            },
            .{
                .binding = 1,
                .resource = .{ .buffer = .{
                    .buffer = buffer_b,
                    .offset = 0,
                    .size = size * size * @sizeOf(f32),
                } },
            },
            .{
                .binding = 2,
                .resource = .{ .buffer = .{
                    .buffer = buffer_result,
                    .offset = 0,
                    .size = size * size * @sizeOf(f32),
                } },
            },
        },
    }) orelse return error.BindGroupCreationFailed;
    defer bind_group.deinit();

    // Dispatch compute
    const encoder = ctx.device.createCommandEncoder(.{
        .label = "Matrix Multiply Commands",
    }) orelse return error.CommandEncoderCreationFailed;
    defer encoder.deinit();

    const compute_pass = encoder.beginComputePass(.{
        .label = "Matrix Multiply Pass",
    });
    defer compute_pass.deinit();

    compute_pass.setPipeline(pipeline.pipeline);
    compute_pass.setBindGroup(0, bind_group);
    compute_pass.dispatchWorkgroups(workgroup_count_x, workgroup_count_y, 1);
    compute_pass.end();

    const command = encoder.finish(.{
        .label = "Matrix Multiply Command",
    }) orelse return error.CommandCreationFailed;
    defer command.deinit();

    ctx.queue.submit(&[_]*gpu.CommandBuffer{command});

    // Wait for completion
    ctx.queue.onSubmittedWorkDone(null, null);

    // Read results
    const result = try buffer_manager.readBuffer(f32, buffer_result, size * size, allocator);
    defer allocator.free(result);

    print("Matrix Multiplication Results (256x256):\n");
    print("Result[0,0]={d:.2}, Result[0,1]={d:.2}\n", .{ result[0], result[1] });
    print("Result[1,0]={d:.2}, Result[1,1]={d:.2}\n", .{ result[size], result[size + 1] });

    // Verify a few results
    const expected_00 = matrix_a[0] * matrix_b[0] + matrix_a[1] * matrix_b[size];
    print("Expected[0,0]={d:.2}, Got={d:.2}\n", .{ expected_00, result[0] });
}

/// Image processing example using GPU compute
fn imageProcessing(ctx: *GPUContext, allocator: std.mem.Allocator) !void {
    const width = 512;
    const height = 512;
    const workgroup_size = 16;
    const workgroup_count_x = (width + workgroup_size - 1) / workgroup_size;
    const workgroup_count_y = (height + workgroup_size - 1) / workgroup_size;

    const compute_shader =
        \\@group(0) @binding(0) var input_texture: texture_2d<f32>;
        \\@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
        \\
        \\@compute @workgroup_size(16, 16)
        \\fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        \\    let coords = vec2<i32>(i32(global_id.x), i32(global_id.y));
        \\    let dimensions = textureDimensions(input_texture);
        \\    
        \\    if (coords.x >= i32(dimensions.x) || coords.y >= i32(dimensions.y)) {
        \\        return;
        \\    }
        \\    
        \\    // Simple blur filter
        \\    var sum = vec4<f32>(0.0);
        \\    let radius = 3;
        \\    var count = 0.0;
        \\    
        \\    for (var dy = -radius; dy <= radius; dy++) {
        \\        for (var dx = -radius; dx <= radius; dx++) {
        \\            let sample_coords = coords + vec2<i32>(dx, dy);
        \\            if (sample_coords.x >= 0 && sample_coords.x < i32(dimensions.x) &&
        \\                sample_coords.y >= 0 && sample_coords.y < i32(dimensions.y)) {
        \\                sum += textureLoad(input_texture, sample_coords, 0);
        \\                count += 1.0;
        \\            }
        \\        }
        \\    }
        \\    
        \\    let blurred = sum / count;
        \\    textureStore(output_texture, coords, blurred);
        \\}
    ;

    var pipeline = try ComputePipeline.init(ctx.device, compute_shader);
    defer pipeline.deinit();

    // Create input texture
    const input_texture = ctx.device.createTexture(.{
        .label = "Input Texture",
        .size = .{
            .width = width,
            .height = height,
            .depth_or_array_layers = 1,
        },
        .mip_level_count = 1,
        .sample_count = 1,
        .dimension = .@"2d",
        .format = .rgba8_unorm,
        .usage = .{ .texture_binding = true, .copy_dst = true },
        .view_formats = &[_]gpu.Texture.Format{.rgba8_unorm},
    }) orelse return error.TextureCreationFailed;
    defer input_texture.deinit();

    // Create output texture
    const output_texture = ctx.device.createTexture(.{
        .label = "Output Texture",
        .size = .{
            .width = width,
            .height = height,
            .depth_or_array_layers = 1,
        },
        .mip_level_count = 1,
        .sample_count = 1,
        .dimension = .@"2d",
        .format = .rgba8_unorm,
        .usage = .{ .storage_binding = true, .copy_src = true },
        .view_formats = &[_]gpu.Texture.Format{.rgba8_unorm},
    }) orelse return error.TextureCreationFailed;
    defer output_texture.deinit();

    // Create texture views
    const input_view = input_texture.createView(.{
        .label = "Input View",
        .format = .rgba8_unorm,
        .dimension = .@"2d",
        .base_mip_level = 0,
        .mip_level_count = 1,
        .base_array_layer = 0,
        .array_layer_count = 1,
        .aspect = .all,
    }) orelse return error.TextureViewCreationFailed;
    defer input_view.deinit();

    const output_view = output_texture.createView(.{
        .label = "Output View",
        .format = .rgba8_unorm,
        .dimension = .@"2d",
        .base_mip_level = 0,
        .mip_level_count = 1,
        .base_array_layer = 0,
        .array_layer_count = 1,
        .aspect = .all,
    }) orelse return error.TextureViewCreationFailed;
    defer output_view.deinit();

    // Generate test image data
    var image_data = try allocator.alloc(u8, width * height * 4);
    defer allocator.free(image_data);

    for (0..height) |y| {
        for (0..width) |x| {
            const index = (y * width + x) * 4;
            image_data[index] = @intCast((x * 255) / width); // R
            image_data[index + 1] = @intCast((y * 255) / height); // G
            image_data[index + 2] = 128; // B
            image_data[index + 3] = 255; // A
        }
    }

    // Upload image data
    ctx.queue.writeTexture(.{ .texture = input_texture, .mip_level = 0, .origin = .{ .x = 0, .y = 0, .z = 0 } }, image_data, .{ .bytes_per_row = width * 4, .rows_per_image = height }, .{ .width = width, .height = height, .depth_or_array_layers = 1 });

    // Create bind group
    const bind_group = ctx.device.createBindGroup(.{
        .label = "Image Processing Bind Group",
        .layout = pipeline.bind_group_layout,
        .entries = &[_]gpu.BindGroup.Entry{
            .{
                .binding = 0,
                .resource = .{ .texture_view = input_view },
            },
            .{
                .binding = 1,
                .resource = .{ .texture_view = output_view },
            },
        },
    }) orelse return error.BindGroupCreationFailed;
    defer bind_group.deinit();

    // Dispatch compute
    const encoder = ctx.device.createCommandEncoder(.{
        .label = "Image Processing Commands",
    }) orelse return error.CommandEncoderCreationFailed;
    defer encoder.deinit();

    const compute_pass = encoder.beginComputePass(.{
        .label = "Image Processing Pass",
    });
    defer compute_pass.deinit();

    compute_pass.setPipeline(pipeline.pipeline);
    compute_pass.setBindGroup(0, bind_group);
    compute_pass.dispatchWorkgroups(workgroup_count_x, workgroup_count_y, 1);
    compute_pass.end();

    const command = encoder.finish(.{
        .label = "Image Processing Command",
    }) orelse return error.CommandCreationFailed;
    defer command.deinit();

    ctx.queue.submit(&[_]*gpu.CommandBuffer{command});

    // Wait for completion
    ctx.queue.onSubmittedWorkDone(null, null);

    print("Image processing completed (512x512 blur filter)\n");
}

/// Main function demonstrating GPU compute capabilities
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("Initializing GPU context...\n");
    var ctx = try GPUContext.init();
    defer ctx.deinit();

    print("GPU initialized successfully!\n");
    ctx.printDeviceInfo();

    // Run examples
    print("\n=== Vector Addition Example ===\n");
    try vectorAdd(&ctx, allocator);

    print("\n=== Matrix Multiplication Example ===\n");
    try matrixMultiply(&ctx, allocator);

    print("\n=== Image Processing Example ===\n");
    try imageProcessing(&ctx, allocator);

    print("\nAll GPU operations completed successfully!\n");
}

test "gpu context initialization" {
    const testing = std.testing;

    var ctx = GPUContext.init() catch {
        // Skip test if GPU is not available
        print("Skipping GPU test - no suitable GPU found\n", .{});
        return;
    };
    defer ctx.deinit();

    try testing.expect(ctx.device != null);
    try testing.expect(ctx.queue != null);
    print("GPU test passed - device initialized successfully\n", .{});
}

test "gpu vector addition" {
    const testing = std.testing;

    var ctx = GPUContext.init() catch {
        print("Skipping GPU vector addition test - no suitable GPU found\n", .{});
        return;
    };
    defer ctx.deinit();

    try vectorAdd(&ctx, testing.allocator);
    print("GPU vector addition test passed\n", .{});
}
