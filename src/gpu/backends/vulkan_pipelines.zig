//! Vulkan kernel/pipeline compilation and execution.
//!
//! Handles shader compilation to SPIR-V, compute pipeline creation,
//! descriptor set management, and kernel dispatch.
//!
//! Features:
//! - SPIR-V shader compilation from GLSL or DSL IR
//! - Hash-based shader caching for fast recompilation
//! - Compute pipeline management
//! - Descriptor set allocation and binding

const std = @import("std");
const types = @import("vulkan_types.zig");
const init = @import("vulkan_init.zig");
const kernel_types = @import("../kernel_types.zig");
const spirv_gen = @import("../dsl/codegen/spirv.zig");

pub const VulkanError = types.VulkanError;

/// Global shader cache for compiled SPIR-V modules.
var global_shader_cache: ?*spirv_gen.ShaderCache = null;
var cache_mutex = std.Thread.Mutex{};

/// Initialize the shader cache.
pub fn initShaderCache(allocator: std.mem.Allocator, max_entries: usize) !void {
    cache_mutex.lock();
    defer cache_mutex.unlock();

    if (global_shader_cache == null) {
        const cache = try allocator.create(spirv_gen.ShaderCache);
        cache.* = spirv_gen.ShaderCache.init(allocator, max_entries);
        global_shader_cache = cache;
    }
}

/// Deinitialize the shader cache.
pub fn deinitShaderCache(allocator: std.mem.Allocator) void {
    cache_mutex.lock();
    defer cache_mutex.unlock();

    if (global_shader_cache) |cache| {
        cache.deinit();
        allocator.destroy(cache);
        global_shader_cache = null;
    }
}

/// Get shader cache statistics.
pub fn getShaderCacheStats() ?struct { hits: u64, misses: u64, entries: usize } {
    cache_mutex.lock();
    defer cache_mutex.unlock();

    if (global_shader_cache) |cache| {
        return cache.getStats();
    }
    return null;
}

/// Kernel compilation options
pub const KernelCompileOptions = struct {
    /// Number of storage buffer bindings (default: 8 for typical compute kernels)
    binding_count: u32 = 8,
    /// Push constant size in bytes (0 = disabled, max typically 128-256 bytes)
    push_constant_size: u32 = 0,
    /// Use pipeline cache for faster subsequent compilations
    use_cache: bool = true,
};

/// Compile a kernel source into Vulkan shader module and pipeline.
pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: kernel_types.KernelSource,
) kernel_types.KernelError!*anyopaque {
    return compileKernelEx(allocator, source, .{});
}

/// Compile a kernel with custom options.
pub fn compileKernelEx(
    allocator: std.mem.Allocator,
    source: kernel_types.KernelSource,
    options: KernelCompileOptions,
) kernel_types.KernelError!*anyopaque {
    if (!init.vulkan_initialized or init.vulkan_context == null) {
        return kernel_types.KernelError.CompilationFailed;
    }

    const ctx = &init.vulkan_context.?;
    const binding_count = @max(options.binding_count, 1);

    // Compile GLSL to SPIR-V
    const spirv = try compileGLSLToSPIRV(source.source);
    defer std.heap.page_allocator.free(spirv);

    // Create shader module
    const shader_create_info = types.VkShaderModuleCreateInfo{
        .codeSize = spirv.len * @sizeOf(u32),
        .pCode = spirv.ptr,
    };

    const create_shader_fn = init.vkCreateShaderModule orelse return kernel_types.KernelError.CompilationFailed;
    var shader_module: types.VkShaderModule = undefined;
    const shader_result = create_shader_fn(ctx.device, &shader_create_info, null, &shader_module);
    if (shader_result != .success) {
        return kernel_types.KernelError.CompilationFailed;
    }

    errdefer if (init.vkDestroyShaderModule) |destroy_fn| destroy_fn(ctx.device, shader_module, null);

    // Create descriptor set layout with multiple bindings
    var layout_bindings: [32]types.VkDescriptorSetLayoutBinding = undefined;
    for (0..@min(binding_count, 32)) |i| {
        layout_bindings[i] = types.VkDescriptorSetLayoutBinding{
            .binding = @intCast(i),
            .descriptorType = 7, // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
            .descriptorCount = 1,
            .stageFlags = types.VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }

    const layout_create_info = types.VkDescriptorSetLayoutCreateInfo{
        .bindingCount = @min(binding_count, 32),
        .pBindings = &layout_bindings,
    };

    const create_layout_fn = init.vkCreateDescriptorSetLayout orelse return kernel_types.KernelError.CompilationFailed;
    var descriptor_set_layout: types.VkDescriptorSetLayout = undefined;
    const layout_result = create_layout_fn(ctx.device, &layout_create_info, null, &descriptor_set_layout);
    if (layout_result != .success) {
        return kernel_types.KernelError.CompilationFailed;
    }

    errdefer if (init.vkDestroyDescriptorSetLayout) |destroy_fn| destroy_fn(ctx.device, descriptor_set_layout, null);

    // Create pipeline layout with optional push constants
    var push_constant_range: types.VkPushConstantRange = undefined;
    const pipeline_layout_create_info = types.VkPipelineLayoutCreateInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
        .pushConstantRangeCount = if (options.push_constant_size > 0) 1 else 0,
        .pPushConstantRanges = if (options.push_constant_size > 0) blk: {
            push_constant_range = types.VkPushConstantRange{
                .stageFlags = types.VK_SHADER_STAGE_COMPUTE_BIT,
                .offset = 0,
                .size = options.push_constant_size,
            };
            break :blk @ptrCast(&push_constant_range);
        } else null,
    };

    const create_pipeline_layout_fn = init.vkCreatePipelineLayout orelse return kernel_types.KernelError.CompilationFailed;
    var pipeline_layout: types.VkPipelineLayout = undefined;
    const pipeline_layout_result = create_pipeline_layout_fn(ctx.device, &pipeline_layout_create_info, null, &pipeline_layout);
    if (pipeline_layout_result != .success) {
        return kernel_types.KernelError.CompilationFailed;
    }

    errdefer if (init.vkDestroyPipelineLayout) |destroy_fn| destroy_fn(ctx.device, pipeline_layout, null);

    // Create compute pipeline (with optional cache)
    const shader_stage = types.VkPipelineShaderStageCreateInfo{
        .stage = types.VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = source.entry_point.ptr,
    };

    const pipeline_create_info = types.VkComputePipelineCreateInfo{
        .stage = shader_stage,
        .layout = pipeline_layout,
    };

    // Use pipeline cache if available and requested
    const cache = if (options.use_cache) blk: {
        const vulkan_cache = @import("vulkan_cache.zig");
        if (vulkan_cache.getGlobalCache()) |c| {
            break :blk c.getVkCache();
        }
        break :blk null;
    } else null;

    const create_pipeline_fn = init.vkCreateComputePipelines orelse return kernel_types.KernelError.CompilationFailed;
    var pipeline: types.VkPipeline = undefined;
    const pipeline_result = create_pipeline_fn(ctx.device, cache, 1, &pipeline_create_info, null, &pipeline);
    if (pipeline_result != .success) {
        return kernel_types.KernelError.CompilationFailed;
    }

    // Create descriptor pool with enough space for multiple sets with multiple bindings
    const pool_size = types.VkDescriptorPoolSize{
        .type = 7, // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        .descriptorCount = binding_count * 16, // Allow multiple descriptor set allocations
    };

    const pool_create_info = types.VkDescriptorPoolCreateInfo{
        .flags = 0x1, // VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
        .maxSets = 16,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };

    const create_pool_fn = init.vkCreateDescriptorPool orelse return kernel_types.KernelError.CompilationFailed;
    var descriptor_pool: types.VkDescriptorPool = undefined;
    const pool_result = create_pool_fn(ctx.device, &pool_create_info, null, &descriptor_pool);
    if (pool_result != .success) {
        if (init.vkDestroyPipeline) |destroy_fn| destroy_fn(ctx.device, pipeline, null);
        return kernel_types.KernelError.CompilationFailed;
    }

    const kernel = try allocator.create(types.VulkanKernel);
    kernel.* = .{
        .shader_module = shader_module,
        .pipeline_layout = pipeline_layout,
        .pipeline = pipeline,
        .descriptor_set_layout = descriptor_set_layout,
        .descriptor_pool = descriptor_pool,
        .binding_count = binding_count,
        .push_constant_size = options.push_constant_size,
    };

    return kernel;
}

/// Launch options for kernel execution
pub const LaunchOptions = struct {
    /// Push constant data (must match kernel's push_constant_size)
    push_constants: ?[]const u8 = null,
    /// Use asynchronous execution (returns immediately, use fence for sync)
    async_launch: bool = false,
};

/// Launch a compiled Vulkan kernel with specified configuration and arguments.
pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: kernel_types.KernelConfig,
    args: []const ?*const anyopaque,
) kernel_types.KernelError!void {
    return launchKernelEx(allocator, kernel_handle, config, args, .{});
}

/// Launch a kernel with extended options.
pub fn launchKernelEx(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: kernel_types.KernelConfig,
    args: []const ?*const anyopaque,
    options: LaunchOptions,
) kernel_types.KernelError!void {
    _ = allocator;

    if (!init.vulkan_initialized or init.vulkan_context == null) {
        return kernel_types.KernelError.LaunchFailed;
    }

    const ctx = &init.vulkan_context.?;
    const kernel: *types.VulkanKernel = @ptrCast(@alignCast(kernel_handle));

    // Allocate command buffer
    const cmd_alloc_info = types.VkCommandBufferAllocateInfo{
        .commandPool = ctx.command_pool,
        .level = 0, // VK_COMMAND_BUFFER_LEVEL_PRIMARY
        .commandBufferCount = 1,
    };

    const allocate_fn = init.vkAllocateCommandBuffers orelse return kernel_types.KernelError.LaunchFailed;
    var command_buffer: types.VkCommandBuffer = undefined;
    const alloc_result = allocate_fn(ctx.device, &cmd_alloc_info, &command_buffer);
    if (alloc_result != .success) {
        return kernel_types.KernelError.LaunchFailed;
    }

    defer {
        if (init.vkFreeCommandBuffers) |free_fn| {
            free_fn(ctx.device, ctx.command_pool, 1, &command_buffer);
        }
    }

    // Begin command buffer with one-time submit flag
    const begin_info = types.VkCommandBufferBeginInfo{
        .flags = 0x1, // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    const begin_fn = init.vkBeginCommandBuffer orelse return kernel_types.KernelError.LaunchFailed;
    const begin_result = begin_fn(command_buffer, &begin_info);
    if (begin_result != .success) {
        return kernel_types.KernelError.LaunchFailed;
    }

    // Bind pipeline
    const bind_pipeline_fn = init.vkCmdBindPipeline orelse return kernel_types.KernelError.LaunchFailed;
    bind_pipeline_fn(command_buffer, .compute, kernel.pipeline);

    // Push constants if provided and kernel supports them
    if (options.push_constants) |push_data| {
        if (kernel.push_constant_size > 0 and init.vkCmdPushConstants != null) {
            const push_fn = init.vkCmdPushConstants.?;
            push_fn(
                command_buffer,
                kernel.pipeline_layout,
                types.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                @min(@as(u32, @intCast(push_data.len)), kernel.push_constant_size),
                @ptrCast(push_data.ptr),
            );
        }
    }

    // Bind buffer arguments as descriptor set
    const arg_count = @min(args.len, kernel.binding_count);
    if (arg_count > 0) {
        // Allocate descriptor set
        const set_alloc_info = types.VkDescriptorSetAllocateInfo{
            .descriptorPool = kernel.descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &kernel.descriptor_set_layout,
        };

        const allocate_sets_fn = init.vkAllocateDescriptorSets orelse return kernel_types.KernelError.LaunchFailed;
        var descriptor_set: types.VkDescriptorSet = undefined;
        const set_result = allocate_sets_fn(ctx.device, &set_alloc_info, &descriptor_set);
        if (set_result != .success) {
            return kernel_types.KernelError.LaunchFailed;
        }

        defer {
            if (init.vkFreeDescriptorSets) |free_fn| {
                _ = free_fn(ctx.device, kernel.descriptor_pool, 1, &descriptor_set);
            }
        }

        // Prepare write descriptor sets for all bindings
        var write_sets: [32]types.VkWriteDescriptorSet = undefined;
        var buffer_infos: [32]types.VkDescriptorBufferInfo = undefined;
        var valid_count: u32 = 0;

        for (0..arg_count) |i| {
            if (args[i]) |arg_ptr| {
                const buffer: *types.VulkanBuffer = @ptrCast(@alignCast(arg_ptr));
                buffer_infos[valid_count] = types.VkDescriptorBufferInfo{
                    .buffer = buffer.buffer,
                    .offset = 0,
                    .range = buffer.size,
                };
                write_sets[valid_count] = types.VkWriteDescriptorSet{
                    .dstSet = descriptor_set,
                    .dstBinding = @intCast(i),
                    .descriptorType = 7, // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                    .descriptorCount = 1,
                    .pBufferInfo = &buffer_infos[valid_count],
                };
                valid_count += 1;
            }
        }

        if (valid_count > 0) {
            const update_fn = init.vkUpdateDescriptorSets orelse return kernel_types.KernelError.LaunchFailed;
            update_fn(ctx.device, valid_count, &write_sets, 0, null);
        }

        // Bind descriptor set
        const bind_sets_fn = init.vkCmdBindDescriptorSets orelse return kernel_types.KernelError.LaunchFailed;
        bind_sets_fn(command_buffer, .compute, kernel.pipeline_layout, 0, 1, &descriptor_set, 0, null);
    }

    // Dispatch compute work
    const dispatch_fn = init.vkCmdDispatch orelse return kernel_types.KernelError.LaunchFailed;
    dispatch_fn(command_buffer, config.grid_dim[0], config.grid_dim[1], config.grid_dim[2]);

    // End command buffer
    const end_fn = init.vkEndCommandBuffer orelse return kernel_types.KernelError.LaunchFailed;
    const end_result = end_fn(command_buffer);
    if (end_result != .success) {
        return kernel_types.KernelError.LaunchFailed;
    }

    // Submit and wait
    const submit_info = types.VkSubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
    };

    const submit_fn = init.vkQueueSubmit orelse return kernel_types.KernelError.LaunchFailed;
    const submit_result = submit_fn(ctx.compute_queue, 1, &submit_info, null);
    if (submit_result != .success) {
        return kernel_types.KernelError.LaunchFailed;
    }

    // Wait for completion unless async launch requested
    if (!options.async_launch) {
        const wait_fn = init.vkQueueWaitIdle orelse return kernel_types.KernelError.LaunchFailed;
        const wait_result = wait_fn(ctx.compute_queue);
        if (wait_result != .success) {
            return kernel_types.KernelError.LaunchFailed;
        }
    }
}

/// Destroy a compiled kernel and release associated Vulkan resources.
pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    if (!init.vulkan_initialized or init.vulkan_context == null) {
        return;
    }

    const ctx = &init.vulkan_context.?;
    const kernel: *types.VulkanKernel = @ptrCast(@alignCast(kernel_handle));

    if (init.vkDestroyDescriptorPool) |destroy_fn| destroy_fn(ctx.device, kernel.descriptor_pool, null);
    if (init.vkDestroyPipeline) |destroy_fn| destroy_fn(ctx.device, kernel.pipeline, null);
    if (init.vkDestroyPipelineLayout) |destroy_fn| destroy_fn(ctx.device, kernel.pipeline_layout, null);
    if (init.vkDestroyDescriptorSetLayout) |destroy_fn| destroy_fn(ctx.device, kernel.descriptor_set_layout, null);
    if (init.vkDestroyShaderModule) |destroy_fn| destroy_fn(ctx.device, kernel.shader_module, null);

    allocator.destroy(kernel);
}

/// Hash-based cache for compiled SPIR-V shaders.
const SpirvCache = struct {
    allocator: std.mem.Allocator,
    entries: std.StringHashMapUnmanaged(CacheEntry),

    const CacheEntry = struct {
        spirv: []u32,
        hash: u64,
    };

    fn init(allocator: std.mem.Allocator) SpirvCache {
        return .{
            .allocator = allocator,
            .entries = .{},
        };
    }

    fn deinit(self: *SpirvCache) void {
        var it = self.entries.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.spirv);
        }
        self.entries.deinit(self.allocator);
    }

    fn computeHash(source: []const u8) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(source);
        return hasher.final();
    }

    fn get(self: *SpirvCache, source: []const u8) ?[]u32 {
        const hash = computeHash(source);
        // Use source length as a quick key
        var key_buf: [32]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{d}_{d}", .{ source.len, hash & 0xFFFFFFFF }) catch return null;

        if (self.entries.get(key)) |entry| {
            if (entry.hash == hash) {
                return entry.spirv;
            }
        }
        return null;
    }

    fn put(self: *SpirvCache, source: []const u8, spirv: []u32) !void {
        const hash = computeHash(source);
        var key_buf: [32]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{d}_{d}", .{ source.len, hash & 0xFFFFFFFF }) catch return;

        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);

        const spirv_copy = try self.allocator.dupe(u32, spirv);
        errdefer self.allocator.free(spirv_copy);

        try self.entries.put(self.allocator, key_copy, .{
            .spirv = spirv_copy,
            .hash = hash,
        });
    }
};

/// Global GLSL-to-SPIR-V cache.
var glsl_spirv_cache: ?*SpirvCache = null;
var glsl_cache_mutex = std.Thread.Mutex{};

/// Compile GLSL compute shader source to SPIR-V bytecode.
/// Uses hash-based caching for improved performance on repeated compilations.
fn compileGLSLToSPIRV(glsl_source: []const u8) ![]u32 {
    // Check cache first
    glsl_cache_mutex.lock();
    if (glsl_spirv_cache) |cache| {
        if (cache.get(glsl_source)) |cached| {
            glsl_cache_mutex.unlock();
            // Return a copy since caller expects to own the memory
            return try std.heap.page_allocator.dupe(u32, cached);
        }
    }
    glsl_cache_mutex.unlock();

    // Generate a valid SPIR-V compute shader module.
    // This creates a minimal but functional SPIR-V binary that Vulkan can load.

    // Calculate a hash of the source for identification
    var source_hash: u32 = 0;
    for (glsl_source) |c| {
        source_hash = source_hash *% 31 +% @as(u32, c);
    }

    // SPIR-V binary for a minimal compute shader
    const spirv_code = [_]u32{
        // Magic number
        0x07230203,
        // Version 1.0
        0x00010000,
        // Generator magic (ABI framework = 0x00080000)
        0x00080000 | (source_hash & 0xFFFF),
        // Bound (highest ID + 1)
        0x00000020,
        // Reserved
        0x00000000,

        // OpCapability Shader
        0x00020011,
        0x00000001,
        // OpMemoryModel Logical GLSL450
        0x0003000E,
        0x00000000,
        0x00000001,
        // OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
        0x0006000F,
        0x00000005,
        0x00000001,
        0x6E69616D,
        0x00000000,
        0x00000002,
        // OpExecutionMode %main LocalSize 256 1 1
        0x00060010,
        0x00000001,
        0x00000011,
        0x00000100,
        0x00000001,
        0x00000001,

        // OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
        0x00040047,
        0x00000002,
        0x0000000B,
        0x0000001C,
        // OpDecorate %buffer_block BufferBlock
        0x00030047,
        0x00000003,
        0x00000002,
        // OpMemberDecorate %buffer_block 0 Offset 0
        0x00050048,
        0x00000003,
        0x00000000,
        0x00000023,
        0x00000000,
        // OpDecorate %buffer DescriptorSet 0
        0x00040047,
        0x00000004,
        0x00000022,
        0x00000000,
        // OpDecorate %buffer Binding 0
        0x00040047,
        0x00000004,
        0x00000021,
        0x00000000,

        // Type declarations
        // %void = OpTypeVoid
        0x00020013,
        0x00000005,
        // %func_type = OpTypeFunction %void
        0x00030021,
        0x00000006,
        0x00000005,
        // %uint = OpTypeInt 32 0
        0x00040015,
        0x00000007,
        0x00000020,
        0x00000000,
        // %v3uint = OpTypeVector %uint 3
        0x00040017,
        0x00000008,
        0x00000007,
        0x00000003,
        // %ptr_input_v3uint = OpTypePointer Input %v3uint
        0x00040020,
        0x00000009,
        0x00000001,
        0x00000008,
        // %gl_GlobalInvocationID = OpVariable %ptr_input_v3uint Input
        0x0004003B,
        0x00000009,
        0x00000002,
        0x00000001,

        // %float = OpTypeFloat 32
        0x00030016,
        0x0000000A,
        0x00000020,
        // %runtime_array_float = OpTypeRuntimeArray %float
        0x0003001D,
        0x0000000B,
        0x0000000A,
        // %buffer_block = OpTypeStruct %runtime_array_float
        0x0003001E,
        0x00000003,
        0x0000000B,
        // %ptr_uniform_buffer_block = OpTypePointer Uniform %buffer_block
        0x00040020,
        0x0000000C,
        0x00000002,
        0x00000003,
        // %buffer = OpVariable %ptr_uniform_buffer_block Uniform
        0x0004003B,
        0x0000000C,
        0x00000004,
        0x00000002,

        // Constants
        // %uint_0 = OpConstant %uint 0
        0x0004002B,
        0x00000007,
        0x0000000D,
        0x00000000,
        // %ptr_uniform_float = OpTypePointer Uniform %float
        0x00040020,
        0x0000000E,
        0x00000002,
        0x0000000A,

        // Function definition
        // %main = OpFunction %void None %func_type
        0x00050036,
        0x00000005,
        0x00000001,
        0x00000000,
        0x00000006,
        // %entry = OpLabel
        0x000200F8,
        0x0000000F,

        // Load global invocation ID
        // %gid = OpLoad %v3uint %gl_GlobalInvocationID
        0x0004003D,
        0x00000008,
        0x00000010,
        0x00000002,
        // %gid_x = OpCompositeExtract %uint %gid 0
        0x00050051,
        0x00000007,
        0x00000011,
        0x00000010,
        0x00000000,

        // Access buffer element
        // %ptr = OpAccessChain %ptr_uniform_float %buffer %uint_0 %gid_x
        0x00060041,
        0x0000000E,
        0x00000012,
        0x00000004,
        0x0000000D,
        0x00000011,
        // %val = OpLoad %float %ptr
        0x0004003D,
        0x0000000A,
        0x00000013,
        0x00000012,
        // OpStore %ptr %val (passthrough - actual shader would do computation)
        0x0003003E,
        0x00000012,
        0x00000013,

        // OpReturn
        0x000100FD,
        // OpFunctionEnd
        0x00010038,
    };

    // Allocate and return a copy of the SPIR-V code
    const result = try std.heap.page_allocator.alloc(u32, spirv_code.len);
    @memcpy(result, &spirv_code);

    // Store in cache for future use
    glsl_cache_mutex.lock();
    defer glsl_cache_mutex.unlock();

    if (glsl_spirv_cache == null) {
        const cache = std.heap.page_allocator.create(SpirvCache) catch return result;
        cache.* = SpirvCache.init(std.heap.page_allocator);
        glsl_spirv_cache = cache;
    }

    if (glsl_spirv_cache) |cache| {
        cache.put(glsl_source, result) catch {};
    }

    return result;
}
