//! Vulkan VTable Backend Implementation
//!
//! Implements the unified GPU backend interface for Vulkan, providing
//! memory allocation, kernel compilation, and dispatch.

const std = @import("std");
const builtin = @import("builtin");
const interface = @import("../../interface.zig");
const vulkan = @import("../vulkan.zig");

const VulkanKernel = vulkan.VulkanKernel;
const VulkanError = vulkan.VulkanError;
const VkBuffer = vulkan.VkBuffer;
const VkDeviceMemory = vulkan.VkDeviceMemory;
const VkShaderModule = vulkan.VkShaderModule;
const VkPipelineLayout = vulkan.VkPipelineLayout;
const VkPipeline = vulkan.VkPipeline;
const VkDescriptorSetLayout = vulkan.VkDescriptorSetLayout;
const VkDescriptorPool = vulkan.VkDescriptorPool;
const VkDescriptorSet = vulkan.VkDescriptorSet;
const VkCommandBuffer = vulkan.VkCommandBuffer;
const VkPhysicalDeviceProperties = vulkan.VkPhysicalDeviceProperties;

pub const VulkanBackend = struct {
    allocator: std.mem.Allocator,

    // Track resources
    allocations: std.ArrayListUnmanaged(Allocation),
    kernels: std.ArrayListUnmanaged(CompiledKernel),

    const Allocation = struct {
        ptr: *anyopaque,
        buffer: VkBuffer,
        memory: VkDeviceMemory,
        size: usize,
    };

    const CompiledKernel = struct {
        handle: *anyopaque, // Points to VulkanKernel
        name: []const u8,
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) interface.BackendError!*Self {
        // Initialize global Vulkan state if needed
        if (!vulkan.vulkan_initialized.load(.acquire)) {
            vulkan.initVulkanGlobal(allocator) catch |err| {
                return switch (err) {
                    VulkanError.InitializationFailed => interface.BackendError.InitFailed,
                    else => interface.BackendError.NotAvailable,
                };
            };
        }

        const self = allocator.create(Self) catch return interface.BackendError.OutOfMemory;
        self.* = .{
            .allocator = allocator,
            .allocations = .empty,
            .kernels = .empty,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        const ctx = vulkan.vulkan_context.?;

        // Free allocations
        for (self.allocations.items) |alloc| {
            vulkan.vkUnmapMemory.?(ctx.device, alloc.memory);
            vulkan.vkDestroyBuffer.?(ctx.device, alloc.buffer, null);
            vulkan.vkFreeMemory.?(ctx.device, alloc.memory, null);
        }
        self.allocations.deinit(self.allocator);

        // Destroy kernels
        for (self.kernels.items) |k| {
            const kernel: *VulkanKernel = @ptrCast(@alignCast(k.handle));
            vulkan.vkDestroyPipeline.?(ctx.device, kernel.pipeline, null);
            vulkan.vkDestroyPipelineLayout.?(ctx.device, kernel.pipeline_layout, null);
            vulkan.vkDestroyDescriptorSetLayout.?(ctx.device, kernel.descriptor_set_layout, null);
            vulkan.vkDestroyDescriptorPool.?(ctx.device, kernel.descriptor_pool, null);
            vulkan.vkDestroyShaderModule.?(ctx.device, kernel.shader_module, null);
            self.allocator.destroy(kernel);
            self.allocator.free(k.name);
        }
        self.kernels.deinit(self.allocator);

        self.allocator.destroy(self);
    }

    pub fn getDeviceCount(_: *Self) u32 {
        if (!vulkan.vulkan_initialized.load(.acquire)) return 0;
        return 1;
    }

    pub fn getDeviceCaps(_: *Self, device_id: u32) interface.BackendError!interface.DeviceCaps {
        if (device_id != 0) return interface.BackendError.DeviceNotFound;
        const ctx = vulkan.vulkan_context orelse return interface.BackendError.NotAvailable;

        var caps = interface.DeviceCaps{
            .max_threads_per_block = 1024,
            .max_shared_memory = 32768,
            .warp_size = 32,
            .supports_fp16 = true,
            .supports_fp64 = true,
            .supports_int8 = true,
            .unified_memory = false,
        };

        var props: VkPhysicalDeviceProperties = undefined;
        vulkan.vkGetPhysicalDeviceProperties.?(ctx.physical_device, @ptrCast(&props));

        const name_len = std.mem.indexOfScalar(u8, &props.deviceName, 0) orelse 256;
        @memcpy(caps.name[0..name_len], props.deviceName[0..name_len]);
        caps.name_len = name_len;

        caps.max_threads_per_block = props.limits.maxComputeWorkGroupInvocations;
        caps.max_shared_memory = props.limits.maxComputeSharedMemorySize;

        // Total device memory from memory heaps
        var total_mem: u64 = 0;
        for (0..ctx.memory_properties.memoryHeapCount) |h| {
            total_mem += ctx.memory_properties.memoryHeaps[h].size;
        }
        caps.total_memory = total_mem;

        return caps;
    }

    pub fn allocate(self: *Self, size: usize, flags: interface.MemoryFlags) interface.MemoryError!*anyopaque {
        _ = flags;
        const ctx = vulkan.vulkan_context.?;

        const buffer_info = vulkan.VkBufferCreateInfo{
            .size = size,
            .usage = vulkan.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vulkan.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vulkan.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = 0, // Exclusive
        };

        var buffer: VkBuffer = undefined;
        if (vulkan.vkCreateBuffer.?(ctx.device, &buffer_info, null, &buffer) != .success) {
            return interface.MemoryError.OutOfMemory;
        }

        var mem_reqs: vulkan.VkMemoryRequirements = undefined;
        vulkan.vkGetBufferMemoryRequirements.?(ctx.device, buffer, @ptrCast(&mem_reqs));

        const mem_type_index = vulkan.findMemoryType(mem_reqs.memoryTypeBits, vulkan.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vulkan.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) catch {
            vulkan.vkDestroyBuffer.?(ctx.device, buffer, null);
            return interface.MemoryError.OutOfMemory;
        };

        const alloc_info = vulkan.VkMemoryAllocateInfo{
            .allocationSize = mem_reqs.size,
            .memoryTypeIndex = mem_type_index,
        };

        var memory: VkDeviceMemory = undefined;
        if (vulkan.vkAllocateMemory.?(ctx.device, &alloc_info, null, &memory) != .success) {
            vulkan.vkDestroyBuffer.?(ctx.device, buffer, null);
            return interface.MemoryError.OutOfMemory;
        }

        if (vulkan.vkBindBufferMemory.?(ctx.device, buffer, memory, 0) != .success) {
            vulkan.vkFreeMemory.?(ctx.device, memory, null);
            vulkan.vkDestroyBuffer.?(ctx.device, buffer, null);
            return interface.MemoryError.OutOfMemory;
        }

        var ptr: ?*anyopaque = null;
        if (vulkan.vkMapMemory.?(ctx.device, memory, 0, size, 0, &ptr) != .success) {
            vulkan.vkFreeMemory.?(ctx.device, memory, null);
            vulkan.vkDestroyBuffer.?(ctx.device, buffer, null);
            return interface.MemoryError.OutOfMemory;
        }

        self.allocations.append(self.allocator, .{
            .ptr = ptr.?,
            .buffer = buffer,
            .memory = memory,
            .size = size,
        }) catch {
            return interface.MemoryError.OutOfMemory;
        };

        return ptr.?;
    }

    pub fn free(self: *Self, ptr: *anyopaque) void {
        const ctx = vulkan.vulkan_context.?;
        for (self.allocations.items, 0..) |alloc, i| {
            if (alloc.ptr == ptr) {
                vulkan.vkUnmapMemory.?(ctx.device, alloc.memory);
                vulkan.vkDestroyBuffer.?(ctx.device, alloc.buffer, null);
                vulkan.vkFreeMemory.?(ctx.device, alloc.memory, null);
                _ = self.allocations.swapRemove(i);
                return;
            }
        }
    }

    pub fn copyToDevice(_: *Self, dst: *anyopaque, src: []const u8) interface.MemoryError!void {
        const dst_ptr: [*]u8 = @ptrCast(dst);
        @memcpy(dst_ptr[0..src.len], src);
    }

    pub fn copyFromDevice(_: *Self, dst: []u8, src: *anyopaque) interface.MemoryError!void {
        const src_ptr: [*]const u8 = @ptrCast(src);
        @memcpy(dst, src_ptr[0..dst.len]);
    }

    pub fn copyToDeviceAsync(self: *Self, dst: *anyopaque, src: []const u8, stream: ?*anyopaque) interface.MemoryError!void {
        _ = stream;
        return Self.copyToDevice(self, dst, src);
    }

    pub fn copyFromDeviceAsync(self: *Self, dst: []u8, src: *anyopaque, stream: ?*anyopaque) interface.MemoryError!void {
        _ = stream;
        return Self.copyFromDevice(self, dst, src);
    }

    pub fn compileKernel(self: *Self, allocator: std.mem.Allocator, source: []const u8, kernel_name: []const u8) interface.KernelError!*anyopaque {
        const ctx = vulkan.vulkan_context.?;

        // 1. Create Shader Module (assumes SPIR-V source)
        if (source.len % 4 != 0) return interface.KernelError.CompileFailed;

        const create_info = vulkan.VkShaderModuleCreateInfo{
            .codeSize = source.len,
            .pCode = @ptrCast(@alignCast(source.ptr)),
        };

        var shader_module: VkShaderModule = undefined;
        if (vulkan.vkCreateShaderModule.?(ctx.device, &create_info, null, &shader_module) != .success) {
            return interface.KernelError.CompileFailed;
        }

        // 2. Create Descriptor Set Layout (8 storage buffers)
        var bindings: [8]vulkan.VkDescriptorSetLayoutBinding = undefined;
        for (0..8) |i| {
            bindings[i] = vulkan.VkDescriptorSetLayoutBinding{
                .binding = @intCast(i),
                .descriptorType = 7, // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                .descriptorCount = 1,
                .stageFlags = vulkan.VK_SHADER_STAGE_COMPUTE_BIT,
            };
        }

        const descriptor_layout_info = vulkan.VkDescriptorSetLayoutCreateInfo{
            .bindingCount = 8,
            .pBindings = &bindings,
        };

        var descriptor_set_layout: VkDescriptorSetLayout = undefined;
        if (vulkan.vkCreateDescriptorSetLayout.?(ctx.device, &descriptor_layout_info, null, &descriptor_set_layout) != .success) {
            return interface.KernelError.CompileFailed;
        }

        // 3. Create Pipeline Layout
        const pipeline_layout_info = vulkan.VkPipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = @ptrCast(&descriptor_set_layout),
        };

        var pipeline_layout: VkPipelineLayout = undefined;
        if (vulkan.vkCreatePipelineLayout.?(ctx.device, &pipeline_layout_info, null, &pipeline_layout) != .success) {
            vulkan.vkDestroyDescriptorSetLayout.?(ctx.device, descriptor_set_layout, null);
            return interface.KernelError.CompileFailed;
        }

        // 4. Create Compute Pipeline
        const shader_stage_info = vulkan.VkPipelineShaderStageCreateInfo{
            .stage = vulkan.VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_module,
            .pName = "main",
        };

        const pipeline_info = vulkan.VkComputePipelineCreateInfo{
            .stage = shader_stage_info,
            .layout = pipeline_layout,
        };

        var pipeline: VkPipeline = undefined;
        if (vulkan.vkCreateComputePipelines.?(ctx.device, null, 1, @ptrCast(&pipeline_info), null, @ptrCast(&pipeline)) != .success) {
            vulkan.vkDestroyPipelineLayout.?(ctx.device, pipeline_layout, null);
            vulkan.vkDestroyDescriptorSetLayout.?(ctx.device, descriptor_set_layout, null);
            return interface.KernelError.CompileFailed;
        }

        // 5. Create Descriptor Pool
        const pool_size = vulkan.VkDescriptorPoolSize{
            .type = 7, // STORAGE_BUFFER
            .descriptorCount = 8,
        };
        const pool_info = vulkan.VkDescriptorPoolCreateInfo{
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = @ptrCast(&pool_size),
        };
        var descriptor_pool: VkDescriptorPool = undefined;
        if (vulkan.vkCreateDescriptorPool.?(ctx.device, &pool_info, null, &descriptor_pool) != .success) {
            return interface.KernelError.CompileFailed;
        }

        const vulkan_kernel = allocator.create(VulkanKernel) catch return interface.KernelError.CompileFailed;
        vulkan_kernel.* = .{
            .shader_module = shader_module,
            .pipeline_layout = pipeline_layout,
            .pipeline = pipeline,
            .descriptor_set_layout = descriptor_set_layout,
            .descriptor_pool = descriptor_pool,
        };

        // Track
        const name_copy = self.allocator.dupe(u8, kernel_name) catch return interface.KernelError.CompileFailed;
        self.kernels.append(self.allocator, .{ .handle = vulkan_kernel, .name = name_copy }) catch return interface.KernelError.CompileFailed;

        return vulkan_kernel;
    }

    pub fn launchKernel(self: *Self, kernel_handle: *anyopaque, config: interface.LaunchConfig, args: []const *anyopaque) interface.KernelError!void {
        const ctx = vulkan.vulkan_context.?;
        const kernel: *VulkanKernel = @ptrCast(@alignCast(kernel_handle));

        // 1. Allocate Descriptor Set
        const alloc_info = vulkan.VkDescriptorSetAllocateInfo{
            .descriptorPool = kernel.descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = @ptrCast(&kernel.descriptor_set_layout),
        };
        var descriptor_set: VkDescriptorSet = undefined;
        if (vulkan.vkAllocateDescriptorSets.?(ctx.device, &alloc_info, @ptrCast(&descriptor_set)) != .success) {
            return interface.KernelError.LaunchFailed;
        }

        // 2. Update Descriptor Set
        var writes: [8]vulkan.VkWriteDescriptorSet = undefined;
        var buffer_infos: [8]vulkan.VkDescriptorBufferInfo = undefined;

        for (args, 0..) |arg, i| {
            if (i >= 8) break;

            var buffer_handle: VkBuffer = undefined;
            var found = false;
            for (self.allocations.items) |alloc| {
                if (alloc.ptr == arg) {
                    buffer_handle = alloc.buffer;
                    found = true;
                    break;
                }
            }
            if (!found) continue;

            buffer_infos[i] = vulkan.VkDescriptorBufferInfo{
                .buffer = buffer_handle,
                .offset = 0,
                .range = 0xFFFFFFFFFFFFFFFF, // VK_WHOLE_SIZE
            };

            writes[i] = vulkan.VkWriteDescriptorSet{
                .dstSet = descriptor_set,
                .dstBinding = @intCast(i),
                .descriptorCount = 1,
                .descriptorType = 7, // STORAGE_BUFFER
                .pBufferInfo = &buffer_infos[i],
            };
        }

        vulkan.vkUpdateDescriptorSets.?(ctx.device, @intCast(@min(args.len, 8)), &writes, 0, null);

        // 3. Record Command Buffer
        const alloc_cmd_info = vulkan.VkCommandBufferAllocateInfo{
            .commandPool = ctx.command_pool,
            .level = 0, // PRIMARY
            .commandBufferCount = 1,
        };
        var cmd_buffer: VkCommandBuffer = undefined;
        if (vulkan.vkAllocateCommandBuffers.?(ctx.device, &alloc_cmd_info, @ptrCast(&cmd_buffer)) != .success) {
            return interface.KernelError.LaunchFailed;
        }

        const begin_info = vulkan.VkCommandBufferBeginInfo{
            .flags = 0x00000004, // ONE_TIME_SUBMIT
        };
        _ = vulkan.vkBeginCommandBuffer.?(cmd_buffer, &begin_info);

        vulkan.vkCmdBindPipeline.?(cmd_buffer, .compute, kernel.pipeline);
        vulkan.vkCmdBindDescriptorSets.?(cmd_buffer, .compute, kernel.pipeline_layout, 0, 1, @ptrCast(&descriptor_set), 0, null);
        vulkan.vkCmdDispatch.?(cmd_buffer, config.grid_x, config.grid_y, config.grid_z);

        _ = vulkan.vkEndCommandBuffer.?(cmd_buffer);

        // 4. Submit
        const submit_info = vulkan.VkSubmitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = @ptrCast(&cmd_buffer),
        };
        if (vulkan.vkQueueSubmit.?(ctx.compute_queue, 1, @ptrCast(&submit_info), null) != .success) {
            return interface.KernelError.LaunchFailed;
        }

        _ = vulkan.vkQueueWaitIdle.?(ctx.compute_queue);
        vulkan.vkFreeCommandBuffers.?(ctx.device, ctx.command_pool, 1, @ptrCast(&cmd_buffer));
        _ = vulkan.vkFreeDescriptorSets.?(ctx.device, kernel.descriptor_pool, 1, @ptrCast(&descriptor_set));
    }

    pub fn destroyKernel(self: *Self, kernel_handle: *anyopaque) void {
        const ctx = vulkan.vulkan_context.?;
        const kernel: *VulkanKernel = @ptrCast(@alignCast(kernel_handle));

        for (self.kernels.items, 0..) |k, i| {
            if (k.handle == kernel_handle) {
                _ = self.kernels.swapRemove(i);
                self.allocator.free(k.name);
                break;
            }
        }

        if (vulkan.vkDestroyPipeline) |destroy_pipeline| {
            destroy_pipeline(ctx.device, kernel.pipeline, null);
        }
        if (vulkan.vkDestroyPipelineLayout) |destroy_pipeline_layout| {
            destroy_pipeline_layout(ctx.device, kernel.pipeline_layout, null);
        }
        if (vulkan.vkDestroyDescriptorSetLayout) |destroy_descriptor_set_layout| {
            destroy_descriptor_set_layout(ctx.device, kernel.descriptor_set_layout, null);
        }
        if (vulkan.vkDestroyDescriptorPool) |destroy_descriptor_pool| {
            destroy_descriptor_pool(ctx.device, kernel.descriptor_pool, null);
        }
        if (vulkan.vkDestroyShaderModule) |destroy_shader_module| {
            destroy_shader_module(ctx.device, kernel.shader_module, null);
        }

        self.allocator.destroy(kernel);
    }

    pub fn synchronize(_: *Self) interface.BackendError!void {
        const ctx = vulkan.vulkan_context.?;
        _ = vulkan.vkQueueWaitIdle.?(ctx.compute_queue);
    }
};
