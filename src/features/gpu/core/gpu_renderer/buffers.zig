const std = @import("std");
const gpu = std.gpu;
const print = std.debug.print;

const config = @import("config.zig");

const BufferUsage = config.BufferUsage;
const GpuError = config.GpuError;
const GPUHandle = config.GPUHandle;
const Backend = config.Backend;

/// Mock GPU types for CPU fallback
pub const MockGPU = struct {
    pub const Instance = struct {
        allocator: std.mem.Allocator,

        pub fn create(allocator: std.mem.Allocator) !*Instance {
            const instance = try allocator.create(Instance);
            instance.* = .{ .allocator = allocator };
            return instance;
        }

        pub fn deinit(self: *Instance) void {
            self.allocator.destroy(self);
        }

        pub fn requestAdapter(self: *Instance) !*Adapter {
            return try Adapter.create(self.allocator);
        }
    };

    pub const Adapter = struct {
        allocator: std.mem.Allocator,

        pub fn create(allocator: std.mem.Allocator) !*Adapter {
            const adapter = try allocator.create(Adapter);
            adapter.* = .{ .allocator = allocator };
            return adapter;
        }

        pub fn deinit(self: *Adapter) void {
            self.allocator.destroy(self);
        }

        pub fn getName(self: *Adapter) []const u8 {
            _ = self;
            return "CPU Fallback Renderer";
        }

        pub fn requestDevice(self: *Adapter) !*Device {
            return try Device.create(self.allocator);
        }
    };

    pub const Device = struct {
        allocator: std.mem.Allocator,
        queue: *Queue,

        pub fn create(allocator: std.mem.Allocator) !*Device {
            const device = try allocator.create(Device);
            const queue = try Queue.create(allocator);
            device.* = .{ .allocator = allocator, .queue = queue };
            return device;
        }

        pub fn deinit(self: *Device) void {
            self.queue.deinit();
            self.allocator.destroy(self);
        }

        pub fn createBuffer(self: *Device, size: usize, usage: BufferUsage) !*MockGPU.Buffer {
            _ = usage;
            return try MockGPU.Buffer.create(self.allocator, size);
        }
    };

    pub const Queue = struct {
        allocator: std.mem.Allocator,

        pub fn create(allocator: std.mem.Allocator) !*Queue {
            const queue = try allocator.create(Queue);
            queue.* = .{ .allocator = allocator };
            return queue;
        }

        pub fn deinit(self: *Queue) void {
            self.allocator.destroy(self);
        }

        pub fn writeBuffer(self: *Queue, buffer: *MockGPU.Buffer, data: []const u8) void {
            _ = self;
            @memcpy(buffer.data[0..@min(data.len, buffer.size)], data[0..@min(data.len, buffer.size)]);
        }

        pub fn submit(self: *Queue) void {
            _ = self;
            // No-op for CPU fallback
        }

        pub fn onSubmittedWorkDone(self: *Queue) void {
            _ = self;
            // No-op for CPU fallback
        }
    };

    pub const Buffer = struct {
        allocator: std.mem.Allocator,
        data: []align(16) u8,
        size: usize,

        pub fn create(allocator: std.mem.Allocator, size: usize) !*MockGPU.Buffer {
            const buffer = try allocator.create(MockGPU.Buffer);
            const data_raw = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(16), size);
            @memset(data_raw, 0);
            const data_aligned: []align(16) u8 = @alignCast(data_raw);
            buffer.* = .{ .allocator = allocator, .data = data_aligned, .size = size };
            return buffer;
        }

        pub fn deinit(self: *MockGPU.Buffer) void {
            // alignedAlloc memory must be freed with free() on Zig 0.15 allocators
            self.allocator.free(self.data);
            self.allocator.destroy(self);
        }

        pub fn getMappedRange(self: *MockGPU.Buffer, comptime T: type, offset: usize, length: usize) ?[]T {
            const byte_offset = offset * @sizeOf(T);
            const byte_length = length * @sizeOf(T);
            if (byte_offset + byte_length > self.size) return null;

            const bytes = self.data[byte_offset .. byte_offset + byte_length];
            const aligned_bytes: []align(@alignOf(T)) u8 = @alignCast(bytes);
            return std.mem.bytesAsSlice(T, aligned_bytes);
        }

        pub fn unmap(self: *MockGPU.Buffer) void {
            _ = self;
            // No-op for CPU fallback
        }
    };
};

/// Hardware GPU context backed by std.gpu for native backends
pub const HardwareContext = struct {
    instance: *gpu.Instance,
    adapter: *gpu.Adapter,
    device: *gpu.Device,
    queue: *gpu.Queue,

    pub fn init(backend: Backend) !HardwareContext {
        var backends = gpu.Instance.Backends{};
        switch (backend) {
            .vulkan => backends.vulkan = true,
            .metal => backends.metal = true,
            .dx12 => backends.dx12 = true,
            .opengl => backends.gl = true,
            .webgpu => backends.webgpu = true,
            else => backends = gpu.Instance.Backends.primary,
        }

        const instance = gpu.Instance.create(.{
            .backends = backends,
            .dx12_shader_compiler = .fxc,
        }) orelse return GpuError.GpuInstanceCreationFailed;

        const adapter = instance.requestAdapter(.{
            .power_preference = .high_performance,
            .force_fallback_adapter = false,
        }) orelse return GpuError.NoSuitableAdapter;

        const device = adapter.requestDevice(.{
            .label = "abi-gpu-device",
            .required_features = .{},
            .required_limits = .{},
        }) orelse return GpuError.DeviceCreationFailed;

        const queue = device.queue;

        return .{
            .instance = instance,
            .adapter = adapter,
            .device = device,
            .queue = queue,
        };
    }

    pub fn deinit(self: *HardwareContext) void {
        self.device.deinit();
        self.adapter.deinit();
        self.instance.deinit();
    }
};

/// Extended GPU context with compiler support for CPU fallback
pub const GPUContext = struct {
    instance: *MockGPU.Instance,
    adapter: *MockGPU.Adapter,
    device: *MockGPU.Device,
    queue: *MockGPU.Queue,
    allocator: std.mem.Allocator,

    /// Initialize GPU context with error handling
    pub fn init(allocator: std.mem.Allocator) !GPUContext {
        const instance = try MockGPU.Instance.create(allocator);
        const adapter = try instance.requestAdapter();
        const device = try adapter.requestDevice();
        const queue = device.queue;

        return .{
            .instance = instance,
            .adapter = adapter,
            .device = device,
            .queue = queue,
            .allocator = allocator,
        };
    }

    /// Initialize Vulkan-specific context
    pub fn initVulkan(allocator: std.mem.Allocator) !GPUContext {
        // Mock Vulkan initialization - in real implementation, this would:
        // 1. Create VkInstance with validation layers
        // 2. Select physical device
        // 3. Create logical device with compute/graphics queues
        // 4. Initialize memory allocator
        return try init(allocator);
    }

    /// Initialize Metal-specific context
    pub fn initMetal(allocator: std.mem.Allocator) !GPUContext {
        // Mock Metal initialization - in real implementation, this would:
        // 1. Get default MTLDevice
        // 2. Create command queue
        // 3. Initialize Metal Performance Shaders framework
        return try init(allocator);
    }

    /// Initialize DirectX 12-specific context
    pub fn initDX12(allocator: std.mem.Allocator) !GPUContext {
        // Mock DX12 initialization - in real implementation, this would:
        // 1. Create D3D12Device
        // 2. Create command queue
        // 3. Initialize DXGI swap chain
        return try init(allocator);
    }

    /// Initialize OpenGL-specific context
    pub fn initOpenGL(allocator: std.mem.Allocator) !GPUContext {
        // Mock OpenGL initialization - in real implementation, this would:
        // 1. Create OpenGL context
        // 2. Load OpenGL extensions
        // 3. Initialize vertex array objects
        return try init(allocator);
    }

    /// Initialize OpenCL-specific context
    pub fn initOpenCL(allocator: std.mem.Allocator) !GPUContext {
        // Mock OpenCL initialization - in real implementation, this would:
        // 1. Query OpenCL platforms and devices
        // 2. Create OpenCL context
        // 3. Create command queue
        return try init(allocator);
    }

    /// Initialize CUDA-specific context
    pub fn initCUDA(allocator: std.mem.Allocator) !GPUContext {
        // Mock CUDA initialization - in real implementation, this would:
        // 1. Initialize CUDA runtime
        // 2. Select CUDA device
        // 3. Create CUDA context
        return try init(allocator);
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
        print("Device Features: CPU Fallback Mode\n", .{});
    }
};

/// Buffer manager that can operate on CPU or hardware GPU resources
pub const BufferManager = struct {
    device: Device,
    queue: Queue,

    const Device = union(enum) {
        mock: *MockGPU.Device,
        hardware: *gpu.Device,
    };

    const Queue = union(enum) {
        mock: *MockGPU.Queue,
        hardware: *gpu.Queue,
    };

    fn toGpuUsage(usage: BufferUsage) gpu.Buffer.Usage {
        return .{
            .map_read = usage.map_read,
            .map_write = usage.map_write,
            .copy_src = usage.copy_src,
            .copy_dst = usage.copy_dst,
            .storage = usage.storage,
            .uniform = usage.uniform,
            .vertex = usage.vertex,
            .index = usage.index,
        };
    }

    pub fn createBuffer(self: BufferManager, comptime T: type, size: u64, usage: BufferUsage) !Buffer.Resource {
        return switch (self.device) {
            .mock => |device| blk: {
                const byte_size = size * @sizeOf(T);
                const buffer = try device.createBuffer(@intCast(byte_size), usage);
                break :blk Buffer.Resource{ .mock = buffer };
            },
            .hardware => |device| blk: {
                const buffer = device.createBuffer(.{
                    .label = @typeName(T) ++ " Buffer",
                    .size = size * @sizeOf(T),
                    .usage = toGpuUsage(usage),
                    .mapped_at_creation = false,
                }) orelse return GpuError.BufferCreationFailed;
                break :blk Buffer.Resource{ .hardware = buffer };
            },
        };
    }

    pub fn writeBuffer(self: BufferManager, buffer: *Buffer, data: []const u8) void {
        switch (buffer.resource) {
            .mock => |mock_buf| switch (self.queue) {
                .mock => |queue| queue.writeBuffer(mock_buf, data),
                else => {},
            },
            .hardware => |hw_buf| switch (self.queue) {
                .hardware => |queue| queue.writeBuffer(hw_buf, 0, data),
                else => {},
            },
        }
    }

    pub fn readBuffer(
        self: BufferManager,
        comptime T: type,
        buffer: *Buffer,
        size: u64,
        allocator: std.mem.Allocator,
    ) ![]T {
        return switch (buffer.resource) {
            .mock => |mock_buf| blk: {
                const mapped_range = mock_buf.getMappedRange(T, 0, @intCast(size)) orelse return GpuError.BufferMappingFailed;
                break :blk try allocator.dupe(T, mapped_range);
            },
            .hardware => |hw_buf| blk: {
                const device = switch (self.device) {
                    .hardware => |dev| dev,
                    else => return GpuError.UnsupportedBackend,
                };

                const staging = device.createBuffer(.{
                    .label = "abi-staging-buffer",
                    .size = size * @sizeOf(T),
                    .usage = .{ .copy_dst = true, .map_read = true },
                    .mapped_at_creation = false,
                }) orelse return GpuError.BufferCreationFailed;
                defer staging.deinit();

                const encoder = device.createCommandEncoder(.{ .label = "abi-readback-encoder" }) orelse return GpuError.CommandEncoderCreationFailed;
                defer encoder.deinit();

                encoder.copyBufferToBuffer(hw_buf, 0, staging, 0, size * @sizeOf(T));

                const command = encoder.finish(.{ .label = "abi-readback-cmd" }) orelse return GpuError.CommandCreationFailed;
                defer command.deinit();

                switch (self.queue) {
                    .hardware => |queue| {
                        queue.submit(&[_]*gpu.CommandBuffer{command});
                        queue.onSubmittedWorkDone(null, null);
                    },
                    else => return GpuError.UnsupportedBackend,
                }

                const mapped = staging.getMappedRange(T, 0, @intCast(size)) orelse return GpuError.BufferMappingFailed;
                const copy = try allocator.dupe(T, mapped);
                staging.unmap();
                break :blk copy;
            },
        };
    }

    pub fn createBufferWithData(self: BufferManager, comptime T: type, data: []const T, usage: BufferUsage) !Buffer.Resource {
        const resource = try self.createBuffer(T, data.len, usage);
        var tmp = Buffer.init(resource, data.len * @sizeOf(T), usage, 0);
        self.writeBuffer(&tmp, std.mem.sliceAsBytes(data));
        return resource;
    }
};

/// GPU buffer resource with platform abstraction
pub const Buffer = struct {
    handle: GPUHandle,
    resource: Resource,
    size: usize,
    usage: BufferUsage,

    pub const Resource = union(enum) {
        mock: *MockGPU.Buffer,
        hardware: *gpu.Buffer,
    };

    pub fn init(resource: Resource, size: usize, usage: BufferUsage, id: u64) Buffer {
        return .{
            .handle = GPUHandle{ .id = id, .generation = 1 },
            .resource = resource,
            .size = size,
            .usage = usage,
        };
    }

    pub fn deinit(self: *Buffer) void {
        switch (self.resource) {
            .mock => |buf| buf.deinit(),
            .hardware => |buf| buf.deinit(),
        }
    }

    pub fn map(self: *Buffer, allocator: std.mem.Allocator) ![]u8 {
        return switch (self.resource) {
            .mock => |buf| blk: {
                const mapped_range = buf.getMappedRange(u8, 0, self.size) orelse return GpuError.BufferMappingFailed;
                break :blk try allocator.dupe(u8, mapped_range);
            },
            .hardware => return GpuError.UnsupportedBackend,
        };
    }

    pub fn unmap(self: *Buffer) void {
        switch (self.resource) {
            .mock => |buf| buf.unmap(),
            .hardware => |buf| buf.unmap(),
        }
    }

    pub fn getHardware(self: *const Buffer) ?*gpu.Buffer {
        return switch (self.resource) {
            .hardware => |buf| buf,
            else => null,
        };
    }
};

// Shader resource
