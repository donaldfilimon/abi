//! WebAssembly entry point for Abi AI Framework
//!
//! This module provides the WASM-specific initialization and exports
//! for running the AI framework in web browsers with WebGPU support.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const root = @import("root.zig");

// WASM allocator with limited memory for web environment
var wasm_allocator: std.heap.GeneralPurposeAllocator(.{
    .thread_safe = false,
    .safety = false,
}) = undefined;

// Global application state
var app: ?*WasmApp = null;

/// WASM-specific application wrapper
const WasmApp = struct {
    allocator: std.mem.Allocator,
    gpu_renderer: ?*root.gpu_renderer.GPURenderer = null,
    neural_engine: ?*root.neural.NeuralEngine = null,
    frame_count: u64 = 0,
    last_time: f64 = 0,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .allocator = allocator,
        };

        // Initialize GPU renderer if WebGPU is enabled
        if (build_options.enable_webgpu) {
            self.gpu_renderer = try root.gpu_renderer.GPURenderer.init(allocator, .{
                .backend = .webgpu,
                .debug_validation = false,
                .power_preference = .high_performance,
            });
        }

        // Initialize neural engine for AI processing
        if (build_options.enable_simd) {
            self.neural_engine = try root.neural.NeuralEngine.init(allocator, .{
                .use_simd = true,
                .max_threads = 1, // Single-threaded for WASM
            });
        }

        return self;
    }

    pub fn deinit(self: *Self) void {
        if (self.neural_engine) |engine| {
            engine.deinit();
        }
        if (self.gpu_renderer) |renderer| {
            renderer.deinit();
        }
        self.allocator.destroy(self);
    }

    pub fn update(self: *Self, delta_time: f32) !void {
        self.frame_count += 1;

        // Update neural engine
        if (self.neural_engine) |engine| {
            try engine.update(delta_time);
        }
    }

    pub fn render(self: *Self) !void {
        if (self.gpu_renderer) |renderer| {
            try renderer.beginFrame();
            try renderer.clear(.{ .r = 0.1, .g = 0.1, .b = 0.2, .a = 1.0 });

            // Render neural network visualization
            if (self.neural_engine) |engine| {
                try renderer.renderNeuralNetwork(engine);
            }

            try renderer.endFrame();
        }
    }
};

// JavaScript interface exports

/// Initialize the WASM application
export fn wasm_init() bool {
    wasm_allocator = std.heap.GeneralPurposeAllocator(.{
        .thread_safe = false,
        .safety = false,
    }){};

    const allocator = wasm_allocator.allocator();

    app = WasmApp.init(allocator) catch |err| {
        std.log.err("Failed to initialize WASM app: {}", .{err});
        return false;
    };

    std.log.info("Abi AI Framework WASM initialized successfully", .{});
    return true;
}

/// Clean up the WASM application
export fn wasm_deinit() void {
    if (app) |a| {
        a.deinit();
        app = null;
    }
    _ = wasm_allocator.deinit();
}

/// Update frame with delta time in milliseconds
export fn wasm_update(delta_time_ms: f32) bool {
    if (app) |a| {
        const delta_time = delta_time_ms / 1000.0;
        a.update(delta_time) catch |err| {
            std.log.err("Update failed: {}", .{err});
            return false;
        };
        return true;
    }
    return false;
}

/// Render a frame
export fn wasm_render() bool {
    if (app) |a| {
        a.render() catch |err| {
            std.log.err("Render failed: {}", .{err});
            return false;
        };
        return true;
    }
    return false;
}

/// Get current frame count
export fn wasm_get_frame_count() u64 {
    if (app) |a| {
        return a.frame_count;
    }
    return 0;
}

/// Process AI text input and return processed result
export fn wasm_process_text(text_ptr: [*]const u8, text_len: usize, result_ptr: [*]u8, result_max_len: usize) usize {
    if (app == null) return 0;

    const text = text_ptr[0..text_len];
    const result_buffer = result_ptr[0..result_max_len];

    // Simple text processing example
    const processed = std.fmt.bufPrint(result_buffer, "AI processed: {s}", .{text}) catch {
        return 0;
    };

    return processed.len;
}

/// Run SIMD benchmark and return operations per second
export fn wasm_simd_benchmark() f64 {
    if (app == null) return 0.0;

    const allocator = wasm_allocator.allocator();

    // Allocate test vectors
    const size = 1024;
    const a = allocator.alloc(f32, size) catch return 0.0;
    defer allocator.free(a);
    const b = allocator.alloc(f32, size) catch return 0.0;
    defer allocator.free(b);

    // Initialize with test data
    for (a, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }
    for (b, 0..) |*val, i| {
        val.* = @floatFromInt(i * 2);
    }

    // Run benchmark
    const start_time = std.time.milliTimestamp();
    const iterations = 10000;

    var result: f32 = 0;
    for (0..iterations) |_| {
        result += root.simd_vector.dotProductSIMD(a, b);
    }

    const end_time = std.time.milliTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end_time - start_time));

    // Calculate operations per second
    const ops_per_second = (@as(f64, @floatFromInt(iterations)) * 1000.0) / duration_ms;

    // Prevent optimization of result
    std.doNotOptimizeAway(result);

    return ops_per_second;
}

/// Set neural network weights from JavaScript
export fn wasm_set_neural_weights(weights_ptr: [*]const f32, weights_len: usize) bool {
    if (app) |a| {
        if (a.neural_engine) |engine| {
            const weights = weights_ptr[0..weights_len];
            engine.setWeights(weights) catch {
                return false;
            };
            return true;
        }
    }
    return false;
}

/// Run neural network inference
export fn wasm_neural_inference(input_ptr: [*]const f32, input_len: usize, output_ptr: [*]f32, output_len: usize) bool {
    if (app) |a| {
        if (a.neural_engine) |engine| {
            const input = input_ptr[0..input_len];
            const output = output_ptr[0..output_len];
            engine.inference(input, output) catch {
                return false;
            };
            return true;
        }
    }
    return false;
}

/// WebGPU buffer creation (called from JavaScript)
export fn wasm_create_gpu_buffer(size: u32, usage: u32) u32 {
    if (app) |a| {
        if (a.gpu_renderer) |renderer| {
            const buffer_id = renderer.createBuffer(size, usage) catch {
                return 0;
            };
            return buffer_id;
        }
    }
    return 0;
}

/// Memory allocation exports for JavaScript
export fn wasm_alloc(size: usize) ?[*]u8 {
    const allocator = wasm_allocator.allocator();
    const memory = allocator.alloc(u8, size) catch return null;
    return memory.ptr;
}

export fn wasm_free(ptr: [*]u8, size: usize) void {
    const allocator = wasm_allocator.allocator();
    const memory = ptr[0..size];
    allocator.free(memory);
}

/// Error logging for JavaScript debugging
export fn wasm_log_level() u32 {
    return @intFromEnum(std.log.Level.info);
}

export fn wasm_log(level: u32, message_ptr: [*]const u8, message_len: usize) void {
    const message = message_ptr[0..message_len];
    const log_level: std.log.Level = @enumFromInt(level);

    switch (log_level) {
        .err => std.log.err("{s}", .{message}),
        .warn => std.log.warn("{s}", .{message}),
        .info => std.log.info("{s}", .{message}),
        .debug => std.log.debug("{s}", .{message}),
    }
}

// Panic handler for WASM
pub fn panic(message: []const u8, _: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    std.log.err("WASM Panic: {s}", .{message});
    @trap();
}
