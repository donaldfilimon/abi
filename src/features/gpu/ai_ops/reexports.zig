//! Low-level GPU Module Re-exports for AI Modules
//!
//! These re-exports provide AI modules with access to low-level GPU primitives
//! while centralizing the compile-time gating in one place. When GPU is disabled,
//! stub types are provided that return error.NotAvailable.

const std = @import("std");
const build_options = @import("build_options");
const backend_shared = @import("../backends/shared.zig");

/// GPU backend availability check.
pub const gpu_enabled = build_options.feat_gpu;

/// Device memory management re-exports.
/// Provides DeviceMemory struct with init/deinit and memcpy functions.
pub const memory = if (build_options.feat_gpu and build_options.gpu_fpga)
    // FPGA memory interface would go here
    struct {
        // ... (rest of FPGA stub)
    }
else if (build_options.feat_gpu and build_options.gpu_cuda and backend_shared.dynlibSupported)
    @import("../backends/cuda/memory.zig")
else if (build_options.feat_gpu and @import("builtin").os.tag == .macos)
    struct {
        // Simplified memory for unified Apple Silicon
        pub fn init(_: std.mem.Allocator) !void {}
        pub fn deinit() void {}
        pub const DeviceMemory = struct {
            ptr: ?*anyopaque,
            size: usize,
            allocator: std.mem.Allocator,
            pub fn init(allocator: std.mem.Allocator, size: usize) !@This() {
                const slice = try allocator.alloc(u8, size);
                return @This(){ .ptr = slice.ptr, .size = size, .allocator = allocator };
            }
            pub fn deinit(self: *@This()) void {
                const slice = @as([*]u8, @ptrCast(@alignCast(self.ptr)))[0..self.size];
                self.allocator.free(slice);
            }
        };
        pub fn memcpyHostToDevice(dst: *anyopaque, src: *const anyopaque, size: usize) !void {
            @memcpy(@as([*]u8, @ptrCast(@alignCast(dst)))[0..size], @as([*]const u8, @ptrCast(@alignCast(src)))[0..size]);
        }
        pub fn memcpyDeviceToHost(dst: *anyopaque, src: *const anyopaque, size: usize) !void {
            @memcpy(@as([*]u8, @ptrCast(@alignCast(dst)))[0..size], @as([*]const u8, @ptrCast(@alignCast(src)))[0..size]);
        }
    }
else
    struct {
        pub fn init(_: std.mem.Allocator) !void {
            return error.NotAvailable;
        }

        pub fn deinit() void {}

        pub const DeviceMemory = struct {
            ptr: ?*anyopaque,
            size: usize,
            allocator: std.mem.Allocator,

            pub fn init(_: std.mem.Allocator, _: usize) !@This() {
                return error.NotAvailable;
            }

            pub fn deinit(_: *@This()) void {}
        };

        pub fn memcpyHostToDevice(_: *anyopaque, _: *const anyopaque, _: usize) !void {
            return error.NotAvailable;
        }

        pub fn memcpyDeviceToHost(_: *anyopaque, _: *anyopaque, _: usize) !void {
            return error.NotAvailable;
        }
    };

/// LLM kernel operations re-exports.
/// Provides LlmKernelModule with softmax, rmsnorm, silu, gelu, scale, etc.
pub const llm_kernels = if (build_options.feat_gpu and build_options.gpu_cuda and backend_shared.dynlibSupported)
    @import("../backends/cuda/llm_kernels.zig")
else
    struct {
        pub fn isAvailable() bool {
            return false;
        }

        pub const LlmKernelModule = struct {
            pub fn init(_: std.mem.Allocator) !@This() {
                return error.NotAvailable;
            }

            pub fn deinit(_: *@This()) void {}

            pub fn softmax(_: *@This(), _: u64, _: u32, _: ?*anyopaque) !void {
                return error.NotAvailable;
            }

            pub fn rmsnorm(_: *@This(), _: u64, _: u64, _: u32, _: f32, _: ?*anyopaque) !void {
                return error.NotAvailable;
            }

            pub fn silu(_: *@This(), _: u64, _: u32, _: ?*anyopaque) !void {
                return error.NotAvailable;
            }

            pub fn gelu(_: *@This(), _: u64, _: u32, _: ?*anyopaque) !void {
                return error.NotAvailable;
            }

            pub fn elementwiseMul(_: *@This(), _: u64, _: u64, _: u32, _: ?*anyopaque) !void {
                return error.NotAvailable;
            }

            pub fn elementwiseAdd(_: *@This(), _: u64, _: u64, _: u32, _: ?*anyopaque) !void {
                return error.NotAvailable;
            }

            pub fn scale(_: *@This(), _: u64, _: f32, _: u32, _: ?*anyopaque) !void {
                return error.NotAvailable;
            }
        };
    };

/// cuBLAS operations re-exports.
/// Provides CublasContext with sgemm, sgemmStridedBatched, and matmulRowMajor.
pub const cublas = if (build_options.feat_gpu and build_options.gpu_cuda and backend_shared.dynlibSupported)
    @import("../backends/cuda/cublas.zig")
else
    struct {
        pub fn isAvailable() bool {
            return false;
        }

        pub const CublasOperation = enum { no_trans, trans };

        pub const CublasContext = struct {
            pub fn init() !@This() {
                return error.NotAvailable;
            }

            pub fn deinit(_: *@This()) void {}

            pub fn sgemm(
                _: *@This(),
                _: CublasOperation,
                _: CublasOperation,
                _: i32,
                _: i32,
                _: i32,
                _: f32,
                _: *const anyopaque,
                _: i32,
                _: *const anyopaque,
                _: i32,
                _: f32,
                _: *anyopaque,
                _: i32,
            ) !void {
                return error.NotAvailable;
            }

            pub fn sgemmStridedBatched(
                _: *@This(),
                _: CublasOperation,
                _: CublasOperation,
                _: i32,
                _: i32,
                _: i32,
                _: f32,
                _: *const anyopaque,
                _: i32,
                _: i64,
                _: *const anyopaque,
                _: i32,
                _: i64,
                _: f32,
                _: *anyopaque,
                _: i32,
                _: i64,
                _: i32,
            ) !void {
                return error.NotAvailable;
            }
        };

        pub fn matmulRowMajor(
            _: *CublasContext,
            _: *const anyopaque,
            _: *const anyopaque,
            _: *anyopaque,
            _: i32,
            _: i32,
            _: i32,
        ) !void {
            return error.NotAvailable;
        }
    };

/// GPU backend summary for availability detection.
pub const backend = if (build_options.feat_gpu)
    @import("../backend.zig")
else
    struct {
        // ...
    };

pub const MacosAiOps = if (@import("builtin").os.tag == .macos)
    @import("macos.zig").MacosAiOps
else
    struct {
        pub fn init(_: std.mem.Allocator) !void {
            return error.NotAvailable;
        }
    };
