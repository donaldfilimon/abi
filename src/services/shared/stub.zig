//! Shared Utilities Stub Module
//!
//! Stub implementation when shared utilities are disabled.
//! Provides API compatibility with mod.zig while returning SharedDisabled for operations.

const std = @import("std");

// ============================================================================
// Error Types
// ============================================================================

pub const SharedError = error{
    SharedDisabled,
};

// ============================================================================
// Errors Module Stub
// ============================================================================

pub const errors = struct {
    pub const AbiError = error{
        SharedDisabled,
        Unknown,
    };
};

// ============================================================================
// Logging Module Stub
// ============================================================================

pub const logging = struct {
    pub const Logger = struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator, _: []const u8) Self {
            return .{};
        }
        pub fn deinit(_: *Self) void {}
        pub fn debug(_: *Self, _: []const u8, _: anytype) void {}
        pub fn info(_: *Self, _: []const u8, _: anytype) void {}
        pub fn warn(_: *Self, _: []const u8, _: anytype) void {}
        pub fn err(_: *Self, _: []const u8, _: anytype) void {}
    };

    pub fn log(_: []const u8, _: anytype) void {}
};

pub const log = logging.log;
pub const Logger = logging.Logger;

// ============================================================================
// Plugins Module Stub
// ============================================================================

pub const plugins = struct {
    pub const PluginInfo = struct {
        name: []const u8 = "",
        version: []const u8 = "",
    };

    pub const PluginRegistry = struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator) Self {
            return .{};
        }
        pub fn deinit(_: *Self) void {}
        pub fn register(_: *Self, _: PluginInfo) SharedError!void {
            return error.SharedDisabled;
        }
        pub fn get(_: *Self, _: []const u8) ?PluginInfo {
            return null;
        }
    };
};

// ============================================================================
// SIMD Module Stub
// ============================================================================

pub const simd = struct {
    pub fn vectorAdd(_: []const f32, _: []const f32, _: []f32) void {}
    pub fn vectorDot(_: []const f32, _: []const f32) f32 {
        return 0.0;
    }
    pub fn vectorL2Norm(_: []const f32) f32 {
        return 0.0;
    }
    pub fn cosineSimilarity(_: []const f32, _: []const f32) f32 {
        return 0.0;
    }
    pub fn hasSimdSupport() bool {
        return false;
    }
};

// Re-export SIMD functions
pub const vectorAdd = simd.vectorAdd;
pub const vectorDot = simd.vectorDot;
pub const vectorL2Norm = simd.vectorL2Norm;
pub const cosineSimilarity = simd.cosineSimilarity;
pub const hasSimdSupport = simd.hasSimdSupport;

// ============================================================================
// Utils Module Stub
// ============================================================================

pub const utils = struct {
    pub const SimpleModuleLifecycle = struct {
        const Self = @This();
        initialized: bool = false,

        pub fn init(_: *Self) void {}
        pub fn deinit(_: *Self) void {}
        pub fn isInitialized(self: *const Self) bool {
            return self.initialized;
        }
    };

    pub const LifecycleError = error{
        AlreadyInitialized,
        NotInitialized,
        SharedDisabled,
    };
};

pub const SimpleModuleLifecycle = utils.SimpleModuleLifecycle;
pub const LifecycleError = utils.LifecycleError;

// ============================================================================
// Matrix Module Stub (v2)
// ============================================================================

pub const matrix = struct {
    pub fn Matrix(comptime T: type) type {
        _ = T;
        return struct {
            const Self = @This();
            pub fn alloc(_: std.mem.Allocator, _: usize, _: usize) !Self {
                return error.SharedDisabled;
            }
            pub fn free(_: *Self, _: std.mem.Allocator) void {}
            pub fn rows(_: Self) usize {
                return 0;
            }
            pub fn cols(_: Self) usize {
                return 0;
            }
        };
    }
    pub const Mat32 = Matrix(f32);
    pub const Mat64 = Matrix(f64);
};

// ============================================================================
// Tensor Module Stub (v2)
// ============================================================================

pub const tensor = struct {
    pub const max_rank = 8;
    pub const Shape = struct {
        dims: [max_rank]usize = .{0} ** max_rank,
        rank: usize = 0,
        pub fn init(dims: []const usize) Shape {
            var s = Shape{};
            for (dims, 0..) |d, i| {
                if (i >= max_rank) break;
                s.dims[i] = d;
            }
            s.rank = @min(dims.len, max_rank);
            return s;
        }
        pub fn totalElements(self: Shape) usize {
            if (self.rank == 0) return 0;
            var total: usize = 1;
            for (self.dims[0..self.rank]) |d| total *= d;
            return total;
        }
    };
    pub fn Tensor(comptime T: type) type {
        _ = T;
        return struct {
            const Self = @This();
            pub fn alloc(_: std.mem.Allocator, _: Shape) !Self {
                return error.SharedDisabled;
            }
            pub fn free(_: *Self, _: std.mem.Allocator) void {}
            pub fn shape(_: Self) Shape {
                return .{};
            }
        };
    }
    pub const Tensor32 = Tensor(f32);
    pub const Tensor64 = Tensor(f64);
};

// ============================================================================
// OS Module Stub
// ============================================================================

pub const os = struct {
    pub fn getEnv(_: []const u8) ?[]const u8 {
        return null;
    }
    pub fn getCwd(_: std.mem.Allocator) SharedError![]const u8 {
        return error.SharedDisabled;
    }
};

// ============================================================================
// Time Module Stub
// ============================================================================

pub const time = struct {
    pub fn now() i64 {
        return 0;
    }
    pub fn sleep(_: u64) void {}
    pub fn formatTimestamp(_: i64, _: []u8) []const u8 {
        return "";
    }
};

// ============================================================================
// IO Module Stub
// ============================================================================

pub const io = struct {
    pub fn readFile(_: std.mem.Allocator, _: []const u8) SharedError![]const u8 {
        return error.SharedDisabled;
    }
    pub fn writeFile(_: []const u8, _: []const u8) SharedError!void {
        return error.SharedDisabled;
    }
    pub fn fileExists(_: []const u8) bool {
        return false;
    }
};

// ============================================================================
// Stub Common Module Stub
// ============================================================================

pub const stub_common = struct {
    pub fn notImplemented() SharedError!void {
        return error.SharedDisabled;
    }
};

// ============================================================================
// Security Module Stub
// ============================================================================

pub const security = struct {
    pub const TlsConfig = struct {
        cert_path: ?[]const u8 = null,
        key_path: ?[]const u8 = null,
    };

    pub const ApiKey = struct {
        const Self = @This();
        key: []const u8 = "",

        pub fn validate(_: *const Self) bool {
            return false;
        }
    };

    pub const RbacRole = struct {
        name: []const u8 = "",
        permissions: []const []const u8 = &.{},
    };
};

// ============================================================================
// Utils Sub-modules Stubs
// ============================================================================

pub const memory = struct {
    pub const MemoryPool = struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator, _: usize) Self {
            return .{};
        }
        pub fn deinit(_: *Self) void {}
        pub fn alloc(_: *Self, _: usize) SharedError![]u8 {
            return error.SharedDisabled;
        }
        pub fn free(_: *Self, _: []u8) void {}
    };
};

pub const crypto = struct {
    pub fn hash(_: []const u8) [32]u8 {
        return [_]u8{0} ** 32;
    }
    pub fn hmac(_: []const u8, _: []const u8) [32]u8 {
        return [_]u8{0} ** 32;
    }
    pub fn randomBytes(_: []u8) void {}
};

pub const encoding = struct {
    pub fn base64Encode(_: std.mem.Allocator, _: []const u8) SharedError![]const u8 {
        return error.SharedDisabled;
    }
    pub fn base64Decode(_: std.mem.Allocator, _: []const u8) SharedError![]const u8 {
        return error.SharedDisabled;
    }
};

pub const fs = struct {
    pub fn readDir(_: std.mem.Allocator, _: []const u8) SharedError![]const []const u8 {
        return error.SharedDisabled;
    }
    pub fn createDir(_: []const u8) SharedError!void {
        return error.SharedDisabled;
    }
    pub fn removeFile(_: []const u8) SharedError!void {
        return error.SharedDisabled;
    }
};

pub const http = struct {
    pub const Request = struct {
        method: []const u8 = "GET",
        url: []const u8 = "",
        headers: []const [2][]const u8 = &.{},
        body: ?[]const u8 = null,
    };

    pub const Response = struct {
        status: u16 = 0,
        body: []const u8 = "",
    };

    pub fn request(_: std.mem.Allocator, _: Request) SharedError!Response {
        return error.SharedDisabled;
    }
};

pub const json = struct {
    pub fn parse(_: std.mem.Allocator, _: []const u8) SharedError!std.json.Value {
        return error.SharedDisabled;
    }
    pub fn stringify(_: std.mem.Allocator, _: anytype) SharedError![]const u8 {
        return error.SharedDisabled;
    }
};

pub const net = struct {
    pub const Address = struct {
        host: []const u8 = "",
        port: u16 = 0,
    };

    pub fn resolve(_: []const u8) SharedError!Address {
        return error.SharedDisabled;
    }
};

// ============================================================================
// Legacy Module Stub
// ============================================================================

pub const legacy = struct {
    pub const CoreUtils = struct {
        const Self = @This();

        pub fn init(_: std.mem.Allocator) Self {
            return .{};
        }
        pub fn deinit(_: *Self) void {}
    };
};

// ============================================================================
// Module Lifecycle
// ============================================================================

var initialized: bool = false;

pub fn init(_: std.mem.Allocator) SharedError!void {
    return error.SharedDisabled;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return initialized;
}
