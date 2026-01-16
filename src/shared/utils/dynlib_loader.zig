//! Dynamic Library Loading Utilities
//!
//! Provides cross-platform dynamic library loading with automatic
//! fallback support and function pointer resolution.
//!
//! Usage:
//! ```zig
//! const loader = try DynLibLoader.init(&.{"libcuda.so", "nvcuda.dll"});
//! defer loader.deinit();
//!
//! const cuInit = loader.lookup(CuInitFn, "cuInit") orelse return error.SymbolNotFound;
//! ```

const std = @import("std");
const builtin = @import("builtin");

pub const DynLibError = error{
    LibraryNotFound,
    SymbolNotFound,
    InvalidLibrary,
    PlatformNotSupported,
};

/// Dynamic library loader with multi-name fallback support
pub const DynLibLoader = struct {
    lib: std.DynLib,
    lib_name: []const u8,

    /// Initialize by trying library names in order until one loads
    pub fn init(lib_names: []const []const u8) DynLibError!DynLibLoader {
        for (lib_names) |name| {
            if (std.DynLib.open(name)) |lib| {
                return .{
                    .lib = lib,
                    .lib_name = name,
                };
            } else |_| {
                continue;
            }
        }
        return DynLibError.LibraryNotFound;
    }

    /// Initialize with platform-specific library names
    pub fn initPlatform(config: PlatformLibConfig) DynLibError!DynLibLoader {
        const names: []const []const u8 = switch (builtin.os.tag) {
            .windows => config.windows,
            .linux => config.linux,
            .macos => config.macos,
            else => return DynLibError.PlatformNotSupported,
        };
        return init(names);
    }

    pub fn deinit(self: *DynLibLoader) void {
        self.lib.close();
        self.* = undefined;
    }

    /// Look up a function symbol, returns null if not found
    pub fn lookup(self: *const DynLibLoader, comptime Fn: type, name: [:0]const u8) ?Fn {
        return self.lib.lookup(Fn, name);
    }

    /// Look up a function symbol, returns error if not found
    pub fn lookupRequired(self: *const DynLibLoader, comptime Fn: type, name: [:0]const u8) DynLibError!Fn {
        return self.lib.lookup(Fn, name) orelse DynLibError.SymbolNotFound;
    }

    /// Load multiple functions at once
    pub fn loadFunctions(
        self: *const DynLibLoader,
        comptime FunctionSet: type,
    ) DynLibError!FunctionSet {
        var result: FunctionSet = undefined;
        inline for (std.meta.fields(FunctionSet)) |field| {
            const fn_ptr = self.lib.lookup(field.type, field.name ++ "");
            if (fn_ptr == null and !isOptionalField(field)) {
                return DynLibError.SymbolNotFound;
            }
            @field(result, field.name) = fn_ptr;
        }
        return result;
    }

    fn isOptionalField(field: std.builtin.Type.StructField) bool {
        // Fields starting with "opt_" are considered optional
        return std.mem.startsWith(u8, field.name, "opt_");
    }

    /// Get the name of the loaded library
    pub fn getLibraryName(self: *const DynLibLoader) []const u8 {
        return self.lib_name;
    }
};

/// Platform-specific library name configuration
pub const PlatformLibConfig = struct {
    windows: []const []const u8 = &.{},
    linux: []const []const u8 = &.{},
    macos: []const []const u8 = &.{},
};

/// Common library configurations
pub const CommonLibs = struct {
    pub const cuda = PlatformLibConfig{
        .windows = &.{"nvcuda.dll"},
        .linux = &.{ "libcuda.so.1", "libcuda.so" },
    };

    pub const nvrtc = PlatformLibConfig{
        .windows = &.{ "nvrtc64_120.dll", "nvrtc64_112.dll", "nvrtc64.dll" },
        .linux = &.{ "libnvrtc.so.12", "libnvrtc.so.11", "libnvrtc.so" },
    };

    pub const vulkan = PlatformLibConfig{
        .windows = &.{"vulkan-1.dll"},
        .linux = &.{ "libvulkan.so.1", "libvulkan.so" },
        .macos = &.{"libvulkan.dylib"},
    };

    pub const opencl = PlatformLibConfig{
        .windows = &.{"OpenCL.dll"},
        .linux = &.{ "libOpenCL.so.1", "libOpenCL.so" },
        .macos = &.{ "/System/Library/Frameworks/OpenCL.framework/OpenCL", "libOpenCL.dylib" },
    };
};

/// Function specification for batch loading
pub const FunctionSpec = struct {
    name: [:0]const u8,
    required: bool = true,
};

/// Batch load functions into a struct
pub fn loadFunctionBatch(
    comptime Specs: []const FunctionSpec,
    comptime FnTypes: type,
    loader: *const DynLibLoader,
) DynLibError!FnTypes {
    var result: FnTypes = undefined;
    inline for (Specs, 0..) |spec, i| {
        const fields = std.meta.fields(FnTypes);
        const field = fields[i];
        const fn_ptr = loader.lookup(field.type, spec.name);
        if (fn_ptr == null and spec.required) {
            return DynLibError.SymbolNotFound;
        }
        @field(result, field.name) = fn_ptr;
    }
    return result;
}

test "dynlib loader basic" {
    // Test with a library that likely exists
    const loader = DynLibLoader.init(&.{"kernel32.dll"}) catch |err| {
        // Skip test if library not available (non-Windows)
        if (err == DynLibError.LibraryNotFound) return;
        return err;
    };
    defer @constCast(&loader).deinit();

    // Verify we can look up a common function
    const GetLastError = loader.lookup(*const fn () callconv(.winapi) u32, "GetLastError");
    try std.testing.expect(GetLastError != null);
}

test "platform config" {
    const config = CommonLibs.cuda;
    try std.testing.expect(config.windows.len > 0);
    try std.testing.expect(config.linux.len > 0);
}
