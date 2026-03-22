//! NUMA and CPU affinity management
//!
//! Provides NUMA node detection, CPU topology awareness, and
//! thread affinity control for optimized task execution.

const std = @import("std");

const builtin = @import("builtin");
const linux = std.os.linux;
const windows = std.os.windows;
const posix = std.posix;
const darwin = std.os.darwin;

/// macOS/Darwin thread affinity policy structures
const DarwinMach = struct {
    // Thread affinity is advisory on macOS - we use affinity tags
    // which hint the scheduler to keep threads with same tag on same core
    pub const thread_affinity_policy = extern struct {
        affinity_tag: c_int,
    };
    pub const THREAD_AFFINITY_POLICY: c_int = 4;
    pub const THREAD_AFFINITY_POLICY_COUNT: c_int = 1;

    extern "c" fn pthread_mach_thread_np(thread: std.c.pthread_t) MachPort;
    extern "c" fn thread_policy_set(
        thread: MachPort,
        flavor: c_int,
        policy_info: *const anyopaque,
        count: c_int,
    ) c_int;

    pub const MachPort = c_uint;
    pub const KERN_SUCCESS: c_int = 0;
};

const WindowsKernel32 = struct {
    extern "kernel32" fn GetCurrentThread() callconv(.winapi) windows.HANDLE;
    extern "kernel32" fn SetThreadAffinityMask(
        handle: windows.HANDLE,
        mask: usize,
    ) callconv(.winapi) usize;
    extern "kernel32" fn GetCurrentProcessorNumber() callconv(.winapi) u32;
    extern "kernel32" fn GetLastError() callconv(.winapi) windows.Win32Error;
};

pub const CpuTopology = struct {
    node_count: usize = 1,
    cpu_count: usize,
    cores_per_node: usize,
    threads_per_core: usize,
    nodes: []NumaNode,

    pub fn init(allocator: std.mem.Allocator) !CpuTopology {
        var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
        defer io_backend.deinit();
        const io = io_backend.io();

        const cpu_count = try getCoreCount(allocator, io);

        var nodes = std.ArrayListUnmanaged(NumaNode).empty;
        errdefer {
            for (nodes.items) |*node| {
                node.deinit(allocator);
            }
            nodes.deinit(allocator);
        }

        if (comptime builtin.os.tag == .linux) {
            try nodes.append(allocator, try detectLinuxNuma(allocator, io, cpu_count));
        } else if (comptime builtin.os.tag == .windows) {
            try nodes.append(allocator, try detectWindowsNuma(allocator, cpu_count));
        } else {
            try nodes.append(allocator, try detectDefaultNuma(allocator, cpu_count));
        }

        const node_count = nodes.items.len;
        const cores_per_node = cpu_count / @max(1, node_count);
        const threads_per_core = 1;

        return .{
            .node_count = node_count,
            .cpu_count = cpu_count,
            .cores_per_node = cores_per_node,
            .threads_per_core = threads_per_core,
            .nodes = try nodes.toOwnedSlice(allocator),
        };
    }

    pub fn deinit(self: *CpuTopology, allocator: std.mem.Allocator) void {
        for (self.nodes) |*node| {
            node.deinit(allocator);
        }
        allocator.free(self.nodes);
        self.* = undefined;
    }

    pub fn getNodeForCpu(self: *const CpuTopology, cpu_id: usize) ?*const NumaNode {
        for (self.nodes) |*node| {
            for (node.cpus) |cpu| {
                if (cpu == cpu_id) return node;
            }
        }
        return null;
    }
};

pub const NumaNode = struct {
    id: usize,
    cpus: []usize,
    memory_mb: usize,
    distance: []f64,

    pub fn init(allocator: std.mem.Allocator, id: usize) !NumaNode {
        _ = allocator;
        return .{
            .id = id,
            .cpus = &.{},
            .memory_mb = 0,
            .distance = &.{},
        };
    }

    pub fn deinit(self: *NumaNode, allocator: std.mem.Allocator) void {
        allocator.free(self.cpus);
        allocator.free(self.distance);
        self.* = undefined;
    }
};

pub const AffinityMask = struct {
    mask: []u8,
    size: usize,

    pub fn init(allocator: std.mem.Allocator, cpu_count: usize) !AffinityMask {
        const byte_count = (cpu_count + 7) / 8;
        const mask = try allocator.alloc(u8, byte_count);
        @memset(mask, 0);
        return .{
            .mask = mask,
            .size = cpu_count,
        };
    }

    pub fn deinit(self: *AffinityMask, allocator: std.mem.Allocator) void {
        allocator.free(self.mask);
        self.* = undefined;
    }

    pub fn set(self: *AffinityMask, cpu_id: usize) !void {
        if (cpu_id >= self.size) {
            std.log.err("CPU ID {d} exceeds maximum CPU count {d}", .{ cpu_id, self.size });
            return error.InvalidCpuId;
        }
        const byte_index = cpu_id / 8;
        const bit_index = cpu_id % 8;
        self.mask[byte_index] |= @as(u8, 1) << @intCast(bit_index);
    }

    pub fn clear(self: *AffinityMask, cpu_id: usize) !void {
        if (cpu_id >= self.size) {
            std.log.err("CPU ID {d} exceeds maximum CPU count {d}", .{ cpu_id, self.size });
            return error.InvalidCpuId;
        }
        const byte_index = cpu_id / 8;
        const bit_index = cpu_id % 8;
        self.mask[byte_index] &= ~(@as(u8, 1) << @intCast(bit_index));
    }

    pub fn isSet(self: *const AffinityMask, cpu_id: usize) bool {
        if (cpu_id >= self.size) return false;
        const byte_index = cpu_id / 8;
        const bit_index = cpu_id % 8;
        return (self.mask[byte_index] & (@as(u8, 1) << @intCast(bit_index))) != 0;
    }
};

pub fn setThreadAffinity(cpu_id: usize) !void {
    if (comptime builtin.os.tag == .linux) {
        return setLinuxThreadAffinity(cpu_id);
    } else if (comptime builtin.os.tag == .windows) {
        return setWindowsThreadAffinity(cpu_id);
    } else if (comptime builtin.os.tag == .macos) {
        return setDarwinThreadAffinity(cpu_id);
    } else {
        std.log.warn("Thread affinity not supported on platform: {t}", .{builtin.os.tag});
        // Don't error - just no-op on unsupported platforms
        return;
    }
}

pub fn setThreadAffinityMask(mask: AffinityMask) !void {
    if (comptime builtin.os.tag == .linux) {
        return setLinuxThreadAffinityMask(mask);
    } else if (comptime builtin.os.tag == .windows) {
        return setWindowsThreadAffinityMask(mask);
    } else if (comptime builtin.os.tag == .macos) {
        return setDarwinThreadAffinityMask(mask);
    } else {
        std.log.warn("Thread affinity mask not supported on platform: {t}", .{builtin.os.tag});
        // Don't error - just no-op on unsupported platforms
        return;
    }
}

pub fn getCurrentCpuId() !usize {
    if (comptime builtin.os.tag == .linux) {
        return getLinuxCurrentCpu();
    } else if (comptime builtin.os.tag == .windows) {
        return getWindowsCurrentCpu();
    } else if (comptime builtin.os.tag == .macos) {
        // macOS doesn't expose current CPU ID directly
        // Return 0 as a fallback - scheduler manages placement
        return 0;
    } else {
        std.log.warn("Getting current CPU ID not supported on platform: {t}", .{builtin.os.tag});
        return 0; // Return 0 as safe default instead of error
    }
}

fn getCoreCount(allocator: std.mem.Allocator, io: std.Io) !usize {
    if (comptime builtin.os.tag == .linux) {
        const data = try std.Io.Dir.cwd().readFileAlloc(
            io,
            "/sys/devices/system/cpu/present",
            allocator,
            .limited(64),
        );
        defer allocator.free(data);

        var max_cpu: usize = 0;
        var it = std.mem.tokenizeAny(u8, data, ",\n");
        while (it.next()) |token| {
            if (std.mem.indexOfScalar(u8, token, '-')) |dash| {
                const end_str = token[dash + 1 ..];
                const end_val = try std.fmt.parseInt(usize, end_str, 10);
                if (end_val > max_cpu) {
                    max_cpu = end_val;
                }
            } else {
                const value = try std.fmt.parseInt(usize, token, 10);
                if (value > max_cpu) {
                    max_cpu = value;
                }
            }
        }
        return max_cpu + 1;
    } else if (comptime builtin.os.tag == .windows) {
        return std.Thread.getCpuCount();
    } else if (comptime builtin.os.tag == .freestanding or
        builtin.cpu.arch == .wasm32 or
        builtin.cpu.arch == .wasm64)
    {
        return 1; // WASM/freestanding doesn't support threading
    } else {
        return std.Thread.getCpuCount();
    }
}

fn detectLinuxNuma(allocator: std.mem.Allocator, io: std.Io, cpu_count: usize) !NumaNode {
    var node = try NumaNode.init(allocator, 0);
    errdefer node.deinit(allocator);

    var cpus = std.ArrayListUnmanaged(usize).empty;
    errdefer cpus.deinit(allocator);

    var i: usize = 0;
    while (i < cpu_count) : (i += 1) {
        const cpu_path = try std.fmt.allocPrint(
            allocator,
            "/sys/devices/system/cpu/cpu{d}/topology/physical_package_id",
            .{i},
        );
        defer allocator.free(cpu_path);

        // Use root directory for absolute paths on Linux
        var root_dir = std.Io.Dir.cwd().openDir(io, "/", .{}) catch continue;
        defer root_dir.close(io);
        // cpu_path starts with /, so strip it for relative access from root
        const rel_path = cpu_path[1..];
        if (root_dir.openFile(io, rel_path, .{})) |file| {
            file.close(io);
            try cpus.append(allocator, i);
        } else |_| {}
    }

    node.cpus = try cpus.toOwnedSlice(allocator);
    node.memory_mb = 16384;
    node.distance = try allocator.alloc(f64, 1);
    node.distance[0] = 0.0;

    return node;
}

fn detectWindowsNuma(allocator: std.mem.Allocator, cpu_count: usize) !NumaNode {
    var node = try NumaNode.init(allocator, 0);
    errdefer node.deinit(allocator);

    var cpus = try allocator.alloc(usize, cpu_count);
    for (cpus, 0..) |_, i| {
        cpus[i] = i;
    }
    node.cpus = cpus;
    node.memory_mb = 16384;
    node.distance = try allocator.alloc(f64, 1);
    node.distance[0] = 0.0;

    return node;
}

fn detectDefaultNuma(allocator: std.mem.Allocator, cpu_count: usize) !NumaNode {
    var node = try NumaNode.init(allocator, 0);
    errdefer node.deinit(allocator);

    var cpus = try allocator.alloc(usize, cpu_count);
    for (cpus, 0..) |_, i| {
        cpus[i] = i;
    }
    node.cpus = cpus;
    node.memory_mb = 8192;
    node.distance = try allocator.alloc(f64, 1);
    node.distance[0] = 0.0;

    return node;
}

fn setLinuxThreadAffinity(cpu_id: usize) !void {
    const max_cpus = linux.CPU_SETSIZE * 8;
    if (cpu_id >= max_cpus) {
        return error.InvalidCpuId;
    }

    var set = std.mem.zeroes(linux.cpu_set_t);
    const bits_per = @bitSizeOf(usize);
    const index = cpu_id / bits_per;
    const bit = cpu_id % bits_per;
    set[index] |= @as(usize, 1) << @intCast(bit);

    try linux.sched_setaffinity(0, &set);
}

fn setWindowsThreadAffinity(cpu_id: usize) !void {
    const bits_per = @bitSizeOf(usize);
    if (cpu_id >= bits_per) {
        return error.InvalidCpuId;
    }

    const handle = WindowsKernel32.GetCurrentThread();
    const mask = @as(usize, 1) << @intCast(cpu_id);
    if (WindowsKernel32.SetThreadAffinityMask(handle, mask) == 0) {
        return windows.unexpectedError(WindowsKernel32.GetLastError());
    }
}

fn setLinuxThreadAffinityMask(mask: AffinityMask) !void {
    var set = std.mem.zeroes(linux.cpu_set_t);
    const bits_per = @bitSizeOf(usize);
    const max_cpus = linux.CPU_SETSIZE * 8;

    var cpu_id: usize = 0;
    while (cpu_id < mask.size and cpu_id < max_cpus) {
        if (mask.isSet(cpu_id)) {
            const index = cpu_id / bits_per;
            const bit = cpu_id % bits_per;
            set[index] |= @as(usize, 1) << @intCast(bit);
        }
        cpu_id += 1;
    }

    try linux.sched_setaffinity(0, &set);
}

fn setWindowsThreadAffinityMask(mask: AffinityMask) !void {
    const bits_per = @bitSizeOf(usize);
    if (mask.size > bits_per) {
        return error.UnsupportedPlatform;
    }

    var bitmask: usize = 0;
    var cpu_id: usize = 0;
    while (cpu_id < mask.size) {
        if (mask.isSet(cpu_id)) {
            bitmask |= @as(usize, 1) << @intCast(cpu_id);
        }
        cpu_id += 1;
    }

    const handle = WindowsKernel32.GetCurrentThread();
    if (WindowsKernel32.SetThreadAffinityMask(handle, bitmask) == 0) {
        return windows.unexpectedError(WindowsKernel32.GetLastError());
    }
}

fn getLinuxCurrentCpu() !usize {
    var cpu: usize = 0;
    const rc = linux.getcpu(&cpu, null);
    if (@as(isize, @bitCast(rc)) < 0) {
        return posix.unexpectedErrno(linux.errno(rc));
    }
    return cpu;
}

fn getWindowsCurrentCpu() !usize {
    return @intCast(WindowsKernel32.GetCurrentProcessorNumber());
}

/// macOS/Darwin thread affinity using Mach thread affinity policies.
/// Note: macOS affinity is advisory - the scheduler uses it as a hint,
/// not a hard constraint like Linux/Windows.
fn setDarwinThreadAffinity(cpu_id: usize) !void {
    // Get Mach thread port for current thread
    const thread = DarwinMach.pthread_mach_thread_np(std.c.pthread_self());

    // Set affinity policy with cpu_id as affinity tag
    // Threads with same tag are scheduled on same core when possible
    const policy = DarwinMach.thread_affinity_policy{
        .affinity_tag = @intCast(cpu_id),
    };

    const result = DarwinMach.thread_policy_set(
        thread,
        DarwinMach.THREAD_AFFINITY_POLICY,
        @ptrCast(&policy),
        DarwinMach.THREAD_AFFINITY_POLICY_COUNT,
    );

    if (result != DarwinMach.KERN_SUCCESS) {
        std.log.warn("macOS thread affinity hint failed (advisory only): {d}", .{result});
        // Don't return error - affinity is advisory on macOS
    }
}

fn setDarwinThreadAffinityMask(mask: AffinityMask) !void {
    // Find first set CPU in mask and use that as affinity tag
    var cpu_id: usize = 0;
    while (cpu_id < mask.size) : (cpu_id += 1) {
        if (mask.isSet(cpu_id)) {
            return setDarwinThreadAffinity(cpu_id);
        }
    }
    // No CPUs set in mask - no-op
}

test "affinity mask operations" {
    const allocator = std.testing.allocator;

    var mask = try AffinityMask.init(allocator, 16);
    defer mask.deinit(allocator);

    try mask.set(3);
    try mask.set(7);
    try mask.set(15);

    try std.testing.expect(mask.isSet(3));
    try std.testing.expect(mask.isSet(7));
    try std.testing.expect(mask.isSet(15));
    try std.testing.expect(!mask.isSet(0));

    try mask.clear(7);
    try std.testing.expect(!mask.isSet(7));
    try std.testing.expect(mask.isSet(3));
}

test "cpu topology detection" {
    const allocator = std.testing.allocator;

    var topology = try CpuTopology.init(allocator);
    defer topology.deinit(allocator);

    try std.testing.expect(topology.node_count >= 1);
    try std.testing.expect(topology.cpu_count >= 1);
    try std.testing.expect(topology.cores_per_node >= 1);
}

test {
    std.testing.refAllDecls(@This());
}
