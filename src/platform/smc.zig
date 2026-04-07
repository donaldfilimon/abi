//! macOS SMC (System Management Controller) Reader
//!
//! Reads fan speed, CPU/GPU die temperature, and thermal headroom
//! via IOKit's AppleSMC driver. Platform-guarded: only compiles on macOS.
//! Non-macOS targets return `error.PlatformUnsupported`.

const std = @import("std");
const builtin = @import("builtin");
const sync = @import("../foundation/sync.zig");

/// Thermal reading from SMC sensors.
pub const SmcReading = struct {
    fan_rpm: [4]?u16, // Up to 4 fans (null = not present)
    fan_count: u8,
    cpu_temp_c: f32, // CPU die temperature (°C)
    gpu_temp_c: f32, // GPU die temperature (°C)
    thermal_headroom_c: f32, // Distance to throttle threshold (°C)
};

pub const SmcError = error{
    PlatformUnsupported,
    SmcNotFound,
    SmcConnectFailed,
    SmcReadFailed,
};

/// Maximum temperature before thermal throttling (typical Apple Silicon).
const THROTTLE_THRESHOLD_C: f32 = 105.0;

/// Read current thermal and fan data from SMC.
pub fn read() SmcError!SmcReading {
    if (comptime builtin.os.tag != .macos) {
        return error.PlatformUnsupported;
    }

    return readDarwin();
}

/// Check if SMC reading is available on this platform.
pub fn isAvailable() bool {
    return builtin.os.tag == .macos;
}

// ── Darwin Implementation ──────────────────────────────────────────────

const is_darwin = builtin.os.tag == .macos;

// IOKit extern declarations — only resolved on macOS.
const kern_return_t = if (is_darwin) c_int else void;
const io_connect_t = if (is_darwin) u32 else void;
const io_object_t = if (is_darwin) u32 else void;
const mach_port_t = if (is_darwin) u32 else void;
const CFMutableDictionaryRef = if (is_darwin) *anyopaque else void;

const SMC_CMD_READ_KEYINFO: u8 = 9;
const SMC_CMD_READ_BYTES: u8 = 5;

const SmcKeyData = extern struct {
    key: u32 = 0,
    vers: [6]u8 = .{ 0, 0, 0, 0, 0, 0 },
    p_limit_data: [16]u8 = .{0} ** 16,
    key_info: SmcKeyInfoData = .{},
    result: u8 = 0,
    status: u8 = 0,
    data8: u8 = 0,
    data32: u32 = 0,
    bytes: [32]u8 = .{0} ** 32,
};

const SmcKeyInfoData = extern struct {
    data_size: u32 = 0,
    data_type: u32 = 0,
    data_attributes: u8 = 0,
};

// IOKit C functions (only linked on macOS via build/link.zig).
extern "c" fn IOServiceGetMatchingService(port: mach_port_t, matching: CFMutableDictionaryRef) io_object_t;
extern "c" fn IOServiceOpen(service: io_object_t, owningTask: mach_port_t, conn_type: u32, connect: *io_connect_t) kern_return_t;
extern "c" fn IOServiceClose(connect: io_connect_t) kern_return_t;
extern "c" fn IOServiceMatching(name: [*:0]const u8) CFMutableDictionaryRef;
extern "c" fn IOConnectCallStructMethod(
    connection: io_connect_t,
    selector: u32,
    inputStruct: *const SmcKeyData,
    inputStructCnt: usize,
    outputStruct: *SmcKeyData,
    outputStructCnt: *usize,
) kern_return_t;

// mach_task_self is a macro on macOS; use the underlying function.
extern "c" fn mach_task_self_() mach_port_t;

// Cached IOKit connection — opened once, reused across calls.
// Protected by smc_mu to prevent TOCTOU: two threads must not both see
// conn_initialized == false and both call IOServiceOpen simultaneously.
var smc_mu: sync.BlockingMutex = .{};
var cached_conn: io_connect_t = 0;
var conn_initialized: bool = false;

fn getConnection() SmcError!io_connect_t {
    if (comptime !is_darwin) return error.PlatformUnsupported;
    smc_mu.lock();
    defer smc_mu.unlock();
    if (conn_initialized) return cached_conn;

        if (IOServiceOpen(service, mach_task_self_(), 0, &cached_conn) != 0) {
            return error.SmcConnectFailed;
        }
        conn_initialized = true;
    }
    return cached_conn;
}

fn readDarwin() SmcError!SmcReading {
    if (comptime !is_darwin) return error.PlatformUnsupported;

    const conn = try getConnection();

    var result = SmcReading{
        .fan_rpm = .{ null, null, null, null },
        .fan_count = 0,
        .cpu_temp_c = 0,
        .gpu_temp_c = 0,
        .thermal_headroom_c = 0,
    };

    // Read fan count
    if (readSmcKey(conn, fourCC("FNum"))) |bytes| {
        result.fan_count = bytes[0];
    }

    // Read individual fan RPMs (F0Ac, F1Ac, ...)
    var fan_idx: u8 = 0;
    while (fan_idx < @min(result.fan_count, 4)) : (fan_idx += 1) {
        const key = fanActualKey(fan_idx);
        if (readSmcKey(conn, key)) |bytes| {
            // fpe2 format: 14-bit integer, 2-bit fraction
            const raw = (@as(u16, bytes[0]) << 8) | @as(u16, bytes[1]);
            result.fan_rpm[fan_idx] = raw >> 2;
        }
    }

    // Read CPU die temperature (TC0P or Tp09 on Apple Silicon)
    result.cpu_temp_c = readSmcTemp(conn, fourCC("TC0P")) orelse
        readSmcTemp(conn, fourCC("Tp09")) orelse 0;

    // Read GPU die temperature (TG0P or Tg05 on Apple Silicon)
    result.gpu_temp_c = readSmcTemp(conn, fourCC("TG0P")) orelse
        readSmcTemp(conn, fourCC("Tg05")) orelse 0;

    // Calculate thermal headroom
    const max_temp = @max(result.cpu_temp_c, result.gpu_temp_c);
    result.thermal_headroom_c = if (max_temp > 0) THROTTLE_THRESHOLD_C - max_temp else 0;

    return result;
}

fn readSmcKey(conn: io_connect_t, key: u32) ?[32]u8 {
    if (comptime !is_darwin) return null;

    // Step 1: Get key info
    var input = SmcKeyData{};
    var output = SmcKeyData{};
    input.key = key;
    input.data8 = SMC_CMD_READ_KEYINFO;
    var output_size: usize = @sizeOf(SmcKeyData);

    if (IOConnectCallStructMethod(conn, 2, &input, @sizeOf(SmcKeyData), &output, &output_size) != 0) {
        return null;
    }

    // Step 2: Read the value
    input.key_info.data_size = output.key_info.data_size;
    input.data8 = SMC_CMD_READ_BYTES;
    output = SmcKeyData{};
    output_size = @sizeOf(SmcKeyData);

    if (IOConnectCallStructMethod(conn, 2, &input, @sizeOf(SmcKeyData), &output, &output_size) != 0) {
        return null;
    }

    return output.bytes;
}

fn readSmcTemp(conn: io_connect_t, key: u32) ?f32 {
    const bytes = readSmcKey(conn, key) orelse return null;
    // sp78 format: signed 7.8 fixed-point
    const raw = (@as(i16, @bitCast(@as(u16, bytes[0]) << 8 | @as(u16, bytes[1]))));
    const temp = @as(f32, @floatFromInt(raw)) / 256.0;
    return if (temp > 0 and temp < 150) temp else null;
}

fn fourCC(comptime name: *const [4]u8) u32 {
    return (@as(u32, name[0]) << 24) |
        (@as(u32, name[1]) << 16) |
        (@as(u32, name[2]) << 8) |
        @as(u32, name[3]);
}

fn fanActualKey(idx: u8) u32 {
    // F0Ac, F1Ac, F2Ac, F3Ac
    const digit: u8 = '0' + idx;
    return (@as(u32, 'F') << 24) |
        (@as(u32, digit) << 16) |
        (@as(u32, 'A') << 8) |
        @as(u32, 'c');
}

// ── Tests ──────────────────────────────────────────────────────────────

test "SmcReading default values" {
    const r = SmcReading{
        .fan_rpm = .{ null, null, null, null },
        .fan_count = 0,
        .cpu_temp_c = 0,
        .gpu_temp_c = 0,
        .thermal_headroom_c = 0,
    };
    try std.testing.expectEqual(@as(u8, 0), r.fan_count);
}

test "fourCC encodes correctly" {
    try std.testing.expectEqual(@as(u32, 0x46_4e_75_6d), fourCC("FNum"));
    try std.testing.expectEqual(@as(u32, 0x54_43_30_50), fourCC("TC0P"));
}

test "fanActualKey generates correct keys" {
    try std.testing.expectEqual(fourCC("F0Ac"), fanActualKey(0));
    try std.testing.expectEqual(fourCC("F1Ac"), fanActualKey(1));
}

test "isAvailable matches platform" {
    if (builtin.os.tag == .macos) {
        try std.testing.expect(isAvailable());
    } else {
        try std.testing.expect(!isAvailable());
    }
}

test {
    std.testing.refAllDecls(@This());
}
