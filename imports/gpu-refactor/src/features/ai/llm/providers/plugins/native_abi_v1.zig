pub const ABI_VERSION: u32 = 1;

pub const GenerateRequest = extern struct {
    model: [*:0]const u8,
    prompt: [*:0]const u8,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repetition_penalty: f32,
};

pub const GenerateResponse = extern struct {
    status: i32 = -1,
    text_ptr: [*]const u8 = undefined,
    text_len: usize = 0,
    model_ptr: [*]const u8 = undefined,
    model_len: usize = 0,
    error_ptr: ?[*]const u8 = null,
    error_len: usize = 0,
    release: ?*const fn (*GenerateResponse) callconv(.c) void = null,
};

pub const PluginV1 = extern struct {
    abi_version: u32,
    name: [*:0]const u8,
    is_available: ?*const fn () callconv(.c) bool = null,
    generate: *const fn (*const GenerateRequest, *GenerateResponse) callconv(.c) i32,
};

pub const GetPluginFn = *const fn () callconv(.c) *const PluginV1;

pub const Status = enum(i32) {
    ok = 0,
    not_available = 1,
    invalid_request = 2,
    failed = 3,
};

pub fn statusFromCode(code: i32) Status {
    return switch (code) {
        @intFromEnum(Status.ok) => .ok,
        @intFromEnum(Status.not_available) => .not_available,
        @intFromEnum(Status.invalid_request) => .invalid_request,
        @intFromEnum(Status.failed) => .failed,
        else => .failed,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "ABI_VERSION is 1" {
    try @import("std").testing.expectEqual(@as(u32, 1), ABI_VERSION);
}

test "statusFromCode maps known codes" {
    const testing = @import("std").testing;
    try testing.expectEqual(Status.ok, statusFromCode(0));
    try testing.expectEqual(Status.not_available, statusFromCode(1));
    try testing.expectEqual(Status.invalid_request, statusFromCode(2));
    try testing.expectEqual(Status.failed, statusFromCode(3));
}

test "statusFromCode maps unknown codes to failed" {
    const testing = @import("std").testing;
    try testing.expectEqual(Status.failed, statusFromCode(-1));
    try testing.expectEqual(Status.failed, statusFromCode(99));
    try testing.expectEqual(Status.failed, statusFromCode(4));
}

test "GenerateRequest struct layout" {
    const testing = @import("std").testing;
    // Verify the struct is extern and has expected fields
    const info = @typeInfo(GenerateRequest);
    try testing.expect(info == .@"struct");
    try testing.expect(info.@"struct".layout == .@"extern");
    try testing.expectEqual(@as(usize, 7), info.@"struct".fields.len);
}

test "GenerateResponse default values" {
    const resp = GenerateResponse{};
    const testing = @import("std").testing;
    try testing.expectEqual(@as(i32, -1), resp.status);
    try testing.expectEqual(@as(usize, 0), resp.text_len);
    try testing.expectEqual(@as(usize, 0), resp.model_len);
    try testing.expect(resp.error_ptr == null);
    try testing.expectEqual(@as(usize, 0), resp.error_len);
    try testing.expect(resp.release == null);
}

test "PluginV1 struct layout" {
    const testing = @import("std").testing;
    const info = @typeInfo(PluginV1);
    try testing.expect(info == .@"struct");
    try testing.expect(info.@"struct".layout == .@"extern");
    try testing.expectEqual(@as(usize, 4), info.@"struct".fields.len);
}

test "Status enum values" {
    const testing = @import("std").testing;
    try testing.expectEqual(@as(i32, 0), @intFromEnum(Status.ok));
    try testing.expectEqual(@as(i32, 1), @intFromEnum(Status.not_available));
    try testing.expectEqual(@as(i32, 2), @intFromEnum(Status.invalid_request));
    try testing.expectEqual(@as(i32, 3), @intFromEnum(Status.failed));
}

test {
    @import("std").testing.refAllDecls(@This());
}
