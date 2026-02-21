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
