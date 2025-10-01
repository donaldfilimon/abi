pub const ErrorCode = enum(u16) {
    ok = 0,
    invalid_request = 400,
    unavailable = 503,
};

pub const Result = struct {
    code: ErrorCode = .ok,
    message: []const u8 = "",
};
