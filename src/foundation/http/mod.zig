//! Foundation HTTP helpers — leaf split (Approach-1 strangler).
//! Public surface re-exported unchanged via `foundation.http.*`.

pub const io = @import("io.zig");
pub const headers = @import("headers.zig");
pub const auth = @import("auth.zig");
pub const bind = @import("bind.zig");

pub const MAX_REQUEST_SIZE = io.MAX_REQUEST_SIZE;
pub const HttpReadResult = io.HttpReadResult;
pub const readHttpResponse = io.readHttpResponse;
pub const writeHttpAll = io.writeHttpAll;
pub const writeUnauthorized = io.writeUnauthorized;
pub const findHttpBody = io.findHttpBody;
pub const readHttpRequest = io.readHttpRequest;

pub const parseContentLength = headers.parseContentLength;
pub const requestTargetWithinBuffer = headers.requestTargetWithinBuffer;
pub const headerValue = headers.headerValue;
pub const reasonPhrase = headers.reasonPhrase;

pub const hasBearerToken = auth.hasBearerToken;

pub const bindLoopback = bind.bindLoopback;

test {
    const std = @import("std");
    _ = @import("io.zig");
    _ = @import("headers.zig");
    _ = @import("auth.zig");
    _ = @import("bind.zig");
    std.testing.refAllDecls(@This());
}
