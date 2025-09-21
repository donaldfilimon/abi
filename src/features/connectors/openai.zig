//! REST connector for OpenAI embeddings covering HTTP wiring, error mapping,
//! and response parsing shared across the ABI stack.

const std = @import("std");

pub const Allocator = std.mem.Allocator;

/// Errors that can occur during OpenAI API operations
pub const Error = error{
    MissingApiKey,
    NetworkError,
    InvalidResponse,
    OutOfMemory,
    ConnectionRefused,
    ConnectionTimedOut,
    ConnectionResetByPeer,
    NetworkUnreachable,
    UnknownHostName,
    TlsInitializationFailed,
    UnsupportedUriScheme,
    UriHostTooLong,
    CertificateBundleLoadFailure,
    UnexpectedConnectFailure,
    NameServerFailure,
    TemporaryNameServerFailure,
    HostLacksNetworkAddresses,
    UriMissingHost,
    NoSpaceLeft,
    WriteFailed,
    ReadFailed,
    HttpChunkInvalid,
    HttpChunkTruncated,
    HttpHeadersInvalid,
    HttpHeadersOversize,
    HttpRequestTruncated,
    HttpConnectionClosing,
    TooManyHttpRedirects,
    RedirectRequiresResend,
    HttpRedirectLocationMissing,
    HttpRedirectLocationOversize,
    HttpRedirectLocationInvalid,
    HttpContentEncodingUnsupported,
    Overflow,
    InvalidEnumTag,
    InvalidCharacter,
    UnexpectedToken,
    InvalidNumber,
    DuplicateField,
    UnknownField,
    MissingField,
    LengthMismatch,
    SyntaxError,
    UnexpectedEndOfInput,
    BufferUnderrun,
    ValueTooLong,
};

/// Embeds the given text using the OpenAI embeddings API
///
/// Args:
/// - allocator: Memory allocator for dynamic allocations
/// - base_url: Base URL for the OpenAI API (e.g., "https://api.openai.com/v1")
/// - api_key: OpenAI API key for authentication
/// - model: The model to use for embeddings (e.g., "text-embedding-ada-002")
/// - text: The text to embed
///
/// Returns:
/// - A slice of f32 values representing the embedding vector
/// - The caller owns the returned memory and must free it
///
/// Errors:
/// - MissingApiKey: If the api_key is empty
/// - NetworkError: If there's a network communication error or non-200 response
/// - InvalidResponse: If the API response format is unexpected
/// - OutOfMemory: If memory allocation fails
pub fn embedText(allocator: Allocator, base_url: []const u8, api_key: []const u8, model: []const u8, text: []const u8) Error![]f32 {
    if (api_key.len == 0) return Error.MissingApiKey;

    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const url = try std.fmt.allocPrint(allocator, "{s}/embeddings", .{base_url});
    defer allocator.free(url);

    // Build JSON body: {"model":"...","input":"..."}
    const body = try std.fmt.allocPrint(allocator, "{{\"model\":\"{s}\",\"input\":\"{s}\"}}", .{ model, text });
    defer allocator.free(body);

    const uri = std.Uri.parse(url) catch return Error.NetworkError;
    var req = try client.request(.POST, uri, .{});
    defer req.deinit();

    // Set headers manually (since extra_headers is not used elsewhere in repo)
    // Set headers in request before sending
    req.headers.content_type = .{ .override = "application/json" };
    var auth_buf: [512]u8 = undefined;
    const auth_slice = try std.fmt.bufPrint(auth_buf[0..], "Bearer {s}", .{api_key});
    req.headers.authorization = .{ .override = auth_slice };

    try req.sendBodyComplete(body);
    var redirect_buf: [1024]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);

    if (response.head.status != .ok) return Error.NetworkError;
    var list = try std.ArrayList(u8).initCapacity(allocator, 0);
    defer list.deinit(allocator);
    var buf: [8192]u8 = undefined;
    const rdr = response.reader(&.{});
    while (true) {
        const slice: []u8 = buf[0..];
        var slices = [_][]u8{slice};
        const n = rdr.readVec(slices[0..]) catch |err| switch (err) {
            error.ReadFailed => return Error.NetworkError,
            error.EndOfStream => 0,
        };
        if (n == 0) break;
        try list.appendSlice(allocator, buf[0..n]);
    }
    const resp = try list.toOwnedSlice(allocator);
    defer allocator.free(resp);

    // Expected shape: {"data":[{"embedding":[...]}]}
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, resp, .{});
    defer parsed.deinit();
    const root_obj = parsed.value.object;
    const data_val = root_obj.get("data") orelse return Error.InvalidResponse;
    const arr = data_val.array;
    if (arr.items.len == 0) return Error.InvalidResponse;
    const first = arr.items[0].object;
    return parseEmbeddingArray(allocator, first.get("embedding") orelse return Error.InvalidResponse);
}

/// Parses a JSON array of numbers into a slice of f32 values
///
/// Args:
/// - allocator: Memory allocator for the output slice
/// - v: JSON value that should be an array of numbers
///
/// Returns:
/// - A slice of f32 values parsed from the JSON array
/// - The caller owns the returned memory and must free it
///
/// Errors:
/// - OutOfMemory: If memory allocation fails
fn parseEmbeddingArray(allocator: Allocator, v: std.json.Value) Error![]f32 {
    const arr = v.array;
    const out = try allocator.alloc(f32, arr.items.len);
    var i: usize = 0;
    while (i < out.len) : (i += 1) {
        out[i] = @floatCast(arr.items[i].float);
    }
    return out;
}
