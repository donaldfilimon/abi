//! Google Cloud Functions Adapter
//!
//! Provides event parsing, response formatting, and context extraction
//! for Google Cloud Functions. Supports both HTTP triggers and event triggers
//! (Pub/Sub, Cloud Storage, Firestore, etc.).
//!
//! ## Usage
//!
//! ```zig
//! const cloud = @import("abi").cloud;
//!
//! pub fn handler(event: *cloud.CloudEvent, allocator: std.mem.Allocator) !cloud.CloudResponse {
//!     return try cloud.CloudResponse.json(allocator, "{\"message\":\"Hello from GCP!\"}");
//! }
//!
//! pub fn main(init: std.process.Init) !void {
//!     const arena = init.arena.allocator();
//!
//!     try cloud.gcp_functions.runHandler(arena, handler, 8080);
//! }
//! ```

const std = @import("std");
const types = @import("types.zig");
const time = @import("../../services/shared/time.zig");

pub const CloudEvent = types.CloudEvent;
pub const CloudResponse = types.CloudResponse;
pub const CloudHandler = types.CloudHandler;
pub const CloudError = types.CloudError;
pub const HttpMethod = types.HttpMethod;
pub const InvocationMetadata = types.InvocationMetadata;

// Shared helpers from types.zig
const jsonStringOrNull = types.jsonStringOrNull;
const parseJsonStringMap = types.parseJsonStringMap;
const cloneStringMap = types.cloneStringMap;
const parseJsonRoot = types.parseJsonRoot;

/// GCP Functions runtime configuration.
pub const GcpConfig = struct {
    /// Port to listen on (from PORT environment variable).
    port: u16 = 8080,
    /// Project ID (from GCP_PROJECT or GCLOUD_PROJECT).
    project_id: ?[]const u8 = null,
    /// Region (from FUNCTION_REGION).
    region: ?[]const u8 = null,
    /// Function name (from K_SERVICE or FUNCTION_NAME).
    function_name: ?[]const u8 = null,

    pub fn fromEnvironment() GcpConfig {
        return .{
            .port = blk: {
                if (std.c.getenv("PORT")) |port_str| {
                    break :blk std.fmt.parseInt(u16, std.mem.span(port_str), 10) catch 8080;
                }
                break :blk 8080;
            },
            .project_id = if (std.c.getenv("GCP_PROJECT")) |p| std.mem.span(p) else if (std.c.getenv("GCLOUD_PROJECT")) |p| std.mem.span(p) else null,
            .region = if (std.c.getenv("FUNCTION_REGION")) |p| std.mem.span(p) else null,
            .function_name = if (std.c.getenv("K_SERVICE")) |p| std.mem.span(p) else if (std.c.getenv("FUNCTION_NAME")) |p| std.mem.span(p) else null,
        };
    }
};

/// GCP Cloud Functions runtime.
pub const GcpRuntime = struct {
    allocator: std.mem.Allocator,
    config: GcpConfig,
    handler: CloudHandler,
    request_count: u64 = 0,

    /// Initialize the GCP Functions runtime.
    pub fn init(allocator: std.mem.Allocator, handler: CloudHandler) GcpRuntime {
        return .{
            .allocator = allocator,
            .config = GcpConfig.fromEnvironment(),
            .handler = handler,
        };
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(allocator: std.mem.Allocator, handler: CloudHandler, config: GcpConfig) GcpRuntime {
        return .{
            .allocator = allocator,
            .config = config,
            .handler = handler,
        };
    }

    /// Run the HTTP server for Cloud Functions.
    /// GCP Cloud Functions expects an HTTP server on the specified port.
    /// Starts a TCP listener using POSIX sockets and accepts HTTP requests
    /// in a loop, dispatching each to the user's handler.
    pub fn run(self: *GcpRuntime) !void {
        const sock = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
        if (sock < 0) return CloudError.ProviderError;
        defer _ = std.c.close(sock);

        // Enable SO_REUSEADDR to allow quick restarts
        const enable: c_int = 1;
        _ = std.c.setsockopt(
            sock,
            @intCast(std.c.SOL.SOCKET),
            std.c.SO.REUSEADDR,
            @ptrCast(&enable),
            @sizeOf(c_int),
        );

        var addr: std.c.sockaddr.in = .{
            .port = std.mem.nativeToBig(u16, self.config.port),
            .addr = 0, // INADDR_ANY
        };
        if (std.c.bind(sock, @ptrCast(&addr), @sizeOf(std.c.sockaddr.in)) < 0) {
            return CloudError.ProviderError;
        }
        if (std.c.listen(sock, 128) < 0) {
            return CloudError.ProviderError;
        }

        std.log.info("GCP Functions runtime listening on port {d}", .{self.config.port});

        // Accept loop
        while (true) {
            const client = std.c.accept(sock, null, null);
            if (client < 0) continue;
            defer _ = std.c.close(client);

            self.handleClient(client) catch |err| {
                std.log.warn("Error handling request: {t}", .{err});
                // Send a 500 response on error
                const err_resp = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n";
                _ = std.c.write(client, err_resp.ptr, err_resp.len);
            };
        }
    }

    /// Handle a single client connection: read the HTTP request, dispatch
    /// to the user handler, and write back an HTTP response.
    fn handleClient(self: *GcpRuntime, client: std.c.fd_t) !void {
        // Read request into buffer (max 64KB)
        var buf: [65536]u8 = undefined;
        var total_read: usize = 0;

        // Read until we have the full headers (terminated by \r\n\r\n)
        while (total_read < buf.len) {
            const n = std.c.read(client, buf[total_read..].ptr, buf.len - total_read);
            if (n <= 0) break;
            total_read += @intCast(n);
            if (std.mem.indexOf(u8, buf[0..total_read], "\r\n\r\n") != null) break;
        }
        if (total_read == 0) return;

        const request_data = buf[0..total_read];

        // Find end of headers
        const header_end = std.mem.indexOf(u8, request_data, "\r\n\r\n") orelse return;
        const headers_section = request_data[0..header_end];
        const body_start = header_end + 4;

        // Parse request line (first line)
        var line_iter = std.mem.splitSequence(u8, headers_section, "\r\n");
        const request_line = line_iter.next() orelse return;

        // Parse "METHOD /path HTTP/1.x"
        var parts = std.mem.splitScalar(u8, request_line, ' ');
        const method = parts.next() orelse return;
        const path = parts.next() orelse "/";

        // Parse headers into a map
        var header_map: std.StringHashMapUnmanaged([]const u8) = .empty;
        defer header_map.deinit(self.allocator);

        var content_length: usize = 0;
        while (line_iter.next()) |line| {
            if (line.len == 0) break;
            if (std.mem.indexOf(u8, line, ": ")) |sep| {
                const key = line[0..sep];
                const value = line[sep + 2 ..];
                try header_map.put(self.allocator, key, value);

                // Check for Content-Length (case-insensitive)
                if (std.ascii.eqlIgnoreCase(key, "content-length")) {
                    content_length = std.fmt.parseInt(usize, value, 10) catch 0;
                }
            }
        }

        // Read remaining body if Content-Length indicates more data
        var body_data = request_data[body_start..total_read];
        var dynamic_body: ?[]u8 = null;
        defer if (dynamic_body) |db| self.allocator.free(db);

        if (content_length > body_data.len and content_length <= 1048576) {
            // Need to read more body data
            var full_body = try self.allocator.alloc(u8, content_length);
            errdefer self.allocator.free(full_body);
            @memcpy(full_body[0..body_data.len], body_data);
            var body_read = body_data.len;

            while (body_read < content_length) {
                const n = std.c.read(client, full_body[body_read..].ptr, content_length - body_read);
                if (n <= 0) break;
                body_read += @intCast(n);
            }
            dynamic_body = full_body;
            body_data = full_body[0..body_read];
        }

        const body: ?[]const u8 = if (body_data.len > 0) body_data else null;

        // Dispatch to processRequest which handles handler invocation
        var response = try self.processRequest(method, path, header_map, body);
        defer response.deinit();

        // Build HTTP response
        var resp_buf = std.ArrayListUnmanaged(u8).empty;
        defer resp_buf.deinit(self.allocator);
        const writer = resp_buf.writer(self.allocator);

        try std.fmt.format(writer, "HTTP/1.1 {d} {s}\r\n", .{
            response.status_code,
            httpStatusText(response.status_code),
        });

        // Write response headers
        var has_content_type = false;
        var resp_iter = response.headers.iterator();
        while (resp_iter.next()) |entry| {
            try std.fmt.format(writer, "{s}: {s}\r\n", .{ entry.key_ptr.*, entry.value_ptr.* });
            if (std.ascii.eqlIgnoreCase(entry.key_ptr.*, "content-type")) {
                has_content_type = true;
            }
        }
        if (!has_content_type) {
            try writer.writeAll("Content-Type: application/octet-stream\r\n");
        }
        try std.fmt.format(writer, "Content-Length: {d}\r\n", .{response.body.len});
        try writer.writeAll("Connection: close\r\n\r\n");

        // Write body
        try writer.writeAll(response.body);

        // Send full response
        const resp_data = resp_buf.items;
        var sent: usize = 0;
        while (sent < resp_data.len) {
            const n = std.c.write(client, resp_data[sent..].ptr, resp_data.len - sent);
            if (n <= 0) break;
            sent += @intCast(n);
        }
    }

    /// Process a single HTTP request.
    pub fn processRequest(
        self: *GcpRuntime,
        method: []const u8,
        path: []const u8,
        headers: std.StringHashMapUnmanaged([]const u8),
        body: ?[]const u8,
    ) !CloudResponse {
        self.request_count += 1;

        var generated_request_id_buf: [32]u8 = undefined;
        const request_id = blk: {
            // Use X-Cloud-Trace-Context or generate one
            if (headers.get("x-cloud-trace-context")) |trace| {
                break :blk trace;
            }
            // Generate a simple request ID
            const generated = std.fmt.bufPrint(&generated_request_id_buf, "gcp-{d}", .{
                self.request_count,
            }) catch break :blk "unknown";
            break :blk generated;
        };

        var event = try parseHttpRequest(
            self.allocator,
            method,
            path,
            headers,
            body,
            request_id,
        );
        defer event.deinit();

        // Set GCP-specific context
        event.context.project_id = self.config.project_id;
        event.context.region = self.config.region;
        event.context.function_name = self.config.function_name;

        // Execute handler
        if (self.handler(&event, self.allocator)) |response| {
            return response;
        } else |err| {
            var err_name_buf: [128]u8 = undefined;
            const err_name = std.fmt.bufPrint(&err_name_buf, "{t}", .{err}) catch "HandlerError";
            return CloudResponse.err(self.allocator, 500, err_name);
        }
    }
};

/// Parse an HTTP request into a CloudEvent.
pub fn parseHttpRequest(
    allocator: std.mem.Allocator,
    method: []const u8,
    path: []const u8,
    headers: std.StringHashMapUnmanaged([]const u8),
    body: ?[]const u8,
    request_id: []const u8,
) !CloudEvent {
    var event = CloudEvent.init(allocator, .gcp_functions, request_id);
    errdefer event.deinit();

    event.method = HttpMethod.fromString(method);
    event.path = path;
    event.body = body;

    // Clone headers
    event.headers = try cloneStringMap(allocator, headers);

    // Parse query parameters from path
    if (std.mem.indexOf(u8, path, "?")) |query_start| {
        event.path = path[0..query_start];
        const query_string = path[query_start + 1 ..];
        event.query_params = try parseQueryString(allocator, query_string);
    }

    return event;
}

fn jsonValueToOwnedSlice(allocator: std.mem.Allocator, value: std.json.Value) ![]const u8 {
    var buffer = std.ArrayListUnmanaged(u8).empty;
    defer buffer.deinit(allocator);
    try std.json.stringify(value, .{}, buffer.writer(allocator));
    return buffer.toOwnedSlice(allocator);
}

/// Parse a CloudEvent (structured event format).
/// GCP uses the CloudEvents specification for event-driven functions.
pub fn parseCloudEvent(allocator: std.mem.Allocator, raw_event: []const u8) !CloudEvent {
    const parsed = try parseJsonRoot(allocator, raw_event);
    defer parsed.deinit();

    const root = parsed.value;

    // Extract CloudEvents attributes
    const event_id = if (root.object.get("id")) |id| id.string else "unknown";
    const event_type = if (root.object.get("type")) |t| t.string else null;
    const source = if (root.object.get("source")) |s| s.string else null;

    var event = CloudEvent.init(allocator, .gcp_functions, event_id);
    errdefer event.deinit();

    event.event_type = event_type;
    event.source = source;

    // Extract data payload
    if (root.object.get("data")) |data| {
        event.body = if (jsonStringOrNull(data)) |body| body else try jsonValueToOwnedSlice(allocator, data);
    }

    // Extract timestamp
    if (root.object.get("time")) |time_value| {
        // Parse ISO 8601 timestamp (simplified)
        _ = time_value;
        event.timestamp = time.unixSeconds();
    }

    return event;
}

/// Parse a Pub/Sub message.
pub fn parsePubSubMessage(allocator: std.mem.Allocator, raw_event: []const u8) !CloudEvent {
    const parsed = try parseJsonRoot(allocator, raw_event);
    defer parsed.deinit();

    const root = parsed.value;

    // Pub/Sub wraps the message in a 'message' field
    const message = root.object.get("message") orelse root;

    const message_id = if (message.object.get("messageId")) |id|
        id.string
    else if (message.object.get("message_id")) |id|
        id.string
    else
        "unknown";

    var event = CloudEvent.init(allocator, .gcp_functions, message_id);
    errdefer event.deinit();

    event.event_type = "google.cloud.pubsub.topic.v1.messagePublished";

    // Get the message data (base64 encoded)
    if (message.object.get("data")) |data| {
        if (data == .string) {
            // Decode base64
            const decoded = try decodeBase64(allocator, data.string);
            event.body = decoded;
        }
    }

    // Get attributes as headers
    if (message.object.get("attributes")) |attrs| {
        if (attrs != .null) {
            event.headers = try parseJsonStringMap(allocator, attrs);
        }
    }

    return event;
}

/// Parse a Cloud Storage event.
pub fn parseStorageEvent(allocator: std.mem.Allocator, raw_event: []const u8) !CloudEvent {
    const parsed = try parseJsonRoot(allocator, raw_event);
    defer parsed.deinit();

    const root = parsed.value;

    const event_id = if (root.object.get("id")) |id| id.string else "unknown";

    var event = CloudEvent.init(allocator, .gcp_functions, event_id);
    errdefer event.deinit();

    // Determine event type from the event
    if (root.object.get("kind")) |kind| {
        if (std.mem.eql(u8, kind.string, "storage#object")) {
            event.event_type = "google.cloud.storage.object.v1.finalized";
        }
    }

    // Extract bucket and object information
    if (root.object.get("bucket")) |bucket| {
        event.source = bucket.string;
    }

    // Store the whole event as the body for full access
    event.body = raw_event;

    return event;
}

/// Format a CloudResponse for GCP Functions (HTTP response).
pub fn formatResponse(allocator: std.mem.Allocator, response: *const CloudResponse) ![]const u8 {
    // For HTTP functions, GCP expects a standard HTTP response
    // The body is returned directly, not wrapped in JSON
    _ = allocator;
    return response.body;
}

/// Run a handler as a GCP Cloud Function.
pub fn runHandler(allocator: std.mem.Allocator, handler: CloudHandler, port: u16) !void {
    var runtime = GcpRuntime.initWithConfig(allocator, handler, .{ .port = port });
    try runtime.run();
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Map an HTTP status code to its standard reason phrase.
fn httpStatusText(code: u16) []const u8 {
    return switch (code) {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        301 => "Moved Permanently",
        302 => "Found",
        304 => "Not Modified",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        else => "OK",
    };
}

/// Parse a query string into key-value pairs.
fn parseQueryString(allocator: std.mem.Allocator, query_string: []const u8) !std.StringHashMapUnmanaged([]const u8) {
    var params: std.StringHashMapUnmanaged([]const u8) = .empty;
    errdefer params.deinit(allocator);

    var pairs = std.mem.splitScalar(u8, query_string, '&');
    while (pairs.next()) |pair| {
        if (pair.len == 0) continue;

        if (std.mem.indexOf(u8, pair, "=")) |eq_pos| {
            const key = pair[0..eq_pos];
            const value = pair[eq_pos + 1 ..];
            try params.put(allocator, key, value);
        } else {
            try params.put(allocator, pair, "");
        }
    }

    return params;
}

/// Decode a base64 string.
fn decodeBase64(allocator: std.mem.Allocator, encoded: []const u8) ![]const u8 {
    const decoder = std.base64.standard.Decoder;
    const decoded_len = decoder.calcSizeForSlice(encoded) catch return CloudError.EventParseFailed;
    const decoded = try allocator.alloc(u8, decoded_len);
    errdefer allocator.free(decoded);

    decoder.decode(decoded, encoded) catch return CloudError.EventParseFailed;
    return decoded;
}

// ============================================================================
// Tests
// ============================================================================

test "parseHttpRequest" {
    const allocator = std.testing.allocator;

    var headers: std.StringHashMapUnmanaged([]const u8) = .empty;
    defer headers.deinit(allocator);
    try headers.put(allocator, "content-type", "application/json");

    var event = try parseHttpRequest(
        allocator,
        "POST",
        "/api/test?id=123",
        headers,
        "{\"data\":\"test\"}",
        "req-123",
    );
    defer event.deinit();

    try std.testing.expectEqual(HttpMethod.POST, event.method.?);
    try std.testing.expectEqualStrings("/api/test", event.path.?);
    try std.testing.expectEqualStrings("{\"data\":\"test\"}", event.body.?);
    try std.testing.expectEqualStrings("123", event.query_params.?.get("id").?);
    try std.testing.expectEqualStrings("application/json", event.getHeader("Content-Type").?);
}

test "parseCloudEvent" {
    const allocator = std.testing.allocator;

    const raw_event =
        \\{
        \\  "id": "event-123",
        \\  "type": "google.cloud.storage.object.v1.finalized",
        \\  "source": "//storage.googleapis.com/projects/_/buckets/my-bucket",
        \\  "data": {"name": "test.txt", "bucket": "my-bucket"}
        \\}
    ;

    var event = try parseCloudEvent(allocator, raw_event);
    defer {
        if (event.body) |body| {
            allocator.free(body);
        }
        event.deinit();
    }

    try std.testing.expectEqualStrings("event-123", event.request_id);
    try std.testing.expectEqualStrings("google.cloud.storage.object.v1.finalized", event.event_type.?);
}

test "parseQueryString" {
    const allocator = std.testing.allocator;

    var params = try parseQueryString(allocator, "foo=bar&baz=qux&empty=");
    defer params.deinit();

    try std.testing.expectEqualStrings("bar", params.get("foo").?);
    try std.testing.expectEqualStrings("qux", params.get("baz").?);
    try std.testing.expectEqualStrings("", params.get("empty").?);
}

test "GcpConfig fromEnvironment defaults" {
    const config = GcpConfig.fromEnvironment();
    try std.testing.expectEqual(@as(u16, 8080), config.port);
}
