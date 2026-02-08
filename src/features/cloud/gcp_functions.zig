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
//! pub fn main() !void {
//!     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//!     defer _ = gpa.deinit();
//!     const allocator = gpa.allocator();
//!
//!     try cloud.gcp_functions.runHandler(allocator, handler, 8080);
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
const cloneStringMapLowercase = types.cloneStringMapLowercase;
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
                if (std.posix.getenv("PORT")) |port_str| {
                    break :blk std.fmt.parseInt(u16, port_str, 10) catch 8080;
                }
                break :blk 8080;
            },
            .project_id = std.posix.getenv("GCP_PROJECT") orelse std.posix.getenv("GCLOUD_PROJECT"),
            .region = std.posix.getenv("FUNCTION_REGION"),
            .function_name = std.posix.getenv("K_SERVICE") orelse std.posix.getenv("FUNCTION_NAME"),
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
    pub fn run(self: *GcpRuntime) !void {
        // In a real implementation, this would start an HTTP server
        // For now, we provide the interface
        _ = self;
        return CloudError.ProviderError;
    }

    /// Process a single HTTP request.
    pub fn processRequest(
        self: *GcpRuntime,
        method: []const u8,
        path: []const u8,
        headers: std.StringHashMap([]const u8),
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
    headers: std.StringHashMap([]const u8),
    body: ?[]const u8,
    request_id: []const u8,
) !CloudEvent {
    var event = CloudEvent.init(allocator, .gcp_functions, request_id);
    errdefer event.deinit();

    event.method = HttpMethod.fromString(method);
    event.path = path;
    event.body = body;

    // Clone headers
    event.headers = try cloneStringMapLowercase(allocator, headers);

    // Parse query parameters from path
    if (std.mem.indexOf(u8, path, "?")) |query_start| {
        event.path = path[0..query_start];
        const query_string = path[query_start + 1 ..];
        event.query_params = try parseQueryString(allocator, query_string);
    }

    return event;
}

fn jsonValueToOwnedSlice(allocator: std.mem.Allocator, value: std.json.Value) ![]const u8 {
    var buffer = std.ArrayListUnmanaged(u8){};
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

/// Format headers for GCP Functions response.
pub fn formatHeaders(response: *const CloudResponse) std.StringHashMap([]const u8) {
    return response.headers;
}

/// Run a handler as a GCP Cloud Function.
pub fn runHandler(allocator: std.mem.Allocator, handler: CloudHandler, port: u16) !void {
    var runtime = GcpRuntime.initWithConfig(allocator, handler, .{ .port = port });
    try runtime.run();
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse a query string into key-value pairs.
fn parseQueryString(allocator: std.mem.Allocator, query_string: []const u8) !std.StringHashMap([]const u8) {
    var params = std.StringHashMap([]const u8).init(allocator);
    errdefer params.deinit();

    var pairs = std.mem.splitScalar(u8, query_string, '&');
    while (pairs.next()) |pair| {
        if (pair.len == 0) continue;

        if (std.mem.indexOf(u8, pair, "=")) |eq_pos| {
            const key = pair[0..eq_pos];
            const value = pair[eq_pos + 1 ..];
            try params.put(key, value);
        } else {
            try params.put(pair, "");
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

    var headers = std.StringHashMap([]const u8).init(allocator);
    defer headers.deinit();
    try headers.put("content-type", "application/json");

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
