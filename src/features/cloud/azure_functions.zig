//! Azure Functions Adapter
//!
//! Provides event parsing, response formatting, and context extraction
//! for Azure Functions. Supports HTTP triggers, timer triggers, and
//! various Azure service triggers (Blob Storage, Queue, Event Hubs, etc.).
//!
//! ## Usage
//!
//! ```zig
//! const cloud = @import("abi").cloud;
//!
//! pub fn handler(event: *cloud.CloudEvent, allocator: std.mem.Allocator) !cloud.CloudResponse {
//!     return try cloud.CloudResponse.json(allocator, "{\"message\":\"Hello from Azure!\"}");
//! }
//!
//! pub fn main() !void {
//!     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//!     defer _ = gpa.deinit();
//!     const allocator = gpa.allocator();
//!
//!     try cloud.azure_functions.runHandler(allocator, handler);
//! }
//! ```

const std = @import("std");
const types = @import("types.zig");

pub const CloudEvent = types.CloudEvent;
pub const CloudResponse = types.CloudResponse;
pub const CloudHandler = types.CloudHandler;
pub const CloudError = types.CloudError;
pub const HttpMethod = types.HttpMethod;
pub const InvocationMetadata = types.InvocationMetadata;

// Shared helpers from types.zig
const jsonStringOrNull = types.jsonStringOrNull;
const parseJsonStringMap = types.parseJsonStringMap;
const parseJsonHeaderMap = types.parseJsonHeaderMap;
const parseJsonRoot = types.parseJsonRoot;

/// Azure Functions runtime configuration.
pub const AzureConfig = struct {
    /// Functions host port (from FUNCTIONS_HTTPWORKER_PORT).
    port: u16 = 7071,
    /// Function name.
    function_name: ?[]const u8 = null,
    /// Invocation ID.
    invocation_id: ?[]const u8 = null,
    /// Azure Website name.
    website_name: ?[]const u8 = null,
    /// Azure region.
    region: ?[]const u8 = null,

    pub fn fromEnvironment() AzureConfig {
        return .{
            .port = blk: {
                if (std.c.getenv("FUNCTIONS_HTTPWORKER_PORT")) |port_str| {
                    break :blk std.fmt.parseInt(u16, std.mem.span(port_str), 10) catch 7071;
                }
                break :blk 7071;
            },
            .function_name = if (std.c.getenv("FUNCTION_NAME")) |p| std.mem.span(p) else null,
            .website_name = if (std.c.getenv("WEBSITE_SITE_NAME")) |p| std.mem.span(p) else null,
            .region = if (std.c.getenv("REGION_NAME")) |p| std.mem.span(p) else null,
        };
    }
};

/// Azure Functions trigger types.
pub const TriggerType = enum {
    http,
    timer,
    blob,
    queue,
    event_hub,
    service_bus,
    cosmos_db,
    event_grid,
    unknown,

    pub fn fromString(s: []const u8) TriggerType {
        return std.StaticStringMap(TriggerType).initComptime(.{
            .{ "httpTrigger", .http },
            .{ "timerTrigger", .timer },
            .{ "blobTrigger", .blob },
            .{ "queueTrigger", .queue },
            .{ "eventHubTrigger", .event_hub },
            .{ "serviceBusTrigger", .service_bus },
            .{ "cosmosDBTrigger", .cosmos_db },
            .{ "eventGridTrigger", .event_grid },
        }).get(s) orelse .unknown;
    }
};

/// Azure Functions custom handler runtime.
pub const AzureRuntime = struct {
    allocator: std.mem.Allocator,
    config: AzureConfig,
    handler: CloudHandler,
    request_count: u64 = 0,

    /// Initialize the Azure Functions runtime.
    pub fn init(allocator: std.mem.Allocator, handler: CloudHandler) AzureRuntime {
        return .{
            .allocator = allocator,
            .config = AzureConfig.fromEnvironment(),
            .handler = handler,
        };
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(allocator: std.mem.Allocator, handler: CloudHandler, config: AzureConfig) AzureRuntime {
        return .{
            .allocator = allocator,
            .config = config,
            .handler = handler,
        };
    }

    /// Run the custom handler HTTP server.
    /// Azure Functions custom handlers communicate via HTTP on
    /// FUNCTIONS_HTTPWORKER_PORT (default 7071). Starts a TCP listener
    /// using POSIX sockets and dispatches each request to the user handler.
    pub fn run(self: *AzureRuntime) !void {
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

        std.log.info("Azure Functions custom handler listening on port {d}", .{self.config.port});

        // Accept loop
        while (true) {
            const client = std.c.accept(sock, null, null);
            if (client < 0) continue;
            defer _ = std.c.close(client);

            self.handleClient(client) catch |err| {
                std.log.warn("Error handling Azure request: {t}", .{err});
                const err_resp = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n";
                _ = std.c.write(client, err_resp.ptr, err_resp.len);
            };
        }
    }

    /// Handle a single client connection from the Azure Functions host.
    /// Reads the HTTP request body (JSON invocation payload), processes it
    /// via the user handler, and writes back an HTTP response with the
    /// formatted Azure invocation response.
    fn handleClient(self: *AzureRuntime, client: std.c.fd_t) !void {
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

        // Parse Content-Length from headers
        var content_length: usize = 0;
        var line_iter = std.mem.splitSequence(u8, headers_section, "\r\n");
        _ = line_iter.next(); // skip request line
        while (line_iter.next()) |line| {
            if (line.len == 0) break;
            if (std.mem.indexOf(u8, line, ": ")) |sep| {
                const key = line[0..sep];
                const value = line[sep + 2 ..];
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

        if (body_data.len == 0) return;

        // Process the Azure invocation request (JSON body from Functions host)
        const response_json = try self.processInvocation(body_data);
        defer self.allocator.free(response_json);

        // Build HTTP response
        var resp_buf = std.ArrayListUnmanaged(u8).empty;
        defer resp_buf.deinit(self.allocator);
        const writer = resp_buf.writer(self.allocator);

        try writer.writeAll("HTTP/1.1 200 OK\r\n");
        try writer.writeAll("Content-Type: application/json\r\n");
        try std.fmt.format(writer, "Content-Length: {d}\r\n", .{response_json.len});
        try writer.writeAll("Connection: close\r\n\r\n");
        try writer.writeAll(response_json);

        // Send full response
        const resp_data = resp_buf.items;
        var sent: usize = 0;
        while (sent < resp_data.len) {
            const n = std.c.write(client, resp_data[sent..].ptr, resp_data.len - sent);
            if (n <= 0) break;
            sent += @intCast(n);
        }
    }

    /// Process an invocation request from the Functions host.
    pub fn processInvocation(self: *AzureRuntime, raw_request: []const u8) ![]const u8 {
        self.request_count += 1;

        var event = try parseInvocationRequest(self.allocator, raw_request);
        defer event.deinit();

        // Set Azure-specific context
        event.context.function_name = self.config.function_name;
        event.context.invocation_id = event.request_id;

        // Execute handler
        var response: CloudResponse = undefined;
        if (self.handler(&event, self.allocator)) |resp| {
            response = resp;
        } else |err| {
            var err_name_buf: [128]u8 = undefined;
            const err_name = std.fmt.bufPrint(&err_name_buf, "{t}", .{err}) catch "HandlerError";
            response = try CloudResponse.err(self.allocator, 500, err_name);
        }
        defer response.deinit();

        // Format response for Azure Functions host
        return formatInvocationResponse(self.allocator, &response);
    }
};

/// Parse an Azure Functions invocation request.
/// Azure custom handlers receive requests in a specific JSON format.
pub fn parseInvocationRequest(allocator: std.mem.Allocator, raw_request: []const u8) !CloudEvent {
    const parsed = try parseJsonRoot(allocator, raw_request);
    defer parsed.deinit();

    const root = parsed.value;

    // Extract invocation metadata
    const invocation_id = if (root.object.get("Metadata")) |meta|
        if (meta.object.get("InvocationId")) |id| id.string else "unknown"
    else
        "unknown";

    var event = CloudEvent.init(allocator, .azure_functions, invocation_id);
    errdefer event.deinit();

    // Check if this is an HTTP trigger
    if (root.object.get("Data")) |data| {
        if (data.object.get("req")) |req| {
            try parseHttpTrigger(&event, req);
        } else {
            // Non-HTTP trigger - store raw data
            try parseNonHttpTrigger(&event, data, root);
        }
    }

    return event;
}

fn getStringField(root: std.json.Value, key: []const u8) ?[]const u8 {
    const value = root.object.get(key) orelse return null;
    return jsonStringOrNull(value);
}

fn jsonValueToOwnedSlice(allocator: std.mem.Allocator, value: std.json.Value) ![]const u8 {
    var buffer = std.ArrayListUnmanaged(u8).empty;
    defer buffer.deinit(allocator);
    try std.json.stringify(value, .{}, buffer.writer(allocator));
    return buffer.toOwnedSlice(allocator);
}

/// Parse HTTP trigger request data.
fn parseHttpTrigger(event: *CloudEvent, req: std.json.Value) !void {
    // HTTP method
    if (req.object.get("Method")) |method| {
        event.method = HttpMethod.fromString(method.string);
    }

    // URL/Path
    if (req.object.get("Url")) |url| {
        event.path = url.string;
    }

    // Body
    if (req.object.get("Body")) |body| event.body = jsonStringOrNull(body);

    // Headers
    if (req.object.get("Headers")) |headers| {
        if (headers != .null) {
            event.headers = try parseJsonHeaderMap(event.allocator, headers);
        }
    }

    // Query parameters
    if (req.object.get("Query")) |query| {
        if (query != .null) {
            event.query_params = try parseJsonStringMap(event.allocator, query);
        }
    }
}

/// Parse non-HTTP trigger data (Queue, Blob, Timer, etc.).
fn parseNonHttpTrigger(event: *CloudEvent, data: std.json.Value, root: std.json.Value) !void {
    // Determine trigger type from metadata
    if (root.object.get("Metadata")) |meta| {
        if (meta.object.get("sys")) |sys| {
            if (sys.object.get("MethodName")) |method_name| {
                event.event_type = method_name.string;
            }
        }
    }

    // For queue/blob triggers, the message/content is in the data
    var iter = data.object.iterator();
    while (iter.next()) |entry| {
        // Skip the "req" key if present
        if (std.mem.eql(u8, entry.key_ptr.*, "req")) continue;

        // The first non-req value is typically the trigger data
        event.source = entry.key_ptr.*;

        if (entry.value_ptr.* == .string) {
            event.body = entry.value_ptr.string;
        } else {
            // Serialize complex data as JSON
            event.body = try jsonValueToOwnedSlice(event.allocator, entry.value_ptr.*);
        }
        break;
    }
}

/// Parse a timer trigger event.
pub fn parseTimerTrigger(allocator: std.mem.Allocator, raw_event: []const u8) !CloudEvent {
    const parsed = try parseJsonRoot(allocator, raw_event);
    defer parsed.deinit();

    const root = parsed.value;

    var event = CloudEvent.init(allocator, .azure_functions, "timer");
    errdefer event.deinit();

    event.event_type = "timer";

    // Timer info contains schedule details
    if (root.object.get("Data")) |data| {
        if (data.object.get("timer")) |timer| {
            // Check if timer is past due
            if (timer.object.get("IsPastDue")) |past_due| {
                if (past_due.bool) {
                    // Store metadata about past due execution
                    event.headers = .empty;
                    try event.headers.?.put(allocator, "x-timer-past-due", "true");
                }
            }

            // Get schedule status
            if (timer.object.get("ScheduleStatus")) |status| {
                event.body = try jsonValueToOwnedSlice(allocator, status);
            }
        }
    }

    return event;
}

/// Parse a Blob Storage trigger event.
pub fn parseBlobTrigger(allocator: std.mem.Allocator, raw_event: []const u8) !CloudEvent {
    const parsed = try parseJsonRoot(allocator, raw_event);
    defer parsed.deinit();

    const root = parsed.value;

    var event = CloudEvent.init(allocator, .azure_functions, "blob");
    errdefer event.deinit();

    event.event_type = "Microsoft.Storage.BlobCreated";

    if (root.object.get("Data")) |data| {
        // Blob content or URI
        var iter = data.object.iterator();
        while (iter.next()) |entry| {
            if (!std.mem.eql(u8, entry.key_ptr.*, "req")) {
                event.source = entry.key_ptr.*;

                if (entry.value_ptr.* == .string) {
                    event.body = entry.value_ptr.string;
                }
                break;
            }
        }
    }

    // Blob metadata from trigger metadata
    if (root.object.get("Metadata")) |meta| {
        event.headers = .empty;

        if (getStringField(meta, "BlobTrigger")) |trigger| {
            try event.headers.?.put(allocator, "x-blob-trigger", trigger);
        }
        if (getStringField(meta, "Uri")) |uri| {
            try event.headers.?.put(allocator, "x-blob-uri", uri);
        }
    }

    return event;
}

/// Format a CloudResponse for Azure Functions custom handler.
pub fn formatInvocationResponse(allocator: std.mem.Allocator, response: *const CloudResponse) ![]const u8 {
    var buffer = std.ArrayListUnmanaged(u8).empty;
    errdefer buffer.deinit(allocator);

    const writer = buffer.writer(allocator);

    // Azure Functions custom handler response format
    try writer.writeAll("{\"Outputs\":{\"res\":{");

    // Status code
    try std.fmt.format(writer, "\"statusCode\":{d}", .{response.status_code});

    // Body
    try writer.writeAll(",\"body\":");
    try std.json.stringify(response.body, .{}, writer);

    // Headers
    try writer.writeAll(",\"headers\":{");
    var first = true;
    var iter = response.headers.iterator();
    while (iter.next()) |entry| {
        if (!first) try writer.writeAll(",");
        first = false;
        try std.json.stringify(entry.key_ptr.*, .{}, writer);
        try writer.writeAll(":");
        try std.json.stringify(entry.value_ptr.*, .{}, writer);
    }
    try writer.writeAll("}");

    try writer.writeAll("}},\"Logs\":[]}");

    return buffer.toOwnedSlice(allocator);
}

/// Run a handler as an Azure Function.
pub fn runHandler(allocator: std.mem.Allocator, handler: CloudHandler) !void {
    var runtime = AzureRuntime.init(allocator, handler);
    try runtime.run();
}

/// Generate a function.json binding configuration.
pub fn generateFunctionJson(
    allocator: std.mem.Allocator,
    function_name: []const u8,
    trigger_type: TriggerType,
) ![]const u8 {
    var buffer = std.ArrayListUnmanaged(u8).empty;
    errdefer buffer.deinit(allocator);

    const writer = buffer.writer(allocator);

    try writer.writeAll("{\n  \"bindings\": [\n");

    switch (trigger_type) {
        .http => {
            try std.fmt.format(writer,
                \\    {{
                \\      "authLevel": "anonymous",
                \\      "type": "httpTrigger",
                \\      "direction": "in",
                \\      "name": "req",
                \\      "methods": ["get", "post", "put", "delete"]
                \\    }},
                \\    {{
                \\      "type": "http",
                \\      "direction": "out",
                \\      "name": "res"
                \\    }}
            , .{});
        },
        .timer => {
            try std.fmt.format(writer,
                \\    {{
                \\      "name": "timer",
                \\      "type": "timerTrigger",
                \\      "direction": "in",
                \\      "schedule": "0 */5 * * * *"
                \\    }}
            , .{});
        },
        .blob => {
            try std.fmt.format(writer,
                \\    {{
                \\      "name": "blob",
                \\      "type": "blobTrigger",
                \\      "direction": "in",
                \\      "path": "container/{{name}}",
                \\      "connection": "AzureWebJobsStorage"
                \\    }}
            , .{});
        },
        .queue => {
            try std.fmt.format(writer,
                \\    {{
                \\      "name": "message",
                \\      "type": "queueTrigger",
                \\      "direction": "in",
                \\      "queueName": "myqueue",
                \\      "connection": "AzureWebJobsStorage"
                \\    }}
            , .{});
        },
        else => {
            try writer.writeAll("    // Configure trigger bindings\n");
        },
    }

    try writer.writeAll("\n  ],\n");
    try std.fmt.format(writer, "  \"customHandler\": {{\n    \"description\": {{\n      \"defaultExecutablePath\": \"{s}\"\n    }}\n  }}\n}}\n", .{function_name});

    return buffer.toOwnedSlice(allocator);
}

// ============================================================================
// Tests
// ============================================================================

test "parseInvocationRequest HTTP trigger" {
    const allocator = std.testing.allocator;

    const raw_request =
        \\{
        \\  "Data": {
        \\    "req": {
        \\      "Method": "POST",
        \\      "Url": "https://example.com/api/test",
        \\      "Headers": {"Content-Type": ["application/json"]},
        \\      "Query": {"id": "123"},
        \\      "Body": "{\"data\":\"test\"}"
        \\    }
        \\  },
        \\  "Metadata": {
        \\    "InvocationId": "inv-123"
        \\  }
        \\}
    ;

    var event = try parseInvocationRequest(allocator, raw_request);
    defer event.deinit();

    try std.testing.expectEqualStrings("inv-123", event.request_id);
    try std.testing.expectEqual(HttpMethod.POST, event.method.?);
    try std.testing.expectEqualStrings("{\"data\":\"test\"}", event.body.?);
    try std.testing.expectEqualStrings("application/json", event.getHeader("Content-Type").?);
}

test "formatInvocationResponse" {
    const allocator = std.testing.allocator;

    var response = try CloudResponse.json(allocator, "{\"ok\":true}");
    defer response.deinit();

    const formatted = try formatInvocationResponse(allocator, &response);
    defer allocator.free(formatted);

    // Verify it's valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, formatted, .{});
    defer parsed.deinit();

    try std.testing.expect(parsed.value.object.get("Outputs") != null);
}

test "TriggerType fromString" {
    try std.testing.expectEqual(TriggerType.http, TriggerType.fromString("httpTrigger"));
    try std.testing.expectEqual(TriggerType.timer, TriggerType.fromString("timerTrigger"));
    try std.testing.expectEqual(TriggerType.unknown, TriggerType.fromString("unknown"));
}

test "AzureConfig fromEnvironment defaults" {
    const config = AzureConfig.fromEnvironment();
    try std.testing.expectEqual(@as(u16, 7071), config.port);
}

test "generateFunctionJson HTTP" {
    const allocator = std.testing.allocator;

    const json = try generateFunctionJson(allocator, "myfunction", .http);
    defer allocator.free(json);

    // Verify it contains expected content
    try std.testing.expect(std.mem.indexOf(u8, json, "httpTrigger") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "bindings") != null);
}
