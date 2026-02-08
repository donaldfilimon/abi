//! Middleware Types
//!
//! Core types for the HTTP middleware pipeline. Middleware functions
//! can inspect/modify requests and responses, or short-circuit processing.

const std = @import("std");
const server = @import("../server/mod.zig");
const user_id_key = "user_id";

/// Context passed through the middleware chain.
pub const MiddlewareContext = struct {
    /// The parsed HTTP request.
    request: *server.ParsedRequest,
    /// The response builder.
    response: *server.ResponseBuilder,
    /// Allocator for middleware use.
    allocator: std.mem.Allocator,
    /// Custom state that middleware can set/get.
    state: std.StringHashMap([]const u8),
    /// Whether processing should continue.
    should_continue: bool = true,
    /// Whether response has been sent.
    response_sent: bool = false,
    /// Path parameters extracted from route.
    path_params: std.StringHashMap([]const u8),
    /// Request start time for timing.
    start_time: i64,

    /// Creates a new middleware context.
    pub fn init(
        allocator: std.mem.Allocator,
        request: *server.ParsedRequest,
        response: *server.ResponseBuilder,
    ) MiddlewareContext {
        return .{
            .request = request,
            .response = response,
            .allocator = allocator,
            .state = std.StringHashMap([]const u8).init(allocator),
            .path_params = std.StringHashMap([]const u8).init(allocator),
            .start_time = std.time.milliTimestamp(),
        };
    }

    /// Cleans up context resources.
    pub fn deinit(self: *MiddlewareContext) void {
        self.state.deinit();
        self.path_params.deinit();
    }

    /// Sets a state value.
    pub fn set(self: *MiddlewareContext, key: []const u8, value: []const u8) !void {
        try self.state.put(key, value);
    }

    /// Gets a state value.
    pub fn get(self: *const MiddlewareContext, key: []const u8) ?[]const u8 {
        return self.state.get(key);
    }

    /// Gets a path parameter.
    pub fn getParam(self: *const MiddlewareContext, name: []const u8) ?[]const u8 {
        return self.path_params.get(name);
    }

    /// Aborts the middleware chain and sends response immediately.
    pub fn abort(self: *MiddlewareContext) void {
        self.should_continue = false;
    }

    /// Returns elapsed time in milliseconds.
    pub fn elapsedMs(self: *const MiddlewareContext) i64 {
        return std.time.milliTimestamp() - self.start_time;
    }

    /// Checks if user is authenticated (via state).
    pub fn isAuthenticated(self: *const MiddlewareContext) bool {
        return self.getUserId() != null;
    }

    /// Gets the authenticated user ID.
    pub fn getUserId(self: *const MiddlewareContext) ?[]const u8 {
        return self.state.get(user_id_key);
    }
};

/// Middleware function signature.
pub const MiddlewareFn = *const fn (ctx: *MiddlewareContext) anyerror!void;

/// Chain of middleware functions.
pub const MiddlewareChain = struct {
    middlewares: std.ArrayListUnmanaged(MiddlewareFn),
    allocator: std.mem.Allocator,

    /// Creates a new middleware chain.
    pub fn init(allocator: std.mem.Allocator) MiddlewareChain {
        return .{
            .middlewares = .empty,
            .allocator = allocator,
        };
    }

    /// Cleans up resources.
    pub fn deinit(self: *MiddlewareChain) void {
        self.middlewares.deinit(self.allocator);
    }

    /// Adds a middleware to the chain.
    pub fn use(self: *MiddlewareChain, middleware: MiddlewareFn) !void {
        try self.middlewares.append(self.allocator, middleware);
    }

    /// Executes all middleware in order.
    pub fn execute(self: *const MiddlewareChain, ctx: *MiddlewareContext) !void {
        for (self.middlewares.items) |middleware| {
            if (!ctx.should_continue) break;
            try middleware(ctx);
        }
    }

    /// Returns the number of middleware in the chain.
    pub fn len(self: *const MiddlewareChain) usize {
        return self.middlewares.items.len;
    }

    /// Clears all middleware.
    pub fn clear(self: *MiddlewareChain) void {
        self.middlewares.clearRetainingCapacity();
    }
};

/// Handler function that runs after middleware.
pub const HandlerFn = *const fn (ctx: *MiddlewareContext) anyerror!void;

/// Combined middleware + handler for a route.
pub const RouteHandler = struct {
    middleware: ?MiddlewareChain,
    handler: HandlerFn,

    pub fn execute(self: *const RouteHandler, ctx: *MiddlewareContext) !void {
        // Run route-specific middleware
        if (self.middleware) |*mw| {
            try mw.execute(ctx);
            if (!ctx.should_continue) return;
        }

        // Run handler
        try self.handler(ctx);
    }
};

fn makeTestRequest(allocator: std.mem.Allocator, path: []const u8) server.ParsedRequest {
    return .{
        .method = .GET,
        .path = path,
        .query = null,
        .version = .http_1_1,
        .headers = std.StringHashMap([]const u8).init(allocator),
        .body = null,
        .raw_path = path,
        .allocator = allocator,
        .owned_data = null,
    };
}

test "MiddlewareContext basic operations" {
    const allocator = std.testing.allocator;

    // Create mock request/response
    var request = makeTestRequest(allocator, "/test");
    defer request.deinit();

    var response = server.ResponseBuilder.init(allocator);
    defer response.deinit();

    var ctx = MiddlewareContext.init(allocator, &request, &response);
    defer ctx.deinit();

    // Test state
    try ctx.set("key", "value");
    try std.testing.expectEqualStrings("value", ctx.get("key").?);
    try std.testing.expect(ctx.get("nonexistent") == null);

    // Test abort
    try std.testing.expect(ctx.should_continue);
    ctx.abort();
    try std.testing.expect(!ctx.should_continue);
}

test "MiddlewareChain execution" {
    const allocator = std.testing.allocator;

    const middleware1 = struct {
        fn run(ctx: *MiddlewareContext) !void {
            try ctx.set("mw1", "1");
        }
    }.run;

    const middleware2 = struct {
        fn run(ctx: *MiddlewareContext) !void {
            try ctx.set("mw2", "1");
        }
    }.run;

    var chain = MiddlewareChain.init(allocator);
    defer chain.deinit();

    try chain.use(middleware1);
    try chain.use(middleware2);

    try std.testing.expectEqual(@as(usize, 2), chain.len());

    // Create context
    var request = makeTestRequest(allocator, "/");
    defer request.deinit();

    var response = server.ResponseBuilder.init(allocator);
    defer response.deinit();

    var ctx = MiddlewareContext.init(allocator, &request, &response);
    defer ctx.deinit();

    try chain.execute(&ctx);
    try std.testing.expectEqualStrings("1", ctx.get("mw1").?);
    try std.testing.expectEqualStrings("1", ctx.get("mw2").?);
}

test "MiddlewareChain abort stops execution" {
    const allocator = std.testing.allocator;

    const abortMiddleware = struct {
        fn run(ctx: *MiddlewareContext) !void {
            try ctx.set("first", "ran");
            ctx.abort();
        }
    }.run;

    const neverRuns = struct {
        fn run(ctx: *MiddlewareContext) !void {
            try ctx.set("second", "ran");
        }
    }.run;

    var chain = MiddlewareChain.init(allocator);
    defer chain.deinit();

    try chain.use(abortMiddleware);
    try chain.use(neverRuns);

    var request = makeTestRequest(allocator, "/");
    defer request.deinit();

    var response = server.ResponseBuilder.init(allocator);
    defer response.deinit();

    var ctx = MiddlewareContext.init(allocator, &request, &response);
    defer ctx.deinit();

    try chain.execute(&ctx);

    try std.testing.expectEqualStrings("ran", ctx.get("first").?);
    try std.testing.expect(ctx.get("second") == null);
}
