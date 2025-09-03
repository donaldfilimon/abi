//! Discord Plugin (Example/Skeleton)
//!
//! This plugin implements the standard Abi plugin interface and exposes a small
//! C-compatible API for Discord operations. It is designed to compile without
//! external dependencies while leaving clear extension points to integrate a
//! real Discord client (e.g. discord.zig, zCord, or concord-zig).

const std = @import("std");
const abi = @import("abi");
const plugins = abi.plugins;
const http = std.http;

// =============================================================================
// TYPE DEFINITIONS AND IMPORTS
// =============================================================================

/// PluginInfo provides metadata about the plugin, including its name, version,
/// author, description, and capabilities. This information is used by the host
/// application to identify, load, and manage the plugin.
const PluginInfo = plugins.types.PluginInfo;

/// PluginVersion represents the semantic version of the plugin using major,
/// minor, and patch numbers. It enables compatibility checking and update
/// management between the plugin and host application.
const PluginVersion = plugins.types.PluginVersion;

/// PluginType enumerates the different categories of plugins that can be
/// implemented, such as protocol handlers, data processors, or UI extensions.
/// This helps the host categorize and route plugin functionality appropriately.
const PluginType = plugins.types.PluginType;

/// PluginConfig holds configuration parameters for the plugin as key-value
/// pairs, allowing runtime customization of plugin behavior without requiring
/// recompilation or plugin restart.
const PluginConfig = plugins.types.PluginConfig;

/// PluginContext provides the execution context for the plugin, including
/// references to the host application's allocator, logging facilities, and
/// other shared resources needed during plugin operation.
const PluginContext = plugins.types.PluginContext;

/// PluginInterface defines the standardized function table that all plugins
/// must implement to be compatible with the host application. This includes
/// lifecycle management, configuration, status reporting, and event handling.
const PluginInterface = plugins.interface.PluginInterface;

/// PLUGIN_ABI_VERSION specifies the Application Binary Interface (ABI) version
/// that this plugin was compiled against, ensuring binary compatibility
/// between the plugin and host application at runtime.
const PLUGIN_ABI_VERSION = plugins.interface.PLUGIN_ABI_VERSION;

// =============================================================================
// PLUGIN STATE MANAGEMENT
// =============================================================================

/// DiscordState maintains the complete internal state of the Discord plugin,
/// including connection status, authentication credentials, configuration
/// parameters, and active components for Discord Gateway and REST API
/// interactions. This centralized state ensures thread-safe access and
/// proper resource management throughout the plugin's lifecycle.
const DiscordState = struct {
    /// Indicates whether the plugin has completed its initialization phase
    /// and is ready for configuration and startup operations.
    initialized: bool = false,

    /// Indicates whether the plugin is actively running and processing
    /// Discord events and API requests.
    running: bool = false,

    /// Running counter of messages successfully sent through this plugin
    /// instance, used for metrics and debugging purposes.
    messages_sent: u64 = 0,

    /// Discord bot authentication token as an owned string slice. This is
    /// stored securely and used for all authenticated API requests.
    token: ?[]u8 = null,

    /// Discord Gateway intents bitfield specifying which events the bot
    /// should receive. Controls the scope of real-time event subscriptions.
    intents: u32 = 0,

    /// Discord application ID for this bot, required for slash command
    /// registration and other application-specific API operations.
    application_id: ?[]u8 = null,

    /// Optional guild ID for guild-specific operations such as registering
    /// guild commands or managing guild-specific bot behavior.
    guild_id: ?[]u8 = null,

    /// Background thread handle for the Discord Gateway WebSocket connection,
    /// responsible for maintaining real-time connectivity with Discord.
    gateway_thread: ?std.Thread = null,

    /// Atomic flag to signal the gateway thread to stop processing and
    /// gracefully shut down the WebSocket connection.
    gateway_stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    /// Discord REST API client instance for making HTTP requests to Discord's
    /// REST endpoints, including rate limiting and error handling.
    rest: ?*DiscordRestClient = null,

    /// Releases all allocated resources and resets the state to its initial
    /// configuration. Should be called during plugin shutdown to prevent
    /// memory leaks and ensure clean termination.
    ///
    /// - Parameters:
    ///   - allocator: The allocator used to free owned string resources.
    pub fn deinit(self: *DiscordState, allocator: std.mem.Allocator) void {
        if (self.token) |t| allocator.free(t);
        if (self.application_id) |v| allocator.free(v);
        if (self.guild_id) |v| allocator.free(v);
        if (self.rest) |r| {
            r.deinit();
            allocator.destroy(r);
        }
        self.* = .{};
    }
};

/// Global plugin state instance. This singleton maintains all plugin state
/// across the plugin's lifecycle and provides thread-safe access patterns.
var g_state: DiscordState = .{};

/// Configuration key for the Discord bot token parameter. The host application
/// should provide this value through the plugin configuration system.
const CONFIG_KEY_TOKEN = "DISCORD_TOKEN";

/// Configuration key for Discord Gateway intents as a comma-separated string
/// or decimal value. Controls which events the bot receives from Discord.
const CONFIG_KEY_INTENTS = "DISCORD_INTENTS";

/// Configuration key for the Discord application ID. Required for slash
/// command registration and other application-specific operations.
const CONFIG_KEY_APPLICATION_ID = "DISCORD_APPLICATION_ID";

/// Configuration key for an optional guild ID. When provided, enables
/// guild-specific command registration and operations.
const CONFIG_KEY_GUILD_ID = "DISCORD_GUILD_ID";

// =============================================================================
// DISCORD PUBLIC API (C-COMPATIBLE INTERFACE)
// =============================================================================

/// DiscordApi defines a C-compatible function table that external applications
/// can use to interact with Discord through this plugin. The host application
/// obtains this interface via the get_api("discord_v1") call and casts the
/// returned pointer to this struct type for direct function access.
const DiscordApi = extern struct {
    /// Sends a text message to a specified Discord channel.
    ///
    /// - Parameters:
    ///   - channel_id: The Discord channel ID as a 64-bit unsigned integer.
    ///   - content: The message content as a null-terminated UTF-8 string.
    /// - Returns: 0 on success, or a negative error code on failure.
    send_message: *const fn (channel_id: u64, content: [*:0]const u8) callconv(.c) c_int,

    /// Updates the bot's presence status text visible to other Discord users.
    ///
    /// - Parameters:
    ///   - status: The new status text as a null-terminated UTF-8 string.
    /// - Returns: 0 on success, or a negative error code on failure.
    set_presence: *const fn (status: [*:0]const u8) callconv(.c) c_int,

    /// Performs a connectivity check to verify the Discord connection is active.
    ///
    /// - Returns: 0 if connected and responsive, or a negative error code if not.
    ping: *const fn () callconv(.c) c_int,
};

/// Implementation of the `send_message` function in the DiscordApi interface.
///
/// Attempts to send a message to the specified Discord channel using the REST API.
/// If the plugin is not running or lacks proper configuration, it falls back to
/// logging the message locally while still incrementing the sent message counter.
///
/// - Parameters:
///   - channel_id: The target Discord channel ID as a 64-bit unsigned integer.
///   - content: The message content as a null-terminated UTF-8 string.
/// - Returns: 0 on success, or a negative error code indicating the failure reason.
fn apiSendMessage(channel_id: u64, content: [*:0]const u8) callconv(.c) c_int {
    const msg = std.mem.span(content);
    if (!g_state.running or g_state.rest == null or g_state.token == null) {
        std.log.warn("[discord_plugin] send_message called while plugin not running; falling back to log", .{});
        g_state.messages_sent += 1;
        std.log.info("[discord_plugin] send_message(log): {s}", .{msg});
        return 0;
    }

    // POST /channels/{channel_id}/messages
    var body_allocator = std.heap.page_allocator;
    const json_body = buildJsonContentBody(body_allocator, msg) catch |err| {
        std.log.err("[discord_plugin] failed to build JSON body: {}", .{err});
        return -1;
    };
    defer body_allocator.free(json_body);

    const rest = g_state.rest.?;
    const path = std.fmt.allocPrint(body_allocator, "/channels/{d}/messages", .{channel_id}) catch {
        return -1;
    };
    defer body_allocator.free(path);

    const res = rest.request(.POST, path, json_body, "/channels/:id/messages") catch |err| {
        std.log.err("[discord_plugin] REST error: {}", .{err});
        return -1;
    };
    defer body_allocator.free(res);

    g_state.messages_sent += 1;
    std.log.info("[discord_plugin] send_message: status OK, bytes={d}", .{res.len});
    return 0;
}

/// Implementation of the `set_presence` function in the DiscordApi interface.
///
/// Updates the bot's Discord presence status text. This is a placeholder
/// implementation that logs the status change locally. A full implementation
/// would send the presence update through the Discord Gateway connection.
///
/// - Parameters:
///   - status: The new status text as a null-terminated UTF-8 string.
/// - Returns: 0 on success, -1 if the plugin is not running.
fn apiSetPresence(status: [*:0]const u8) callconv(.c) c_int {
    if (!g_state.running) return -1;
    const text = std.mem.span(status);
    std.log.info("[discord_plugin] set_presence: {s}", .{text});
    return 0;
}

/// Implementation of the `ping` function in the DiscordApi interface.
///
/// Performs a simple connectivity and health check. Currently returns success
/// if the plugin is initialized. A full implementation might check Gateway
/// heartbeat latency or REST API responsiveness.
///
/// - Returns: 0 if the plugin is initialized and healthy, -1 otherwise.
fn apiPing() callconv(.c) c_int {
    // Could check gateway heartbeat latency; here we return OK if initialized
    return if (g_state.initialized) 0 else -1;
}

/// Static instance of the DiscordApi function table, populated with function
/// pointers to the actual implementation functions. This is returned by the
/// getApi() function when requested by the host application.
var DISCORD_API: DiscordApi = .{
    .send_message = apiSendMessage,
    .set_presence = apiSetPresence,
    .ping = apiPing,
};

// =============================================================================
// STRUCTURED LOGGING UTILITIES
// =============================================================================

/// Log provides a structured logging interface with different severity levels.
/// All log messages are prefixed with [DISCORD] and the appropriate level
/// indicator for easy identification in log aggregation systems.
pub const Log = struct {
    /// Logs an informational message for normal operational events.
    ///
    /// - Parameters:
    ///   - fmt: The format string for the log message.
    ///   - args: The arguments to be formatted into the message.
    pub fn info(comptime fmt: []const u8, args: anytype) void {
        std.debug.print("[DISCORD][INFO] " ++ fmt ++ "\n", args);
    }

    /// Logs a warning message for potentially problematic conditions.
    ///
    /// - Parameters:
    ///   - fmt: The format string for the log message.
    ///   - args: The arguments to be formatted into the message.
    pub fn warn(comptime fmt: []const u8, args: anytype) void {
        std.debug.print("[DISCORD][WARN] " ++ fmt ++ "\n", args);
    }

    /// Logs an error message for serious problems that require attention.
    ///
    /// - Parameters:
    ///   - fmt: The format string for the log message.
    ///   - args: The arguments to be formatted into the message.
    pub fn err(comptime fmt: []const u8, args: anytype) void {
        std.debug.print("[DISCORD][ERROR] " ++ fmt ++ "\n", args);
    }
};

/// Constructs a JSON message body for Discord's message creation endpoint.
///
/// This is a minimal JSON builder that creates a properly formatted message
/// payload. A production implementation should include proper JSON escaping
/// and support for additional message properties like embeds or attachments.
///
/// - Parameters:
///   - allocator: The allocator to use for the returned JSON string.
///   - content: The message content string to be JSON-encoded.
/// - Returns: An allocated JSON string ready for HTTP transmission.
/// - Errors: Returns allocation errors if memory allocation fails.
pub fn buildJsonContentBody(allocator: std.mem.Allocator, content: []const u8) ![]u8 {
    // Note: Minimal builder without escaping for skeleton purposes
    return std.fmt.allocPrint(allocator, "{{\"content\":\"{s}\"}}", .{content});
}

// =============================================================================
// WEBSOCKET TRANSPORT WITH HEARTBEATS AND RECONNECTION
// =============================================================================

/// WebSocketTransport manages the Discord Gateway WebSocket connection with
/// automatic reconnection, heartbeat management, and session resumption.
pub const WebSocketTransport = struct {
    allocator: std.mem.Allocator,
    connected: bool = false,
    connection_state: ConnectionState = .disconnected,
    heartbeat_interval_ms: u32 = GATEWAY_CONFIG.DEFAULT_HEARTBEAT_MS,
    last_sequence: ?u64 = null,
    stop_flag: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    last_heartbeat_ack: std.atomic.Value(i64) = std.atomic.Value(i64).init(0),
    session_id: ?[]u8 = null,
    resume_gateway_url: ?[]u8 = null,
    reconnect_attempt: u32 = 0,
    max_reconnect_attempts: u32 = GATEWAY_CONFIG.MAX_RECONNECT_ATTEMPTS,
    last_close_code: ?DiscordCloseCode = null,
    should_resume: bool = false,

    /// Errors that can occur during reconnection attempts
    pub const ReconnectError = error{
        MaxAttemptsReached,
        ConnectionFailed,
        HeartbeatTimeout,
        FatalCloseCode,
        AuthenticationFailed,
    };

    /// Initialize a new WebSocket transport instance
    pub fn init(allocator: std.mem.Allocator) WebSocketTransport {
        return .{ .allocator = allocator };
    }

    /// Clean up resources and free allocated memory
    pub fn deinit(self: *WebSocketTransport) void {
        self.clearSession();
    }

    /// Establish connection to Discord Gateway with automatic reconnection
    pub fn connect(self: *WebSocketTransport, token: []const u8, intents: u32) !void {
        var attempt: u32 = 0;

        // Determine connection strategy based on previous close code
        try self.determineConnectionStrategy();

        while (attempt < self.max_reconnect_attempts) {
            if (attempt > 0) {
                const delay_ms = self.calculateBackoffDelay(attempt);
                Log.info("Reconnect attempt {d}/{d} after {d}ms (resume={})", .{ attempt + 1, self.max_reconnect_attempts, delay_ms, self.should_resume });
                self.sleepMs(delay_ms);
            }

            if (self.attemptConnection(token, intents)) {
                self.reconnect_attempt = 0;
                self.last_close_code = null; // Clear close code on successful connection
                return;
            } else |err| {
                Log.warn("Connection attempt {d} failed: {}", .{ attempt + 1, err });

                // Special handling for authentication failures
                if (err == ReconnectError.AuthenticationFailed) {
                    self.should_resume = false; // Force re-identify on auth failure
                    self.clearSession();
                }

                attempt += 1;
            }
        }
        return ReconnectError.MaxAttemptsReached;
    }

    /// Determine whether to resume or re-identify based on close code
    fn determineConnectionStrategy(self: *WebSocketTransport) !void {
        if (self.last_close_code) |close_code| {
            if (!close_code.shouldReconnect()) {
                Log.err("Fatal close code {d}, cannot reconnect", .{@intFromEnum(close_code)});
                return ReconnectError.FatalCloseCode;
            }
            self.should_resume = close_code.shouldResume() and self.session_id != null;
            Log.info("Close code {d}: should_resume={}", .{ @intFromEnum(close_code), self.should_resume });
        }
    }

    /// Attempt to establish a WebSocket connection
    fn attemptConnection(self: *WebSocketTransport, _token: []const u8, _intents: u32) !void {
        // NOTE: This is still scaffolding. In a real implementation:
        // 1. Open WebSocket to wss://gateway.discord.gg?v=10&encoding=json or resume_gateway_url
        // 2. Perform WebSocket handshake
        // 3. Wait for HELLO opcode 10, extract heartbeat_interval
        // 4. Start heartbeat timer
        // 5. Send IDENTIFY or RESUME based on session state

        _ = _token;
        _ = _intents;

        self.connection_state = .connecting;
        self.connected = true;
        self.last_heartbeat_ack.store(std.time.milliTimestamp(), .release);

        const gateway_url = if (self.resume_gateway_url) |url| url else "wss://gateway.discord.gg?v=10&encoding=json";
        Log.info("Gateway connecting to: {s} (scaffold)", .{gateway_url});

        // Simulate receiving HELLO with heartbeat interval
        self.heartbeat_interval_ms = GATEWAY_CONFIG.DEFAULT_HEARTBEAT_MS; // Would come from HELLO payload
        Log.info("HELLO received: heartbeat_interval={d}ms", .{self.heartbeat_interval_ms});

        self.connection_state = .connected;
    }

    /// Send IDENTIFY or RESUME payload based on session state
    pub fn sendIdentify(self: *WebSocketTransport, token: []const u8, intents: u32) !void {
        if (self.should_resume and self.session_id != null) {
            try self.sendResume(token);
        } else {
            try self.sendNewIdentify(token, intents);
        }
    }

    /// Send RESUME payload for existing session
    fn sendResume(self: *WebSocketTransport, _token: []const u8) !void {
        _ = _token;

        self.connection_state = .resuming;
        Log.info("RESUME sent (session_id={s}, seq={?d})", .{ self.session_id.?[0..@min(8, self.session_id.?.len)], self.last_sequence });

        // In real implementation: send resume frame
        // {
        //   "op": 6,
        //   "d": {
        //     "token": "token",
        //     "session_id": "session_id_here",
        //     "seq": 1337
        //   }
        // }

        // Simulate successful resume
        Log.info("RESUMED successfully", .{});
        self.connection_state = .connected;
    }

    /// Send IDENTIFY payload for new session
    fn sendNewIdentify(self: *WebSocketTransport, token: []const u8, intents: u32) !void {
        self.connection_state = .identifying;
        Log.info("IDENTIFY sent (token len={d}, intents={d})", .{ token.len, intents });

        // In real implementation: send identify frame
        // {
        //   "op": 2,
        //   "d": {
        //     "token": "token",
        //     "intents": intents,
        //     "properties": {...}
        //   }
        // }

        // Simulate successful identify and store session info
        const session_id = try self.allocator.dupe(u8, "simulated_session_12345");
        if (self.session_id) |old_sid| {
            self.allocator.free(old_sid);
        }
        self.session_id = session_id;

        // Simulate receiving resume gateway URL
        const resume_url = try self.allocator.dupe(u8, "wss://gateway.discord.gg/?v=10&encoding=json");
        if (self.resume_gateway_url) |old_url| {
            self.allocator.free(old_url);
        }
        self.resume_gateway_url = resume_url;

        Log.info("IDENTIFY successful, session established", .{});
        self.connection_state = .connected;
    }

    /// Send heartbeat and check for timeout
    pub fn sendHeartbeat(self: *WebSocketTransport) !void {
        const now = std.time.milliTimestamp();
        const last_ack = self.last_heartbeat_ack.load(.acquire);
        const timeout_threshold = @as(i64, @intCast(self.heartbeat_interval_ms * GATEWAY_CONFIG.HEARTBEAT_TIMEOUT_MULTIPLIER));

        if (now - last_ack > timeout_threshold) {
            Log.warn("Heartbeat timeout detected (last_ack={d}ms ago)", .{now - last_ack});
            return ReconnectError.HeartbeatTimeout;
        }

        // In real implementation: send heartbeat frame
        // {
        //   "op": 1,
        //   "d": sequence_number or null
        // }

        Log.info("Heartbeat sent (seq={?d})", .{self.last_sequence});
    }

    /// Main read loop for processing incoming WebSocket frames
    pub fn readLoop(self: *WebSocketTransport) !void {
        var event_counter: u32 = 0;

        while (self.connected and !self.stop_flag.load(.acquire)) {
            // Simulate receiving different event types
            const event_type = event_counter % 4;
            switch (event_type) {
                0 => {
                    // HEARTBEAT_ACK
                    self.last_heartbeat_ack.store(std.time.milliTimestamp(), .release);
                    Log.info("HEARTBEAT_ACK received", .{});
                },
                1 => {
                    // GUILD_CREATE
                    Log.info("GUILD_CREATE event received", .{});
                },
                2 => {
                    // MESSAGE_CREATE
                    Log.info("MESSAGE_CREATE event received", .{});
                },
                3 => {
                    // PRESENCE_UPDATE
                    Log.info("PRESENCE_UPDATE event received", .{});
                },
                else => unreachable,
            }

            event_counter += 1;
            self.sleepMs(1000); // Simulate processing time
        }
    }

    /// Calculate exponential backoff delay with jitter
    pub fn calculateBackoffDelay(_: *WebSocketTransport, attempt: u32) u32 {
        // Calculate base delay with exponential growth, capped at maximum
        var base_delay: u32 = GATEWAY_CONFIG.BACKOFF_BASE_MS;

        // Apply exponential growth for each attempt, but cap at maximum
        var i: u32 = 0;
        while (i < attempt and base_delay < GATEWAY_CONFIG.BACKOFF_MAX_MS / 2) : (i += 1) {
            base_delay = @min(base_delay * 2, GATEWAY_CONFIG.BACKOFF_MAX_MS);
        }

        // Ensure we don't exceed the maximum
        base_delay = @min(base_delay, GATEWAY_CONFIG.BACKOFF_MAX_MS);

        // Add jitter to prevent thundering herd (25% of base delay)
        const jitter_range = base_delay / 4;
        const timestamp = std.time.milliTimestamp();
        const jitter = @as(u32, @intCast(@abs(timestamp) % (jitter_range * 2)));

        return @min(base_delay + jitter, GATEWAY_CONFIG.BACKOFF_MAX_MS);
    }

    /// Sleep for specified milliseconds using spin-wait
    fn sleepMs(_: *WebSocketTransport, ms: u32) void {
        const start = std.time.milliTimestamp();
        const target = start + @as(i64, @intCast(ms));

        while (std.time.milliTimestamp() < target) {
            std.atomic.spinLoopHint();
        }
    }

    /// Handle WebSocket close codes with appropriate actions
    pub fn handleCloseCode(self: *WebSocketTransport, close_code: u16) void {
        const discord_code = @as(DiscordCloseCode, @enumFromInt(close_code));
        self.last_close_code = discord_code;
        self.connection_state = .disconnected;
        self.connected = false;

        Log.warn("WebSocket closed with code {d}: {s}", .{ close_code, @tagName(discord_code) });

        switch (discord_code) {
            .authentication_failed => {
                Log.err("Authentication failed - check bot token", .{});
                self.clearSession();
            },
            .invalid_intents => {
                Log.err("Invalid intents - check privileged intent permissions", .{});
            },
            .disallowed_intents => {
                Log.err("Disallowed intents - bot may lack permissions", .{});
            },
            .session_timed_out => {
                Log.warn("Session timed out - will attempt resume if possible", .{});
            },
            .rate_limited => {
                Log.warn("Rate limited by Discord gateway", .{});
            },
            .invalid_seq => {
                Log.warn("Invalid sequence number - session may be corrupted", .{});
            },
            else => {
                Log.info("Gateway closed: {s}", .{@tagName(discord_code)});
            },
        }
    }

    /// Clear session information and reset resume state
    pub fn clearSession(self: *WebSocketTransport) void {
        if (self.session_id) |sid| {
            self.allocator.free(sid);
            self.session_id = null;
        }
        if (self.resume_gateway_url) |url| {
            self.allocator.free(url);
            self.resume_gateway_url = null;
        }
        self.last_sequence = null;
        self.should_resume = false;
        Log.info("Session cleared", .{});
    }

    /// Simulate receiving a close code for testing
    pub fn simulateCloseCode(self: *WebSocketTransport, close_code: DiscordCloseCode) void {
        self.handleCloseCode(@intFromEnum(close_code));
    }
};

// =============================================================================
// REST CLIENT WITH RATE LIMITING AND ERROR HANDLING
// =============================================================================

/// RouteBucket tracks rate limiting information for a specific Discord API
/// route pattern, including remaining requests and reset timestamps.
const RouteBucket = struct {
    /// Number of requests remaining in the current rate limit window.
    remaining: i32 = 1,
    /// Timestamp in milliseconds when the rate limit window resets.
    reset_at_ms: i64 = 0,
};

/// DiscordRestClient provides a comprehensive HTTP client for Discord's REST API
/// with built-in rate limiting, error handling, and request routing.
pub const DiscordRestClient = struct {
    allocator: std.mem.Allocator,
    client: std.http.Client,
    token: []const u8,
    route_buckets: std.StringHashMap(RouteBucket),
    global_reset_at_ms: i64 = 0,

    /// REST API specific error types
    pub const RestError = error{
        HttpConflict,
        RateLimited,
        Unauthorized,
        NotFound,
        InternalServerError,
    } || std.http.Client.RequestError || std.mem.Allocator.Error;

    /// Initialize a new REST client instance
    pub fn init(allocator: std.mem.Allocator, token: []const u8) !*DiscordRestClient {
        const self = try allocator.create(DiscordRestClient);
        self.* = .{
            .allocator = allocator,
            .client = .{ .allocator = allocator },
            .token = token,
            .route_buckets = std.StringHashMap(RouteBucket).init(allocator),
            .global_reset_at_ms = 0,
        };
        return self;
    }

    /// Clean up resources and free allocated memory
    pub fn deinit(self: *DiscordRestClient) void {
        self.client.deinit();
        self.route_buckets.deinit();
        self.allocator.destroy(self);
    }

    /// Get current timestamp in milliseconds
    fn nowMs() i64 {
        return @as(i64, @intCast(std.time.milliTimestamp()));
    }

    /// Sleep until specified deadline
    fn sleepUntil(deadline_ms: i64) void {
        const now = nowMs();
        if (deadline_ms > now) {
            const end = deadline_ms;
            while (nowMs() < end) {
                std.atomic.spinLoopHint();
            }
        }
    }

    /// Wait for rate limit bucket availability
    fn waitForBucket(self: *DiscordRestClient, route_key: []const u8) !void {
        // Check global rate limit first
        if (self.global_reset_at_ms > nowMs()) {
            sleepUntil(self.global_reset_at_ms);
        }

        // Check route-specific rate limit
        const entry = self.route_buckets.getPtr(route_key);
        if (entry) |bucket| {
            if (bucket.reset_at_ms > nowMs() and bucket.remaining <= 0) {
                sleepUntil(bucket.reset_at_ms);
            }
        }
    }

    /// Update rate limit information from response headers
    fn updateFromHeaders(self: *DiscordRestClient, route_key: []const u8, _: anytype) void {
        // TODO: Implement proper header parsing when the correct API is determined
        // For now, we'll skip rate limit header parsing to get the basic functionality working

        var bucket = self.route_buckets.getPtr(route_key);
        if (bucket == null) {
            const inserted = self.route_buckets.put(
                self.allocator.dupe(u8, route_key) catch return,
                .{},
            ) catch return;
            _ = inserted;
            bucket = self.route_buckets.getPtr(route_key);
            if (bucket == null) return;
        }

        // Simplified rate limiting - just use basic defaults
        bucket.?.remaining = @max(bucket.?.remaining - 1, 0);
        if (bucket.?.remaining <= 0) {
            bucket.?.reset_at_ms = nowMs() + 1000; // 1 second cooldown
        }
    }

    /// Make a generic HTTP request with rate limiting and error handling
    pub fn request(self: *DiscordRestClient, method: std.http.Method, path: []const u8, body: ?[]const u8, route_key: []const u8) ![]u8 {
        // Wait for rate limit availability
        try self.waitForBucket(route_key);

        const base = "https://discord.com/api/v10";
        const url = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ base, path });
        defer self.allocator.free(url);

        // Prepare headers
        var headers_buf: [3]std.http.Header = undefined;
        var headers: []std.http.Header = &headers_buf;
        var header_count: usize = 0;

        // Authorization header
        var auth_buf: [512]u8 = undefined;
        const auth_val = std.fmt.bufPrint(&auth_buf, "Bot {s}", .{self.token}) catch unreachable;
        headers[header_count] = .{ .name = "authorization", .value = auth_val };
        header_count += 1;

        // User-Agent header
        headers[header_count] = .{ .name = "user-agent", .value = "abi-discord-plugin (https://github.com/donaldfilimon/abi, 1.0.0)" };
        header_count += 1;

        // Content-Type header for requests with body
        if (body != null) {
            headers[header_count] = .{ .name = "content-type", .value = "application/json" };
            header_count += 1;
        }

        headers = headers[0..header_count];

        // Create and send request
        var req = try self.client.request(method, try std.Uri.parse(url), .{
            .headers = .{},
            .extra_headers = headers,
        });
        defer req.deinit();

        // Send body if provided
        if (body) |b| {
            try req.sendBodyComplete(@constCast(b));
        } else {
            try req.sendBodiless();
        }

        // Wait for response
        var response = try req.receiveHead(&.{});

        // Check response status and handle errors
        const status = response.head.status;
        switch (status.class()) {
            .success => {},
            .client_error => {
                switch (status) {
                    .unauthorized => return RestError.Unauthorized,
                    .not_found => return RestError.NotFound,
                    .conflict => return RestError.HttpConflict,
                    else => return RestError.HttpConflict,
                }
            },
            .server_error => {
                if (status == .too_many_requests) {
                    return RestError.RateLimited;
                }
                return RestError.InternalServerError;
            },
            else => return RestError.InternalServerError,
        }

        // Update rate limit information from response headers
        self.updateFromHeaders(route_key, &response.head);

        // Read response body
        var list = std.ArrayListUnmanaged(u8){};
        defer list.deinit(self.allocator);

        var transfer_buf: [4096]u8 = undefined;
        const reader = response.reader(&transfer_buf);
        try reader.appendRemainingUnlimited(self.allocator, &list);

        return try list.toOwnedSlice(self.allocator);
    }

    /// Convenience methods for common HTTP methods
    pub fn get(self: *DiscordRestClient, path: []const u8, route_key: []const u8) ![]u8 {
        return self.request(.GET, path, null, route_key);
    }

    pub fn post(self: *DiscordRestClient, path: []const u8, body: []const u8, route_key: []const u8) ![]u8 {
        return self.request(.POST, path, body, route_key);
    }

    pub fn patch(self: *DiscordRestClient, path: []const u8, body: []const u8, route_key: []const u8) ![]u8 {
        return self.request(.PATCH, path, body, route_key);
    }

    pub fn delete(self: *DiscordRestClient, path: []const u8, route_key: []const u8) ![]u8 {
        return self.request(.DELETE, path, null, route_key);
    }

    /// Create a new slash command for a specific guild
    pub fn createGuildCommand(self: *DiscordRestClient, app_id: []const u8, guild_id: []const u8, command: EnhancedCommandSpec) !void {
        const path = try std.fmt.allocPrint(self.allocator, "/applications/{s}/guilds/{s}/commands", .{ app_id, guild_id });
        defer self.allocator.free(path);

        const body = try command.toJson(self.allocator);
        defer self.allocator.free(body);

        _ = try self.post(path, body, "/applications/:id/guilds/:id/commands");
    }

    /// Create a new global slash command
    pub fn createGlobalCommand(self: *DiscordRestClient, app_id: []const u8, command: EnhancedCommandSpec) !void {
        const path = try std.fmt.allocPrint(self.allocator, "/applications/{s}/commands", .{app_id});
        defer self.allocator.free(path);

        const body = try command.toJson(self.allocator);
        defer self.allocator.free(body);

        _ = try self.post(path, body, "/applications/:id/commands");
    }

    /// Get all global commands for the application
    pub fn getGlobalCommands(self: *DiscordRestClient, app_id: []const u8) ![]u8 {
        const path = try std.fmt.allocPrint(self.allocator, "/applications/{s}/commands", .{app_id});
        defer self.allocator.free(path);

        return try self.get(path, "/applications/:id/commands");
    }

    /// Delete a global command by ID
    pub fn deleteGlobalCommand(self: *DiscordRestClient, app_id: []const u8, command_id: []const u8) !void {
        const path = try std.fmt.allocPrint(self.allocator, "/applications/{s}/commands/{s}", .{ app_id, command_id });
        defer self.allocator.free(path);

        _ = try self.delete(path, "/applications/:id/commands/:id");
    }

    /// Get all commands for a specific guild
    pub fn getGuildCommands(self: *DiscordRestClient, app_id: []const u8, guild_id: []const u8) ![]u8 {
        const path = try std.fmt.allocPrint(self.allocator, "/applications/{s}/guilds/{s}/commands", .{ app_id, guild_id });
        defer self.allocator.free(path);

        return try self.get(path, "/applications/:id/guilds/:id/commands");
    }

    /// Delete a guild command by ID
    pub fn deleteGuildCommand(self: *DiscordRestClient, app_id: []const u8, guild_id: []const u8, command_id: []const u8) !void {
        const path = try std.fmt.allocPrint(self.allocator, "/applications/{s}/guilds/{s}/commands/{s}", .{ app_id, guild_id, command_id });
        defer self.allocator.free(path);

        _ = try self.delete(path, "/applications/:id/guilds/:id/commands/:id");
    }

    /// Idempotent command upsert for both global and guild commands
    pub fn upsertCommands(self: *DiscordRestClient, app_id: []const u8, guild_id: ?[]const u8, desired_commands: []const EnhancedCommandSpec) !void {
        const scope = if (guild_id) |gid|
            std.fmt.allocPrint(self.allocator, "guild {s}", .{gid[0..@min(8, gid.len)]}) catch "guild"
        else
            "global";
        defer if (guild_id != null) self.allocator.free(scope);

        Log.info("Starting idempotent command upsert for {d} {s} commands", .{ desired_commands.len, scope });

        var created_count: u32 = 0;
        var skipped_count: u32 = 0;
        var error_count: u32 = 0;

        for (desired_commands) |cmd| {
            Log.info("Processing {s} command: {s} (global={})", .{ scope, cmd.name, cmd.global });

            // Skip if command scope doesn't match
            if (cmd.global and guild_id != null) {
                Log.info("Skipping global command {s} in guild context", .{cmd.name});
                skipped_count += 1;
                continue;
            }
            if (!cmd.global and guild_id == null) {
                Log.info("Skipping guild command {s} in global context", .{cmd.name});
                skipped_count += 1;
                continue;
            }

            const create_result = if (guild_id) |gid|
                self.createGuildCommand(app_id, gid, cmd)
            else
                self.createGlobalCommand(app_id, cmd);

            create_result catch |err| {
                switch (err) {
                    RestError.HttpConflict => {
                        Log.info("Command {s} already exists (409 Conflict), skipping", .{cmd.name});
                        skipped_count += 1;
                    },
                    else => {
                        Log.warn("Failed to create command {s}: {} (continuing)", .{ cmd.name, err });
                        error_count += 1;
                    },
                }
                continue;
            };

            created_count += 1;
            Log.info("Successfully created {s} command: {s}", .{ scope, cmd.name });
        }

        Log.info("{s} command upsert complete: {d} created, {d} skipped, {d} errors", .{ scope, created_count, skipped_count, error_count });
    }

    /// Legacy method for backward compatibility
    pub fn upsertGuildCommands(self: *DiscordRestClient, app_id: []const u8, guild_id: []const u8, desired_commands: []const EnhancedCommandSpec) !void {
        return self.upsertCommands(app_id, guild_id, desired_commands);
    }
};

// =============================================================================
// DISCORD PROTOCOL DEFINITIONS
// =============================================================================

/// DiscordCloseCode enumerates all possible WebSocket close codes that can be
/// received from Discord's Gateway, including both standard WebSocket codes
/// and Discord-specific codes. Each code indicates a different reason for
/// connection termination and determines the appropriate reconnection strategy.
pub const DiscordCloseCode = enum(u16) {
    // Standard WebSocket close codes (RFC 6455)
    normal_closure = 1000,
    going_away = 1001,
    protocol_error = 1002,
    unsupported_data = 1003,
    no_status_received = 1005,
    abnormal_closure = 1006,
    invalid_frame_payload_data = 1007,
    policy_violation = 1008,
    message_too_big = 1009,
    mandatory_extension = 1010,
    internal_error = 1011,
    service_restart = 1012,
    try_again_later = 1013,
    bad_gateway = 1014,
    tls_handshake = 1015,

    // Discord-specific close codes (4000-4014)
    unknown_error = 4000,
    unknown_opcode = 4001,
    decode_error = 4002,
    not_authenticated = 4003,
    authentication_failed = 4004,
    already_authenticated = 4005,
    invalid_seq = 4007,
    rate_limited = 4008,
    session_timed_out = 4009,
    invalid_shard = 4010,
    sharding_required = 4011,
    invalid_api_version = 4012,
    invalid_intents = 4013,
    disallowed_intents = 4014,

    /// Determines whether automatic reconnection should be attempted after
    /// receiving this close code. Some codes indicate permanent failures
    /// that require manual intervention rather than automatic retry.
    ///
    /// - Returns: true if reconnection should be attempted, false otherwise.
    pub fn shouldReconnect(self: DiscordCloseCode) bool {
        return switch (self) {
            // Reconnectable close codes
            .normal_closure, .going_away, .internal_error, .service_restart, .try_again_later, .unknown_error, .rate_limited, .session_timed_out => true,
            // Non-reconnectable close codes (require re-identify)
            .authentication_failed, .invalid_api_version, .invalid_intents, .disallowed_intents, .sharding_required => false,
            // Should resume (not reconnect)
            .unknown_opcode, .decode_error, .invalid_seq => true,
            // Fatal errors
            .not_authenticated, .already_authenticated, .invalid_shard => false,
            else => false,
        };
    }

    /// Determines whether session resumption should be attempted rather than
    /// a full re-identification. Resume is appropriate for temporary issues
    /// that don't invalidate the existing session.
    ///
    /// - Returns: true if session resumption should be attempted, false otherwise.
    pub fn shouldResume(self: DiscordCloseCode) bool {
        return switch (self) {
            .unknown_error, .unknown_opcode, .decode_error, .rate_limited => true,
            else => false,
        };
    }
};

/// ConnectionState tracks the current state of the Discord Gateway connection
/// throughout its lifecycle, from initial connection through various operational
/// states and potential error conditions.
pub const ConnectionState = enum {
    /// No active connection to Discord Gateway.
    disconnected,
    /// Attempting to establish WebSocket connection.
    connecting,
    /// Connected but sending IDENTIFY payload.
    identifying,
    /// Connected and sending RESUME payload.
    resuming,
    /// Fully connected and receiving events.
    connected,
    /// Attempting to reconnect after connection loss.
    reconnecting,
    /// Permanent failure requiring manual intervention.
    fatal_error,
};

// =============================================================================
// ENHANCED SLASH COMMAND SYSTEM
// =============================================================================

/// CommandOptionType enumerates all possible types for Discord slash command
/// options, including primitive types, Discord entities, and structural types
/// like subcommands and subcommand groups.
pub const CommandOptionType = enum(u8) {
    sub_command = 1,
    sub_command_group = 2,
    string = 3,
    integer = 4,
    boolean = 5,
    user = 6,
    channel = 7,
    role = 8,
    mentionable = 9,
    number = 10,
    attachment = 11,
};

/// CommandOption defines a single option for a Discord slash command, including
/// its type, requirements, possible choices, and nested options for subcommands.
/// This provides a complete specification for complex command structures.
pub const CommandOption = struct {
    /// The name of the option as it appears in Discord's interface.
    name: []const u8,
    /// Human-readable description of the option's purpose.
    description: []const u8,
    /// The data type of this option (string, integer, user, etc.).
    option_type: CommandOptionType,
    /// Whether this option must be provided by the user.
    required: bool = false,
    /// Predefined choices for this option (optional).
    choices: ?[]const CommandChoice = null,
    /// Nested options for subcommands and subcommand groups.
    options: ?[]const CommandOption = null,

    /// Converts the command option to its JSON representation for Discord's API.
    ///
    /// - Parameters:
    ///   - allocator: The allocator to use for the JSON string.
    /// - Returns: An allocated JSON string representing this option.
    /// - Errors: Returns allocation errors if JSON construction fails.
    pub fn toJson(self: CommandOption, allocator: std.mem.Allocator) ![]u8 {
        var json = std.ArrayListUnmanaged(u8){};
        defer json.deinit(allocator);

        try json.appendSlice(allocator, "{");
        try json.appendSlice(allocator, "\"name\":\"");
        try json.appendSlice(allocator, self.name);
        try json.appendSlice(allocator, "\",\"description\":\"");
        try json.appendSlice(allocator, self.description);
        try json.appendSlice(allocator, "\",\"type\":");
        const type_str = try std.fmt.allocPrint(allocator, "{d}", .{@intFromEnum(self.option_type)});
        defer allocator.free(type_str);
        try json.appendSlice(allocator, type_str);

        if (self.required) {
            try json.appendSlice(allocator, ",\"required\":true");
        }

        if (self.choices) |choices| {
            try json.appendSlice(allocator, ",\"choices\":[");
            for (choices, 0..) |choice, i| {
                if (i > 0) try json.appendSlice(allocator, ",");
                const choice_json = try choice.toJson(allocator);
                defer allocator.free(choice_json);
                try json.appendSlice(allocator, choice_json);
            }
            try json.appendSlice(allocator, "]");
        }

        if (self.options) |options| {
            try json.appendSlice(allocator, ",\"options\":[");
            for (options, 0..) |option, i| {
                if (i > 0) try json.appendSlice(allocator, ",");
                const option_json = try option.toJson(allocator);
                defer allocator.free(option_json);
                try json.appendSlice(allocator, option_json);
            }
            try json.appendSlice(allocator, "]");
        }

        try json.appendSlice(allocator, "}");
        return try json.toOwnedSlice(allocator);
    }
};

/// CommandChoice represents a predefined choice for a command option, supporting
/// string, integer, and floating-point values as specified by Discord's API.
pub const CommandChoice = struct {
    /// The display name for this choice in Discord's interface.
    name: []const u8,
    /// The actual value that will be passed to the bot when selected.
    value: union(enum) {
        string: []const u8,
        integer: i64,
        number: f64,
    },

    /// Converts the command choice to its JSON representation for Discord's API.
    ///
    /// - Parameters:
    ///   - allocator: The allocator to use for the JSON string.
    /// - Returns: An allocated JSON string representing this choice.
    /// - Errors: Returns allocation errors if JSON construction fails.
    pub fn toJson(self: CommandChoice, allocator: std.mem.Allocator) ![]u8 {
        return switch (self.value) {
            .string => |str| try std.fmt.allocPrint(allocator, "{{\"name\":\"{s}\",\"value\":\"{s}\"}}", .{ self.name, str }),
            .integer => |int| try std.fmt.allocPrint(allocator, "{{\"name\":\"{s}\",\"value\":{d}}}", .{ self.name, int }),
            .number => |num| try std.fmt.allocPrint(allocator, "{{\"name\":\"{s}\",\"value\":{d}}}", .{ self.name, num }),
        };
    }
};

/// EnhancedCommandSpec provides a comprehensive specification for Discord slash
/// commands, including options, permissions, and deployment scope (global vs guild).
/// This supports the full range of Discord's command system capabilities.
pub const EnhancedCommandSpec = struct {
    /// The command name as it appears in Discord's slash command interface.
    name: []const u8,
    /// Human-readable description of the command's functionality.
    description: []const u8,
    /// Array of command options defining parameters and subcommands.
    options: ?[]const CommandOption = null,
    /// Whether this command should be registered globally or per-guild.
    global: bool = false,
    /// Whether this command can be used in direct messages.
    dm_permission: bool = true,
    /// Bitfield string specifying required permissions to use this command.
    default_member_permissions: ?[]const u8 = null,

    /// Converts the command specification to its JSON representation for Discord's API.
    ///
    /// - Parameters:
    ///   - allocator: The allocator to use for the JSON string.
    /// - Returns: An allocated JSON string representing this command.
    /// - Errors: Returns allocation errors if JSON construction fails.
    pub fn toJson(self: EnhancedCommandSpec, allocator: std.mem.Allocator) ![]u8 {
        var json = std.ArrayListUnmanaged(u8){};
        defer json.deinit(allocator);

        try json.appendSlice(allocator, "{");
        try json.appendSlice(allocator, "\"name\":\"");
        try json.appendSlice(allocator, self.name);
        try json.appendSlice(allocator, "\",\"description\":\"");
        try json.appendSlice(allocator, self.description);
        try json.appendSlice(allocator, "\"");

        if (!self.dm_permission) {
            try json.appendSlice(allocator, ",\"dm_permission\":false");
        }

        if (self.default_member_permissions) |perms| {
            try json.appendSlice(allocator, ",\"default_member_permissions\":\"");
            try json.appendSlice(allocator, perms);
            try json.appendSlice(allocator, "\"");
        }

        if (self.options) |options| {
            try json.appendSlice(allocator, ",\"options\":[");
            for (options, 0..) |option, i| {
                if (i > 0) try json.appendSlice(allocator, ",");
                const option_json = try option.toJson(allocator);
                defer allocator.free(option_json);
                try json.appendSlice(allocator, option_json);
            }
            try json.appendSlice(allocator, "]");
        }

        try json.appendSlice(allocator, "}");
        return try json.toOwnedSlice(allocator);
    }
};

/// CommandSpec provides a simplified command specification for backward
/// compatibility with existing code that doesn't require advanced features.
pub const CommandSpec = struct {
    /// The command name as it appears in Discord's interface.
    name: []const u8,
    /// Human-readable description of the command's functionality.
    description: []const u8,
};

// =============================================================================
// PLUGIN METADATA AND INTERFACE
// =============================================================================

/// Static plugin information structure that describes this Discord plugin
/// to the host application, including its capabilities, dependencies, and
/// contact information for support and updates.
const PLUGIN_INFO = PluginInfo{
    .name = "discord_integration",
    .version = PluginVersion.init(1, 0, 0),
    .author = "Abi AI Framework Team",
    .description = "Discord API integration plugin (skeleton) exposing a minimal Discord API.",
    .plugin_type = .protocol_handler,
    .abi_version = PLUGIN_ABI_VERSION,
    .provides = &[_][]const u8{"discord_v1"},
    .requires = &.{},
    .dependencies = &.{},
    .license = "MIT",
    .homepage = "https://github.com/donaldfilimon/abi",
    .repository = "https://github.com/donaldfilimon/abi",
};

// =============================================================================
// PLUGIN INTERFACE IMPLEMENTATION
// =============================================================================

/// Returns the static plugin information structure to the host application.
/// This function is called during plugin discovery and loading to determine
/// the plugin's capabilities and compatibility.
///
/// - Returns: A pointer to the static plugin information structure.
fn getInfo() callconv(.c) *const PluginInfo {
    return &PLUGIN_INFO;
}

/// Initializes the Discord plugin with the provided configuration and context.
/// Sets up internal state, validates configuration parameters, and prepares
/// the plugin for startup without actually connecting to Discord services.
///
/// - Parameters:
///   - context: The plugin execution context provided by the host.
/// - Returns: 0 on success, negative error codes on failure.
fn initPlugin(context: *PluginContext) callconv(.c) c_int {
    if (g_state.initialized) return -1;

    // Capture token from config if present; make an owned copy for the plugin
    if (context.config.getParameter(CONFIG_KEY_TOKEN)) |token_value| {
        // Accept either zero-terminated or slice; ensure copy without null
        const allocator = context.allocator;
        const owned = allocator.alloc(u8, token_value.len) catch return -2;
        @memcpy(owned, token_value);
        g_state.token = owned;
    } else {
        std.log.warn("[discord_plugin] No DISCORD_TOKEN provided. Running in stub mode.", .{});
    }

    if (context.config.getParameter(CONFIG_KEY_INTENTS)) |intents_value| {
        // Parse intents as decimal or comma-separated bit names in future; for now decimal
        g_state.intents = std.fmt.parseInt(u32, intents_value, 10) catch 0;
    }
    if (context.config.getParameter(CONFIG_KEY_APPLICATION_ID)) |v| {
        const owned = context.allocator.alloc(u8, v.len) catch return -2;
        @memcpy(owned, v);
        g_state.application_id = owned;
    }
    if (context.config.getParameter(CONFIG_KEY_GUILD_ID)) |v| {
        const owned = context.allocator.alloc(u8, v.len) catch return -2;
        @memcpy(owned, v);
        g_state.guild_id = owned;
    }

    // REST client
    if (g_state.token) |t| {
        g_state.rest = DiscordRestClient.init(context.allocator, t) catch |err| {
            std.log.warn("[discord_plugin] REST init failed: {}", .{err});
            g_state.rest = null;
            return 0;
        };
    }

    g_state.initialized = true;
    g_state.running = false;
    g_state.messages_sent = 0;

    context.log(1, "discord_plugin initialized");
    return 0;
}

/// Cleans up all plugin resources and resets the plugin state. Called during
/// plugin shutdown to ensure proper cleanup and prevent resource leaks.
///
/// - Parameters:
///   - context: The plugin execution context provided by the host.
fn deinitPlugin(context: *PluginContext) callconv(.c) void {
    if (!g_state.initialized) return;
    g_state.deinit(context.allocator);
}

/// Starts the Discord plugin services, including Gateway connection and
/// command registration. This transitions the plugin from initialized to
/// fully operational state.
///
/// - Parameters:
///   - context: The plugin execution context provided by the host.
/// - Returns: 0 on success, negative error codes on failure.
fn startPlugin(context: *PluginContext) callconv(.c) c_int {
    if (!g_state.initialized) return -1;
    if (g_state.running) return -2;

    // Start gateway loop (scaffold)
    const allocator = context.allocator;
    var transport = WebSocketTransport.init(allocator);
    transport.connect(g_state.token orelse "", g_state.intents) catch |err| {
        Log.warn("Gateway connect failed (scaffold): {}", .{err});
    };

    g_state.gateway_stop.store(false, .release);
    const thread = std.Thread.spawn(.{}, gatewayMain, .{ transport, g_state.intents, g_state.token orelse "" }) catch |err| {
        Log.warn("Failed to spawn gateway thread: {}", .{err});
        return -3;
    };
    g_state.gateway_thread = thread;

    // Enhanced command registration with comprehensive examples
    if (g_state.rest != null and g_state.application_id != null) {
        const rest = g_state.rest.?;

        // Define command options and choices
        const priority_choices = [_]CommandChoice{
            .{ .name = "Low", .value = .{ .integer = 1 } },
            .{ .name = "Medium", .value = .{ .integer = 2 } },
            .{ .name = "High", .value = .{ .integer = 3 } },
            .{ .name = "Critical", .value = .{ .integer = 4 } },
        };

        const echo_options = [_]CommandOption{
            .{ .name = "message", .description = "Message to echo back", .option_type = .string, .required = true },
            .{ .name = "times", .description = "Number of times to repeat (1-5)", .option_type = .integer, .required = false },
        };

        const settings_get_option = [_]CommandOption{
            .{ .name = "key", .description = "Setting key to retrieve", .option_type = .string, .required = true },
        };

        const settings_set_options = [_]CommandOption{
            .{ .name = "key", .description = "Setting key to set", .option_type = .string, .required = true },
            .{ .name = "value", .description = "Setting value", .option_type = .string, .required = true },
        };

        const settings_subcommands = [_]CommandOption{
            .{ .name = "get", .description = "Get a setting value", .option_type = .sub_command, .options = &settings_get_option },
            .{ .name = "set", .description = "Set a setting value", .option_type = .sub_command, .options = &settings_set_options },
        };

        const alert_options = [_]CommandOption{
            .{ .name = "message", .description = "Alert message", .option_type = .string, .required = true },
            .{ .name = "priority", .description = "Alert priority level", .option_type = .integer, .required = false, .choices = &priority_choices },
            .{ .name = "channel", .description = "Channel to send alert to", .option_type = .channel, .required = false },
        };

        const commands = [_]EnhancedCommandSpec{
            // Global commands (available in all servers)
            .{ .name = "ping", .description = "Check bot latency and responsiveness", .global = true },
            .{ .name = "help", .description = "Show available commands and usage", .global = true },
            .{ .name = "version", .description = "Display bot version and system information", .global = true },

            // Guild-specific commands with options
            .{ .name = "echo", .description = "Echo a message back with optional repetition", .options = &echo_options, .dm_permission = false },
            .{
                .name = "settings",
                .description = "Manage bot settings for this server",
                .options = &settings_subcommands,
                .dm_permission = false,
                .default_member_permissions = "32", // MANAGE_GUILD permission
            },
            .{
                .name = "alert",
                .description = "Send a priority alert to the server",
                .options = &alert_options,
                .dm_permission = false,
                .default_member_permissions = "8", // ADMINISTRATOR permission
            },
            .{ .name = "status", .description = "Display detailed bot status and connection info" },
            .{ .name = "uptime", .description = "Show how long the bot has been running" },
        };

        Log.info("Registering {d} enhanced slash commands...", .{commands.len});

        // Register global commands
        if (g_state.application_id) |app_id| {
            rest.upsertCommands(app_id, null, &commands) catch |err| {
                Log.err("Global command registration failed: {}", .{err});
            };
        }

        // Register guild commands
        if (g_state.guild_id) |guild_id| {
            rest.upsertCommands(g_state.application_id.?, guild_id, &commands) catch |err| {
                Log.err("Guild command registration failed: {}", .{err});
            };
        }

        Log.info("Enhanced command registration completed", .{});
    } else {
        Log.warn("Skipping command registration - missing configuration (app_id={}, rest={})", .{
            g_state.application_id != null,
            g_state.rest != null,
        });
    }

    g_state.running = true;
    context.log(1, "discord_plugin started");
    return 0;
}

/// Stops the Discord plugin services and gracefully shuts down all connections.
/// This transitions the plugin from running to initialized state.
///
/// - Parameters:
///   - context: The plugin execution context provided by the host.
/// - Returns: 0 on success, negative error codes on failure.
fn stopPlugin(context: *PluginContext) callconv(.c) c_int {
    if (!g_state.running) return -1;

    g_state.gateway_stop.store(true, .release);
    if (g_state.gateway_thread) |t| t.join();
    g_state.gateway_thread = null;

    g_state.running = false;
    context.log(1, "discord_plugin stopped");
    return 0;
}

/// Updates the plugin configuration with new parameters without requiring
/// a full restart. Allows runtime reconfiguration of tokens, intents, and
/// other operational parameters.
///
/// - Parameters:
///   - context: The plugin execution context provided by the host.
///   - config: The new configuration parameters to apply.
/// - Returns: 0 on success, negative error codes on failure.
fn configurePlugin(_context: *PluginContext, _: *const PluginConfig) callconv(.c) c_int {
    if (!g_state.initialized) return -1;

    // Parse configuration parameters using context config
    const config = _context.config;

    if (config.getParameter(CONFIG_KEYS.TOKEN)) |token_value| {
        const allocator = _context.allocator;
        if (g_state.token) |t| allocator.free(t);
        const owned = allocator.dupe(u8, token_value) catch return -1;
        g_state.token = owned;
    }

    if (config.getParameter(CONFIG_KEYS.INTENTS)) |intents_value| {
        g_state.intents = std.fmt.parseInt(u32, intents_value, 10) catch g_state.intents;
    }

    if (config.getParameter(CONFIG_KEYS.APPLICATION_ID)) |v| {
        const allocator = _context.allocator;
        if (g_state.application_id) |x| allocator.free(x);
        const owned = allocator.dupe(u8, v) catch return -1;
        g_state.application_id = owned;
    }

    if (config.getParameter(CONFIG_KEYS.GUILD_ID)) |v| {
        const allocator = _context.allocator;
        if (g_state.guild_id) |x| allocator.free(x);
        const owned = allocator.dupe(u8, v) catch return -1;
        g_state.guild_id = owned;
    }

    return 0;
}

/// Returns the current operational status of the plugin as a simple integer.
/// Used by the host application to monitor plugin health and state.
///
/// - Parameters:
///   - context: The plugin execution context provided by the host.
/// - Returns: 0=uninitialized, 1=initialized, 2=running.
fn getStatus(context: *PluginContext) callconv(.c) c_int {
    _ = context;
    if (!g_state.initialized) return 0; // uninitialized
    if (!g_state.running) return 1; // initialized
    return 2; // running
}

/// Provides detailed metrics about plugin operation in JSON format, including
/// status information and operational counters for monitoring and debugging.
///
/// - Parameters:
///   - context: The plugin execution context provided by the host.
///   - buffer: The buffer to write the JSON metrics into.
///   - buffer_size: The maximum size of the provided buffer.
/// - Returns: The number of bytes written, or -1 on error.
fn getMetrics(context: *PluginContext, buffer: [*]u8, buffer_size: usize) callconv(.c) c_int {
    _ = context;
    const status = if (g_state.running) "running" else if (g_state.initialized) "initialized" else "uninitialized";
    const out = std.fmt.bufPrint(
        buffer[0..buffer_size],
        "{{\"status\":\"{s}\",\"messages_sent\":{d}}}",
        .{ status, g_state.messages_sent },
    ) catch return -1;
    return @intCast(out.len);
}

/// Handles system events sent by the host application, such as lifecycle
/// events or configuration changes. Allows the plugin to respond to
/// system-wide state changes.
///
/// - Parameters:
///   - context: The plugin execution context provided by the host.
///   - event_type: The type of event being sent (host-defined).
///   - event_data: Optional event-specific data.
/// - Returns: 0 on successful handling, negative codes on error.
fn onEvent(context: *PluginContext, event_type: u32, event_data: ?*anyopaque) callconv(.c) c_int {
    _ = event_data;
    // Define basic lifecycle/system events; users can extend as needed
    switch (event_type) {
        1 => context.log(1, "discord_plugin: system startup event"),
        2 => context.log(1, "discord_plugin: system shutdown event"),
        else => context.log(2, "discord_plugin: unknown event"),
    }
    return 0;
}

/// Returns a pointer to the specified API interface if supported by this plugin.
/// Currently supports the "discord_v1" API which provides the DiscordApi
/// function table for direct Discord operations.
///
/// - Parameters:
///   - api_name_z: The null-terminated name of the requested API.
/// - Returns: A pointer to the API interface, or null if not supported.
fn getApi(api_name_z: [*:0]const u8) callconv(.c) ?*anyopaque {
    const api_name = std.mem.span(api_name_z);
    if (std.mem.eql(u8, api_name, "discord_v1")) {
        return &DISCORD_API;
    }
    return null;
}

// We keep process() optional and unused in this skeleton. A future extension can
// use process() to pass structured commands without relying on get_api().

// =============================================================================
// PLUGIN INTERFACE VTABLE AND ENTRY POINT
// =============================================================================

/// Static plugin interface structure containing function pointers to all
/// required plugin methods. This is the primary interface through which
/// the host application interacts with the plugin.
const PLUGIN_INTERFACE = PluginInterface{
    .get_info = getInfo,
    .init = initPlugin,
    .deinit = deinitPlugin,
    .start = startPlugin,
    .stop = stopPlugin,
    .configure = configurePlugin,
    .get_status = getStatus,
    .get_metrics = getMetrics,
    .on_event = onEvent,
    .get_api = getApi,
};

/// Main plugin entry point called by the host application during plugin loading.
/// Returns a pointer to the plugin interface structure that the host will use
/// for all subsequent plugin interactions.
///
/// - Returns: A pointer to the static plugin interface structure.
pub export fn abi_plugin_create() ?*const PluginInterface {
    return &PLUGIN_INTERFACE;
}

/// Main Gateway thread function that manages the Discord WebSocket connection
/// with comprehensive reconnection logic, heartbeat management, and event
/// processing. Runs in a separate thread to avoid blocking the main plugin.
///
/// - Parameters:
///   - transport: The WebSocket transport instance to manage.
///   - intents: The Discord Gateway intents for this connection.
///   - token: The Discord bot token for authentication.
fn gatewayMain(transport: WebSocketTransport, intents: u32, token: []const u8) void {
    var t = transport;

    // Main connection loop with reconnection
    while (!g_state.gateway_stop.load(.acquire)) {
        // Attempt connection with backoff
        t.connect(token, intents) catch |err| {
            Log.err("Gateway connection failed: {}", .{err});
            if (err == WebSocketTransport.ReconnectError.MaxAttemptsReached) {
                Log.err("Max reconnection attempts reached, giving up", .{});
                break;
            }
            continue;
        };

        // Send IDENTIFY/RESUME
        t.sendIdentify(token, intents) catch |err| {
            Log.warn("IDENTIFY/RESUME failed: {}", .{err});
            continue;
        };

        // Start heartbeat and read loops
        const HeartbeatContext = struct {
            transport: *WebSocketTransport,
            stop_flag: *std.atomic.Value(bool),
        };

        var heartbeat_ctx = HeartbeatContext{
            .transport = &t,
            .stop_flag = &g_state.gateway_stop,
        };

        // Spawn heartbeat thread
        const HeartbeatThread = struct {
            fn run(ctx: *HeartbeatContext) void {
                const interval_ms = ctx.transport.heartbeat_interval_ms;
                var tick: u64 = 0;

                while (!ctx.stop_flag.load(.acquire) and ctx.transport.connected) {
                    const start_ms = std.time.milliTimestamp();
                    while (std.time.milliTimestamp() - start_ms < interval_ms and !ctx.stop_flag.load(.acquire)) {
                        std.atomic.spinLoopHint();
                    }

                    if (ctx.stop_flag.load(.acquire)) break;

                    ctx.transport.sendHeartbeat() catch |err| {
                        Log.warn("Heartbeat failed: {}, will reconnect", .{err});
                        ctx.transport.connected = false;
                        break;
                    };

                    tick += 1;
                    if ((tick % 10) == 0) {
                        Log.info("Heartbeat tick {d} (seq={?d})", .{ tick, ctx.transport.last_sequence });
                    }
                }
                Log.info("Heartbeat thread exiting", .{});
            }
        };

        var heartbeat_thread = std.Thread.spawn(.{}, HeartbeatThread.run, .{&heartbeat_ctx}) catch |err| {
            Log.err("Failed to spawn heartbeat thread: {}", .{err});
            continue;
        };

        // Run main read loop
        t.readLoop() catch |err| {
            Log.warn("Gateway read loop error: {}", .{err});
        };

        // Signal heartbeat thread to stop and wait for it
        t.connected = false;
        heartbeat_thread.join();

        if (!g_state.gateway_stop.load(.acquire)) {
            Log.info("Connection lost, will attempt reconnection...", .{});
            // Small delay before reconnection attempt
            t.sleepMs(1000);
        }
    }

    Log.info("Gateway main exiting", .{});
}

// =============================================================================
// CONSTANTS AND CONFIGURATION
// =============================================================================

const CONFIG_KEYS = struct {
    pub const TOKEN = "DISCORD_TOKEN";
    pub const INTENTS = "DISCORD_INTENTS";
    pub const APPLICATION_ID = "DISCORD_APPLICATION_ID";
    pub const GUILD_ID = "DISCORD_GUILD_ID";
};

const GATEWAY_CONFIG = struct {
    pub const DEFAULT_HEARTBEAT_MS: u32 = 41250;
    pub const MAX_RECONNECT_ATTEMPTS: u32 = 5;
    pub const HEARTBEAT_TIMEOUT_MULTIPLIER: u32 = 2;
    pub const BACKOFF_BASE_MS: u32 = 1000;
    pub const BACKOFF_MAX_MS: u32 = 30000;
};
