const std = @import("std");
const types = @import("types.zig");

pub const DiscordBot = struct {
    allocator: std.mem.Allocator,
    token: []const u8,
    rate_limit: RateLimiter,
    ws: ?std.websocket.Client = null,

    pub fn init(allocator: std.mem.Allocator, token: []const u8) DiscordBot {
        return .{ .allocator = allocator, .token = token, .rate_limit = RateLimiter.init(50), .ws = null };
    }

    pub fn deinit(self: *DiscordBot) void {
        if (self.ws) |*w| w.deinit();
    }

    pub fn connect(self: *DiscordBot) !void {
        const gateway_url = try fetchGatewayUrl(self.allocator, self.token);
        defer self.allocator.free(gateway_url);

        self.ws = try std.websocket.Client.connect(self.allocator, gateway_url, .{});
        var ws = self.ws.?;
        var frame_buf: [1024]u8 = undefined;

        while (true) {
            const frame = try ws.readFrame(&frame_buf);
            switch (frame.opcode) {
                .text => try handleTextFrame(self, frame.data),
                else => {},
            }
        }
    }
};

fn handleTextFrame(self: *DiscordBot, payload: []const u8) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, payload, .{});
    defer parsed.deinit();

    const root = &parsed.value;
    const opv = root.object.get("op") orelse return;
    const op = @intCast(types.OpCode, opv.integer);

    switch (op) {
        .hello => if (root.object.get("d")) |data| {
            const interval = data.object.get("heartbeat_interval")?.integer orelse return;
            try heartbeatLoop(self, @as(u32, @intCast(interval)));
            try identify(self);
        } else {},
        else => {},
    }
}

fn heartbeatLoop(self: *DiscordBot, interval: u32) !void {
    var ws = self.ws.?;
    const buf = try self.allocator.alloc(u8, 32);
    defer self.allocator.free(buf);
    var timer = std.time.Timer.start() catch return;
    while (true) {
        try std.json.stringify(.{ .op = @as(u8, @intCast(types.OpCode.heartbeat)), .d = null }, .{}, std.io.fixedBufferStream(buf).writer());
        try ws.writeFrame(.{ .fin = true, .opcode = .text, .data = buf });
        std.time.sleep(interval * std.time.millisecond);
    }
}

fn identify(self: *DiscordBot) !void {
    var ws = self.ws.?;
    const identify = types.Identify{ .token = self.token, .intents = 1 << 15 };
    var buf = std.ArrayList(u8).init(self.allocator);
    defer buf.deinit();
    try std.json.stringify(.{ .op = @as(u8, @intCast(types.OpCode.identify)), .d = identify }, .{}, buf.writer());
    try ws.writeFrame(.{ .fin = true, .opcode = .text, .data = buf.items });
}

fn fetchGatewayUrl(allocator: std.mem.Allocator, token: []const u8) ![]const u8 {
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const auth_value = try std.fmt.allocPrint(allocator, "Bot {s}", .{token});
    defer allocator.free(auth_value);

    const headers = [_]std.http.Header{ .{ .name = "Authorization", .value = auth_value } };
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    const result = try client.fetch(.{ .location = .{ .url = "https://discord.com/api/v10/gateway" }, .extra_headers = &headers, .response_storage = .dynamic(&buf) });
    if (result.status != .ok) return error.BadResponse;

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, buf.items, .{});
    defer parsed.deinit();
    const url_val = parsed.value.object.get("url") orelse return error.BadResponse;
    return allocator.dupe(u8, url_val.string) catch error.OutOfMemory;
}

const RateLimiter = struct {
    capacity: u32,
    tokens: u32,
    last: i128,

    pub fn init(capacity: u32) RateLimiter {
        return .{ .capacity = capacity, .tokens = capacity, .last = std.time.microTimestamp() };
    }

    pub fn allow(self: *RateLimiter) bool {
        const now = std.time.microTimestamp();
        const elapsed = now - self.last;
        const replenish = @min(self.capacity - self.tokens, @as(u32, @intCast(elapsed / 1_000_000)) * self.capacity);
        self.tokens += replenish;
        self.last = now;
        if (self.tokens == 0) return false;
        self.tokens -= 1;
        return true;
    }
};
