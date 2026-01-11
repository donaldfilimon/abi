//! Core span types and span lifecycle management for distributed tracing.
//!
//! Provides the fundamental building blocks for tracing: TraceId, SpanId,
//! attributes, events, links, and the Span struct itself.

const std = @import("std");

pub const TraceId = [16]u8;
pub const SpanId = [8]u8;

pub const SpanKind = enum {
    internal,
    server,
    client,
    producer,
    consumer,
};

pub const SpanStatus = enum {
    unset,
    ok,
    error_status,
};

pub const AttributeValue = union(enum) {
    string: []const u8,
    int: i64,
    float: f64,
    bool: bool,
};

pub const SpanAttribute = struct {
    key: []const u8,
    value: AttributeValue,
};

pub const SpanEvent = struct {
    name: []const u8,
    timestamp: i64,
    attributes: []const SpanAttribute,
};

pub const SpanLink = struct {
    trace_id: TraceId,
    span_id: SpanId,
    attributes: []const SpanAttribute,
};

pub const Span = struct {
    name: []const u8,
    trace_id: TraceId,
    span_id: SpanId,
    parent_span_id: ?SpanId,
    kind: SpanKind,
    start_time: i64,
    end_time: i64 = 0,
    attributes: std.ArrayListUnmanaged(SpanAttribute),
    events: std.ArrayListUnmanaged(SpanEvent),
    links: std.ArrayListUnmanaged(SpanLink),
    status: SpanStatus = .unset,
    error_message: ?[]const u8 = null,
    allocator: std.mem.Allocator,

    pub fn start(
        allocator: std.mem.Allocator,
        name: []const u8,
        trace_id: ?TraceId,
        parent_span_id: ?SpanId,
        kind: SpanKind,
    ) !Span {
        var attrs = std.ArrayListUnmanaged(SpanAttribute).empty;
        errdefer attrs.deinit(allocator);

        var events = std.ArrayListUnmanaged(SpanEvent).empty;
        errdefer events.deinit(allocator);

        var links = std.ArrayListUnmanaged(SpanLink).empty;
        errdefer links.deinit(allocator);

        return .{
            .name = try allocator.dupe(u8, name),
            .trace_id = trace_id orelse generateTraceId(),
            .span_id = generateSpanId(),
            .parent_span_id = parent_span_id,
            .kind = kind,
            .start_time = std.time.timestamp(),
            .attributes = attrs,
            .events = events,
            .links = links,
            .allocator = allocator,
        };
    }

    pub fn end(self: *Span) void {
        self.end_time = std.time.timestamp();
    }

    pub fn deinit(self: *Span) void {
        self.allocator.free(self.name);
        for (self.attributes.items) |attr| {
            self.allocator.free(attr.key);
            switch (attr.value) {
                .string => |s| self.allocator.free(s),
                else => {},
            }
        }
        self.attributes.deinit(self.allocator);
        for (self.events.items) |event| {
            self.allocator.free(event.name);
            for (event.attributes) |attr| {
                self.allocator.free(attr.key);
                switch (attr.value) {
                    .string => |s| self.allocator.free(s),
                    else => {},
                }
            }
            self.allocator.free(event.attributes);
        }
        self.events.deinit(self.allocator);
        for (self.links.items) |link| {
            self.allocator.free(link.attributes);
        }
        self.links.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn setAttribute(self: *Span, key: []const u8, value: AttributeValue) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);

        var value_copy = value;
        switch (value) {
            .string => |s| {
                value_copy = .{ .string = try self.allocator.dupe(u8, s) };
            },
            else => {},
        }

        try self.attributes.append(self.allocator, .{
            .key = key_copy,
            .value = value_copy,
        });
    }

    pub fn addEvent(self: *Span, name: []const u8) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        try self.events.append(self.allocator, .{
            .name = name_copy,
            .timestamp = std.time.timestamp(),
            .attributes = &.{},
        });
    }

    pub fn addLink(self: *Span, other_trace_id: TraceId, other_span_id: SpanId) !void {
        try self.links.append(self.allocator, .{
            .trace_id = other_trace_id,
            .span_id = other_span_id,
            .attributes = &.{},
        });
    }

    pub fn setStatus(self: *Span, status: SpanStatus, message: ?[]const u8) void {
        self.status = status;
        self.error_message = message;
    }

    pub fn recordError(self: *Span, message: []const u8) !void {
        self.status = .error_status;
        const msg_copy = try self.allocator.dupe(u8, message);
        self.error_message = msg_copy;
        try self.addEvent("exception");
    }

    pub fn getDuration(self: Span) i64 {
        return self.end_time - self.start_time;
    }

    pub fn generateTraceId() TraceId {
        var trace_id: TraceId = undefined;
        std.crypto.random.bytes(&trace_id);
        trace_id[0] = 0;
        return trace_id;
    }

    pub fn generateSpanId() SpanId {
        var span_id: SpanId = undefined;
        std.crypto.random.bytes(&span_id);
        return span_id;
    }
};

test "span lifecycle" {
    const allocator = std.testing.allocator;
    var span = try Span.start(allocator, "test-operation", null, null, .internal);
    defer span.deinit();

    try span.setAttribute("key", .{ .string = "value" });
    try span.addEvent("event1");

    span.end();
    try std.testing.expect(span.end_time > span.start_time);
}
