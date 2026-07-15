const std = @import("std");
const connector = @import("connector.zig");
const json_lib = @import("json.zig");

const ConnectorError = connector.ConnectorError;

pub const ConversationRelayEventKind = enum {
    setup,
    user_transcript,
    dtmf,
    interrupt,
    disconnect,
};

pub const EscalationReason = enum {
    human_requested,
    empty_transcript,
    sensitive_topic,
    low_confidence,
    intelligence_signal,
};

pub const ConversationMemory = struct {
    profile_id: []const u8 = "",
    profile_summary: []const u8 = "",
    recall_summary: []const u8 = "",

    pub fn deinit(self: *ConversationMemory, allocator: std.mem.Allocator) void {
        allocator.free(self.profile_id);
        allocator.free(self.profile_summary);
        allocator.free(self.recall_summary);
    }
};

pub const IntelligenceSignal = struct {
    sentiment: []const u8 = "neutral",
    compliance_status: []const u8 = "clear",
    escalation_recommended: bool = false,

    pub fn deinit(self: *IntelligenceSignal, allocator: std.mem.Allocator) void {
        allocator.free(self.sentiment);
        allocator.free(self.compliance_status);
    }
};

pub const ConversationRelayEvent = struct {
    kind: ConversationRelayEventKind,
    conversation_id: []const u8,
    customer_id: []const u8,
    transcript: []const u8 = "",
    digit: []const u8 = "",
    memory: ?ConversationMemory = null,
    intelligence: ?IntelligenceSignal = null,
    owned: bool = false,

    pub fn deinit(self: *ConversationRelayEvent, allocator: std.mem.Allocator) void {
        if (!self.owned) return;
        allocator.free(self.conversation_id);
        allocator.free(self.customer_id);
        allocator.free(self.transcript);
        allocator.free(self.digit);
        if (self.memory) |*memory| memory.deinit(allocator);
        if (self.intelligence) |*signal| signal.deinit(allocator);
    }
};

pub const EscalationPayload = struct {
    conversation_id: []const u8,
    customer_id: []const u8,
    reason_code: []const u8,
    summary: []const u8,
    routing_hints: []const u8,
    owned: bool = false,

    pub fn deinit(self: *EscalationPayload, allocator: std.mem.Allocator) void {
        if (!self.owned) return;
        allocator.free(self.conversation_id);
        allocator.free(self.customer_id);
        allocator.free(self.reason_code);
        allocator.free(self.summary);
        allocator.free(self.routing_hints);
    }
};

pub const ConversationRelayResponse = struct {
    text: []u8,
    escalation: ?EscalationPayload = null,
    owned: bool = true,

    pub fn deinit(self: *ConversationRelayResponse, allocator: std.mem.Allocator) void {
        if (self.owned) allocator.free(self.text);
        if (self.escalation) |*payload| payload.deinit(allocator);
    }
};

pub fn parseConversationRelayEvent(allocator: std.mem.Allocator, payload: []const u8) ConnectorError!ConversationRelayEvent {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, payload, .{}) catch return ConnectorError.InvalidResponse;
    defer parsed.deinit();

    const root = switch (parsed.value) {
        .object => |obj| obj,
        else => return ConnectorError.InvalidResponse,
    };

    const kind_text = try objectStringAny(root, &.{ "type", "event" }) orelse return ConnectorError.InvalidResponse;
    const kind = try parseConversationRelayEventKind(kind_text);

    const conversation_id = try dupeObjectStringAny(allocator, root, &.{ "conversation_id", "conversationId", "call_sid", "callSid" }, "local-conversation");
    errdefer allocator.free(conversation_id);
    const customer_id = try dupeObjectStringAny(allocator, root, &.{ "customer_id", "customerId", "from" }, "anonymous");
    errdefer allocator.free(customer_id);
    const transcript = try dupeObjectStringAny(allocator, root, &.{ "transcript", "text", "utterance" }, "");
    errdefer allocator.free(transcript);
    const digit = try dupeObjectStringAny(allocator, root, &.{ "digit", "dtmf" }, "");
    errdefer allocator.free(digit);
    var memory = try parseConversationMemory(allocator, root);
    errdefer if (memory) |*m| m.deinit(allocator);
    var intelligence = try parseIntelligenceSignal(allocator, root);
    errdefer if (intelligence) |*signal| signal.deinit(allocator);

    return .{
        .kind = kind,
        .conversation_id = conversation_id,
        .customer_id = customer_id,
        .transcript = transcript,
        .digit = digit,
        .memory = memory,
        .intelligence = intelligence,
        .owned = true,
    };
}

pub fn buildConversationRelayJson(allocator: std.mem.Allocator, response: ConversationRelayResponse) ConnectorError![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, "{\"text\":");
    try json_lib.appendJsonString(&out, allocator, response.text);
    try out.appendSlice(allocator, ",\"escalation\":");
    if (response.escalation) |payload| {
        try out.append(allocator, '{');
        try out.appendSlice(allocator, "\"conversation_id\":");
        try json_lib.appendJsonString(&out, allocator, payload.conversation_id);
        try out.appendSlice(allocator, ",\"customer_id\":");
        try json_lib.appendJsonString(&out, allocator, payload.customer_id);
        try out.appendSlice(allocator, ",\"reason_code\":");
        try json_lib.appendJsonString(&out, allocator, payload.reason_code);
        try out.appendSlice(allocator, ",\"summary\":");
        try json_lib.appendJsonString(&out, allocator, payload.summary);
        try out.appendSlice(allocator, ",\"routing_hints\":");
        try json_lib.appendJsonString(&out, allocator, payload.routing_hints);
        try out.append(allocator, '}');
    } else {
        try out.appendSlice(allocator, "null");
    }
    try out.append(allocator, '}');
    return try out.toOwnedSlice(allocator);
}

pub fn buildLocalConversationResponse(
    allocator: std.mem.Allocator,
    event: ConversationRelayEvent,
    agent_reply: []const u8,
) ConnectorError!ConversationRelayResponse {
    switch (event.kind) {
        .setup => return .{
            .text = try allocator.dupe(u8, "Hello, this is ABI support. How can I help today?"),
        },
        .disconnect => return .{
            .text = try allocator.dupe(u8, "Thanks for contacting ABI support."),
        },
        .dtmf => return .{
            .text = try std.fmt.allocPrint(allocator, "I received keypad input {s}.", .{if (event.digit.len > 0) event.digit else "unknown"}),
        },
        .interrupt => return .{
            .text = try allocator.dupe(u8, "I heard an interruption. Please continue and I will adjust."),
        },
        .user_transcript => {},
    }

    if (classifyEscalation(event)) |reason| {
        var payload = try buildEscalationPayload(allocator, event, reason);
        errdefer payload.deinit(allocator);
        return .{
            .text = try allocator.dupe(u8, "I can connect you with a support specialist. Please hold while I pass along the context."),
            .escalation = payload,
        };
    }

    const memory_note = if (event.memory) |memory| memory.recall_summary else "";
    const text = if (memory_note.len > 0)
        try std.fmt.allocPrint(allocator, "{s} I also found this customer context: {s}", .{ agent_reply, memory_note })
    else
        try allocator.dupe(u8, agent_reply);
    return .{ .text = text };
}

pub fn buildEscalationPayload(
    allocator: std.mem.Allocator,
    event: ConversationRelayEvent,
    reason: EscalationReason,
) ConnectorError!EscalationPayload {
    const conversation_id = try allocator.dupe(u8, event.conversation_id);
    errdefer allocator.free(conversation_id);
    const customer_id = try allocator.dupe(u8, event.customer_id);
    errdefer allocator.free(customer_id);
    const reason_code_text = reasonCode(reason);
    const reason_code_value = try allocator.dupe(u8, reason_code_text);
    errdefer allocator.free(reason_code_value);
    const transcript = std.mem.trim(u8, event.transcript, &std.ascii.whitespace);
    const summary = if (transcript.len > 0)
        try std.fmt.allocPrint(allocator, "Voice support escalation for {s}: {s}", .{ customer_id, transcript })
    else
        try std.fmt.allocPrint(allocator, "Voice support escalation for {s}: no usable transcript captured", .{customer_id});
    errdefer allocator.free(summary);
    const routing_hints = try std.fmt.allocPrint(allocator, "queue=support;priority={s};channel=voice;reason={s}", .{
        if (reason == .sensitive_topic or reason == .intelligence_signal) "high" else "normal",
        reason_code_text,
    });
    errdefer allocator.free(routing_hints);

    return .{
        .conversation_id = conversation_id,
        .customer_id = customer_id,
        .reason_code = reason_code_value,
        .summary = summary,
        .routing_hints = routing_hints,
        .owned = true,
    };
}

pub fn classifyEscalation(event: ConversationRelayEvent) ?EscalationReason {
    if (event.intelligence) |signal| {
        if (signal.escalation_recommended) return .intelligence_signal;
    }

    const transcript = std.mem.trim(u8, event.transcript, &std.ascii.whitespace);
    if (transcript.len == 0) return .empty_transcript;
    if (containsAnyIgnoreCase(transcript, &.{ "human", "representative", "real person", "support agent" })) return .human_requested;
    if (containsAnyIgnoreCase(transcript, &.{ "credit card", "card number", "ssn", "social security", "medical", "diagnosis", "debt collection" })) return .sensitive_topic;
    if (transcript.len < 3 or containsAnyIgnoreCase(transcript, &.{ "not sure", "confused", "unknown error" })) return .low_confidence;
    return null;
}

fn parseConversationRelayEventKind(value: []const u8) ConnectorError!ConversationRelayEventKind {
    if (std.ascii.eqlIgnoreCase(value, "setup")) return .setup;
    if (std.ascii.eqlIgnoreCase(value, "user_transcript") or std.ascii.eqlIgnoreCase(value, "transcript") or std.ascii.eqlIgnoreCase(value, "prompt")) return .user_transcript;
    if (std.ascii.eqlIgnoreCase(value, "dtmf")) return .dtmf;
    if (std.ascii.eqlIgnoreCase(value, "interrupt")) return .interrupt;
    if (std.ascii.eqlIgnoreCase(value, "disconnect")) return .disconnect;
    return ConnectorError.InvalidResponse;
}

fn parseConversationMemory(allocator: std.mem.Allocator, root: std.json.ObjectMap) ConnectorError!?ConversationMemory {
    const value = root.get("memory") orelse return null;
    const obj = switch (value) {
        .object => |memory_obj| memory_obj,
        else => return ConnectorError.InvalidResponse,
    };

    const profile_id = try dupeObjectStringAny(allocator, obj, &.{ "profile_id", "profileId" }, "");
    errdefer allocator.free(profile_id);
    const profile_summary = try dupeObjectStringAny(allocator, obj, &.{ "profile_summary", "profileSummary" }, "");
    errdefer allocator.free(profile_summary);
    const recall_summary = try dupeObjectStringAny(allocator, obj, &.{ "recall_summary", "recallSummary" }, "");
    errdefer allocator.free(recall_summary);
    return .{
        .profile_id = profile_id,
        .profile_summary = profile_summary,
        .recall_summary = recall_summary,
    };
}

fn parseIntelligenceSignal(allocator: std.mem.Allocator, root: std.json.ObjectMap) ConnectorError!?IntelligenceSignal {
    const value = root.get("intelligence") orelse return null;
    const obj = switch (value) {
        .object => |signal_obj| signal_obj,
        else => return ConnectorError.InvalidResponse,
    };

    const sentiment = try dupeObjectStringAny(allocator, obj, &.{"sentiment"}, "neutral");
    errdefer allocator.free(sentiment);
    const compliance_status = try dupeObjectStringAny(allocator, obj, &.{ "compliance_status", "complianceStatus" }, "clear");
    errdefer allocator.free(compliance_status);
    const escalation_recommended = (try objectBool(obj, "escalation_recommended")) orelse (try objectBool(obj, "escalationRecommended")) orelse false;
    return .{
        .sentiment = sentiment,
        .compliance_status = compliance_status,
        .escalation_recommended = escalation_recommended,
    };
}

fn dupeObjectStringAny(allocator: std.mem.Allocator, obj: std.json.ObjectMap, keys: []const []const u8, default: []const u8) ConnectorError![]u8 {
    const value = try objectStringAny(obj, keys) orelse default;
    return try allocator.dupe(u8, value);
}

fn objectStringAny(obj: std.json.ObjectMap, keys: []const []const u8) ConnectorError!?[]const u8 {
    for (keys) |key| {
        const value = obj.get(key) orelse continue;
        return switch (value) {
            .string => |s| s,
            else => ConnectorError.InvalidResponse,
        };
    }
    return null;
}

fn objectBool(obj: std.json.ObjectMap, key: []const u8) ConnectorError!?bool {
    const value = obj.get(key) orelse return null;
    return switch (value) {
        .bool => |b| b,
        else => ConnectorError.InvalidResponse,
    };
}

fn reasonCode(reason: EscalationReason) []const u8 {
    return switch (reason) {
        .human_requested => "human_requested",
        .empty_transcript => "empty_transcript",
        .sensitive_topic => "sensitive_topic",
        .low_confidence => "low_confidence",
        .intelligence_signal => "intelligence_signal",
    };
}

fn containsAnyIgnoreCase(haystack: []const u8, needles: []const []const u8) bool {
    for (needles) |needle| {
        if (containsIgnoreCase(haystack, needle)) return true;
    }
    return false;
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
