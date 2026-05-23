const std = @import("std");
const connector = @import("connector.zig");
const http = @import("http.zig");
const json_lib = @import("json.zig");

const ConnectorError = connector.ConnectorError;
const ConnectorConfig = connector.ConnectorConfig;
const Response = connector.Response;

pub const TwilioConfig = struct {
    account_sid: []const u8,
    auth_token: []const u8,
    base_url: []const u8 = "https://api.twilio.com",
    timeout_ms: u32 = 30000,
    transport: connector.TransportMode = .local,
    escalation_url: []const u8 = "",

    pub fn local() TwilioConfig {
        return .{
            .account_sid = "local-account",
            .auth_token = "local-token",
            .transport = .local,
        };
    }
};

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

pub const Client = struct {
    allocator: std.mem.Allocator,
    config: TwilioConfig,

    pub fn init(allocator: std.mem.Allocator, config: TwilioConfig) Client {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn deinit(self: *Client) void {
        _ = self;
    }

    pub fn handleConversationRelayEvent(
        self: *Client,
        allocator: std.mem.Allocator,
        event: ConversationRelayEvent,
        agent_reply: []const u8,
    ) ConnectorError!ConversationRelayResponse {
        try validateTwilioConfig(self.config);
        if (self.config.transport == .live) return ConnectorError.LiveTransportUnavailable;
        return try buildLocalConversationResponse(allocator, event, agent_reply);
    }

    pub fn handleConversationRelayEventLive(
        self: *Client,
        io: std.Io,
        allocator: std.mem.Allocator,
        event: ConversationRelayEvent,
        agent_reply: []const u8,
    ) ConnectorError!ConversationRelayResponse {
        try validateTwilioConfig(self.config);

        const escalation = classifyEscalation(event);
        const escalation_url = if (self.config.escalation_url.len > 0) self.config.escalation_url else "";
        if (escalation != null and escalation_url.len == 0) {
            std.log.warn("Twilio escalation triggered but no escalation_url configured; redirect omitted", .{});
        }
        const twiml = try buildTwiMLSay(allocator, agent_reply, escalation != null, escalation_url);
        defer allocator.free(twiml);

        const form = try buildUrlEncodedForm(allocator, &.{
            .{ "Twiml", twiml },
            .{ "conversation_id", event.conversation_id },
            .{ "customer_id", event.customer_id },
        });
        defer allocator.free(form);

        const auth = try http.basicAuthHeader(allocator, self.config.account_sid, self.config.auth_token);
        defer allocator.free(auth);

        const path = try std.fmt.allocPrint(allocator, "/2010-04-01/Accounts/{s}/Calls.json", .{self.config.account_sid});
        defer allocator.free(path);

        const http_response = try http.httpPostForm(io, allocator, .{
            .api_key = self.config.account_sid,
            .base_url = self.config.base_url,
            .timeout_ms = self.config.timeout_ms,
            .transport = .live,
        }, path, form, &.{
            .{ .name = "authorization", .value = auth },
        });

        const response_text = try allocator.dupe(u8, agent_reply);
        errdefer allocator.free(response_text);

        var result = ConversationRelayResponse{
            .text = response_text,
            .escalation = null,
        };

        if (escalation) |reason| {
            var payload = try buildEscalationPayload(allocator, event, reason);
            errdefer payload.deinit(allocator);
            result.escalation = payload;
        }

        if (http_response.body.len > 0) {
            std.log.info("Twilio live response: {s}", .{http_response.body});
        }
        return result;
    }
};

pub fn validateTwilioConfig(config: TwilioConfig) ConnectorError!void {
    if (config.account_sid.len == 0) return ConnectorError.AuthenticationError;
    if (config.auth_token.len == 0) return ConnectorError.AuthenticationError;
    if (config.base_url.len == 0) return ConnectorError.ConnectionFailed;
    if (config.timeout_ms == 0) return ConnectorError.Timeout;
}

pub fn parseConversationRelayEvent(allocator: std.mem.Allocator, payload: []const u8) ConnectorError!ConversationRelayEvent {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, payload, .{}) catch return ConnectorError.InvalidResponse;
    defer parsed.deinit();

    const root = switch (parsed.value) {
        .object => |obj| obj,
        else => return ConnectorError.InvalidResponse,
    };

    const kind_text = objectStringAny(root, &.{ "type", "event" }) orelse return ConnectorError.InvalidResponse;
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

// --- Private helpers ---

fn buildLocalConversationResponse(
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

fn buildEscalationPayload(
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

fn classifyEscalation(event: ConversationRelayEvent) ?EscalationReason {
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
    if (stringEqlIgnoreCase(value, "setup")) return .setup;
    if (stringEqlIgnoreCase(value, "user_transcript") or stringEqlIgnoreCase(value, "transcript") or stringEqlIgnoreCase(value, "prompt")) return .user_transcript;
    if (stringEqlIgnoreCase(value, "dtmf")) return .dtmf;
    if (stringEqlIgnoreCase(value, "interrupt")) return .interrupt;
    if (stringEqlIgnoreCase(value, "disconnect")) return .disconnect;
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
    return .{
        .sentiment = sentiment,
        .compliance_status = compliance_status,
        .escalation_recommended = objectBool(obj, "escalation_recommended") orelse objectBool(obj, "escalationRecommended") orelse false,
    };
}

fn dupeObjectStringAny(allocator: std.mem.Allocator, obj: std.json.ObjectMap, keys: []const []const u8, default: []const u8) ConnectorError![]u8 {
    const value = objectStringAny(obj, keys) orelse default;
    return try allocator.dupe(u8, value);
}

fn objectStringAny(obj: std.json.ObjectMap, keys: []const []const u8) ?[]const u8 {
    for (keys) |key| {
        const value = obj.get(key) orelse continue;
        return switch (value) {
            .string => |s| s,
            else => null,
        };
    }
    return null;
}

fn objectBool(obj: std.json.ObjectMap, key: []const u8) ?bool {
    const value = obj.get(key) orelse return null;
    return switch (value) {
        .bool => |b| b,
        else => null,
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
        if (stringEqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

fn stringEqlIgnoreCase(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |left, right| {
        if (std.ascii.toLower(left) != std.ascii.toLower(right)) return false;
    }
    return true;
}

// --- Twilio XML/Form helpers ---

fn buildTwiMLSay(allocator: std.mem.Allocator, text: []const u8, escalate: bool, escalation_url: []const u8) ConnectorError![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response><Say>");
    for (text) |byte| {
        switch (byte) {
            '<' => try out.appendSlice(allocator, "&lt;"),
            '>' => try out.appendSlice(allocator, "&gt;"),
            '&' => try out.appendSlice(allocator, "&amp;"),
            '"' => try out.appendSlice(allocator, "&quot;"),
            else => try out.append(allocator, byte),
        }
    }
    try out.appendSlice(allocator, "</Say>");
    if (escalate and escalation_url.len > 0) {
        try out.appendSlice(allocator, "<Redirect>");
        try out.appendSlice(allocator, escalation_url);
        try out.appendSlice(allocator, "</Redirect>");
    }
    try out.appendSlice(allocator, "</Response>");
    return try out.toOwnedSlice(allocator);
}

fn buildUrlEncodedForm(allocator: std.mem.Allocator, fields: []const [2][]const u8) ConnectorError![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    for (fields, 0..) |pair, i| {
        if (i > 0) try out.append(allocator, '&');
        try appendUrlEncoded(&out, allocator, pair[0]);
        try out.append(allocator, '=');
        try appendUrlEncoded(&out, allocator, pair[1]);
    }
    return try out.toOwnedSlice(allocator);
}

fn appendUrlEncoded(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: []const u8) !void {
    for (value) |byte| {
        switch (byte) {
            'A'...'Z', 'a'...'z', '0'...'9', '-', '_', '.', '~' => try out.append(allocator, byte),
            ' ' => try out.appendSlice(allocator, "+"),
            else => try out.print(allocator, "%{X:0>2}", .{byte}),
        }
    }
}
