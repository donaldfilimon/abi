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

const testing = std.testing;

fn eventFor(kind: ConversationRelayEventKind, transcript: []const u8) ConversationRelayEvent {
    return .{
        .kind = kind,
        .conversation_id = "conv-1",
        .customer_id = "cust-1",
        .transcript = transcript,
        .owned = false,
    };
}

test "classifyEscalation: normal transcript is not escalated" {
    try testing.expect(classifyEscalation(eventFor(.user_transcript, "I'd like to check my order status please")) == null);
}

test "classifyEscalation: empty or whitespace-only transcript escalates" {
    try testing.expectEqual(EscalationReason.empty_transcript, classifyEscalation(eventFor(.user_transcript, "")).?);
    try testing.expectEqual(EscalationReason.empty_transcript, classifyEscalation(eventFor(.user_transcript, "   \t  ")).?);
}

test "classifyEscalation: explicit human request keywords escalate case-insensitively" {
    try testing.expectEqual(EscalationReason.human_requested, classifyEscalation(eventFor(.user_transcript, "let me talk to a HUMAN please")).?);
    try testing.expectEqual(EscalationReason.human_requested, classifyEscalation(eventFor(.user_transcript, "I want a representative")).?);
    try testing.expectEqual(EscalationReason.human_requested, classifyEscalation(eventFor(.user_transcript, "connect me with a real person")).?);
    try testing.expectEqual(EscalationReason.human_requested, classifyEscalation(eventFor(.user_transcript, "get me a Support Agent")).?);
}

test "classifyEscalation: sensitive topic keywords escalate case-insensitively" {
    try testing.expectEqual(EscalationReason.sensitive_topic, classifyEscalation(eventFor(.user_transcript, "my credit card was declined")).?);
    try testing.expectEqual(EscalationReason.sensitive_topic, classifyEscalation(eventFor(.user_transcript, "here is my SSN")).?);
    try testing.expectEqual(EscalationReason.sensitive_topic, classifyEscalation(eventFor(.user_transcript, "I need a medical diagnosis")).?);
    try testing.expectEqual(EscalationReason.sensitive_topic, classifyEscalation(eventFor(.user_transcript, "this is about debt collection")).?);
}

test "classifyEscalation: very short or low-confidence phrasing escalates" {
    try testing.expectEqual(EscalationReason.low_confidence, classifyEscalation(eventFor(.user_transcript, "hi")).?);
    try testing.expectEqual(EscalationReason.low_confidence, classifyEscalation(eventFor(.user_transcript, "I'm not sure what happened")).?);
    try testing.expectEqual(EscalationReason.low_confidence, classifyEscalation(eventFor(.user_transcript, "totally confused right now")).?);
    try testing.expectEqual(EscalationReason.low_confidence, classifyEscalation(eventFor(.user_transcript, "got an unknown error code")).?);
}

test "classifyEscalation: intelligence signal overrides transcript content, even a clean one" {
    var event = eventFor(.user_transcript, "everything is going great, thanks!");
    event.intelligence = .{ .escalation_recommended = true };
    try testing.expectEqual(EscalationReason.intelligence_signal, classifyEscalation(event).?);
}

test "classifyEscalation: intelligence signal present but not recommending escalation defers to transcript" {
    var event = eventFor(.user_transcript, "everything is going great, thanks!");
    event.intelligence = .{ .escalation_recommended = false };
    try testing.expect(classifyEscalation(event) == null);
}

test "buildLocalConversationResponse: fixed replies for setup/disconnect/interrupt/dtmf need no classification" {
    const a = testing.allocator;

    var setup = try buildLocalConversationResponse(a, eventFor(.setup, ""), "unused");
    defer setup.deinit(a);
    try testing.expect(std.mem.indexOf(u8, setup.text, "ABI support") != null);
    try testing.expect(setup.escalation == null);

    var disconnect = try buildLocalConversationResponse(a, eventFor(.disconnect, ""), "unused");
    defer disconnect.deinit(a);
    try testing.expect(std.mem.indexOf(u8, disconnect.text, "Thanks for contacting") != null);

    var interrupt = try buildLocalConversationResponse(a, eventFor(.interrupt, ""), "unused");
    defer interrupt.deinit(a);
    try testing.expect(std.mem.indexOf(u8, interrupt.text, "interruption") != null);

    var dtmf_event = eventFor(.dtmf, "");
    dtmf_event.digit = "5";
    var dtmf = try buildLocalConversationResponse(a, dtmf_event, "unused");
    defer dtmf.deinit(a);
    try testing.expect(std.mem.indexOf(u8, dtmf.text, "keypad input 5") != null);

    var dtmf_no_digit = try buildLocalConversationResponse(a, eventFor(.dtmf, ""), "unused");
    defer dtmf_no_digit.deinit(a);
    try testing.expect(std.mem.indexOf(u8, dtmf_no_digit.text, "keypad input unknown") != null);
}

test "buildLocalConversationResponse: user_transcript with no escalation carries the agent reply and memory recall" {
    const a = testing.allocator;

    var plain = try buildLocalConversationResponse(a, eventFor(.user_transcript, "what are your hours?"), "We're open 9-5.");
    defer plain.deinit(a);
    try testing.expectEqualStrings("We're open 9-5.", plain.text);
    try testing.expect(plain.escalation == null);

    var with_memory = eventFor(.user_transcript, "what are your hours?");
    with_memory.memory = .{ .recall_summary = "previously asked about returns" };
    var with_memory_response = try buildLocalConversationResponse(a, with_memory, "We're open 9-5.");
    defer with_memory_response.deinit(a);
    try testing.expect(std.mem.indexOf(u8, with_memory_response.text, "We're open 9-5.") != null);
    try testing.expect(std.mem.indexOf(u8, with_memory_response.text, "previously asked about returns") != null);
}

test "buildLocalConversationResponse: escalating user_transcript attaches an escalation payload" {
    const a = testing.allocator;

    var response = try buildLocalConversationResponse(a, eventFor(.user_transcript, "let me talk to a human"), "unused");
    defer response.deinit(a);

    try testing.expect(std.mem.indexOf(u8, response.text, "support specialist") != null);
    const payload = response.escalation.?;
    try testing.expectEqualStrings("human_requested", payload.reason_code);
    try testing.expectEqualStrings("conv-1", payload.conversation_id);
    try testing.expectEqualStrings("cust-1", payload.customer_id);
    try testing.expect(std.mem.indexOf(u8, payload.routing_hints, "priority=normal") != null);
}

test "buildEscalationPayload: sensitive_topic and intelligence_signal get high routing priority, others normal" {
    const a = testing.allocator;

    var sensitive = try buildEscalationPayload(a, eventFor(.user_transcript, "my ssn is 123"), .sensitive_topic);
    defer sensitive.deinit(a);
    try testing.expect(std.mem.indexOf(u8, sensitive.routing_hints, "priority=high") != null);

    var human = try buildEscalationPayload(a, eventFor(.user_transcript, "get me a human"), .human_requested);
    defer human.deinit(a);
    try testing.expect(std.mem.indexOf(u8, human.routing_hints, "priority=normal") != null);

    var empty = try buildEscalationPayload(a, eventFor(.user_transcript, ""), .empty_transcript);
    defer empty.deinit(a);
    try testing.expect(std.mem.indexOf(u8, empty.summary, "no usable transcript captured") != null);
}

test "parseConversationRelayEvent: recognizes all event kind aliases" {
    const a = testing.allocator;

    const cases = .{
        .{ "{\"type\":\"setup\"}", ConversationRelayEventKind.setup },
        .{ "{\"type\":\"user_transcript\"}", ConversationRelayEventKind.user_transcript },
        .{ "{\"type\":\"transcript\"}", ConversationRelayEventKind.user_transcript },
        .{ "{\"event\":\"prompt\"}", ConversationRelayEventKind.user_transcript },
        .{ "{\"type\":\"DTMF\"}", ConversationRelayEventKind.dtmf },
        .{ "{\"type\":\"interrupt\"}", ConversationRelayEventKind.interrupt },
        .{ "{\"type\":\"disconnect\"}", ConversationRelayEventKind.disconnect },
    };
    inline for (cases) |case| {
        var event = try parseConversationRelayEvent(a, case[0]);
        defer event.deinit(a);
        try testing.expectEqual(case[1], event.kind);
    }
}

test "parseConversationRelayEvent: unknown event type is InvalidResponse" {
    const a = testing.allocator;
    try testing.expectError(ConnectorError.InvalidResponse, parseConversationRelayEvent(a, "{\"type\":\"bogus\"}"));
    try testing.expectError(ConnectorError.InvalidResponse, parseConversationRelayEvent(a, "not json"));
    try testing.expectError(ConnectorError.InvalidResponse, parseConversationRelayEvent(a, "[1,2,3]"));
}

test "parseConversationRelayEvent: applies field aliases and defaults for missing ids" {
    const a = testing.allocator;
    var event = try parseConversationRelayEvent(a, "{\"type\":\"user_transcript\",\"callSid\":\"CA123\",\"from\":\"+15551234567\",\"text\":\"hello\"}");
    defer event.deinit(a);
    try testing.expectEqualStrings("CA123", event.conversation_id);
    try testing.expectEqualStrings("+15551234567", event.customer_id);
    try testing.expectEqualStrings("hello", event.transcript);

    var defaults = try parseConversationRelayEvent(a, "{\"type\":\"setup\"}");
    defer defaults.deinit(a);
    try testing.expectEqualStrings("local-conversation", defaults.conversation_id);
    try testing.expectEqualStrings("anonymous", defaults.customer_id);
}

test "parseConversationRelayEvent: parses nested memory and intelligence objects" {
    const a = testing.allocator;
    var event = try parseConversationRelayEvent(a,
        \\{"type":"user_transcript","transcript":"hi","memory":{"profile_id":"p1","recall_summary":"likes tea"},"intelligence":{"sentiment":"positive","escalation_recommended":true}}
    );
    defer event.deinit(a);

    try testing.expect(event.memory != null);
    try testing.expectEqualStrings("p1", event.memory.?.profile_id);
    try testing.expectEqualStrings("likes tea", event.memory.?.recall_summary);

    try testing.expect(event.intelligence != null);
    try testing.expectEqualStrings("positive", event.intelligence.?.sentiment);
    try testing.expect(event.intelligence.?.escalation_recommended);
}

test "parseConversationRelayEvent: non-object memory/intelligence values are InvalidResponse" {
    const a = testing.allocator;
    try testing.expectError(ConnectorError.InvalidResponse, parseConversationRelayEvent(a, "{\"type\":\"setup\",\"memory\":\"nope\"}"));
    try testing.expectError(ConnectorError.InvalidResponse, parseConversationRelayEvent(a, "{\"type\":\"setup\",\"intelligence\":42}"));
}

test "buildConversationRelayJson: round-trips text with and without an escalation payload" {
    const a = testing.allocator;

    var no_escalation_response = ConversationRelayResponse{ .text = try a.dupe(u8, "hello \"world\"") };
    defer no_escalation_response.deinit(a);
    const no_escalation = try buildConversationRelayJson(a, no_escalation_response);
    defer a.free(no_escalation);
    try json_lib.validateJsonValue(a, no_escalation, .object);
    try testing.expect(std.mem.indexOf(u8, no_escalation, "\"escalation\":null") != null);

    const payload = try buildEscalationPayload(a, eventFor(.user_transcript, "get me a human"), .human_requested);
    var with_escalation_response = ConversationRelayResponse{ .text = try a.dupe(u8, "please hold"), .escalation = payload };
    defer with_escalation_response.deinit(a);
    const with_escalation = try buildConversationRelayJson(a, with_escalation_response);
    defer a.free(with_escalation);
    try json_lib.validateJsonValue(a, with_escalation, .object);
    try testing.expect(std.mem.indexOf(u8, with_escalation, "\"reason_code\":\"human_requested\"") != null);
}

test {
    std.testing.refAllDecls(@This());
}
