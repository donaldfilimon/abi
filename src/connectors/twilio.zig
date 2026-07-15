const std = @import("std");
const connector = @import("connector.zig");
const http = @import("http.zig");
const relay = @import("twilio_relay.zig");

const ConnectorError = connector.ConnectorError;
const Response = connector.Response;

const TWILIO_ACCOUNT_SID_BYTES: usize = 34;
const TWILIO_AUTH_TOKEN_BYTES: usize = 32;

pub const TwilioConfig = struct {
    account_sid: []const u8,
    auth_token: []const u8,
    base_url: []const u8 = "https://api.twilio.com",
    timeout_ms: u32 = 30000,
    transport: connector.TransportMode = .local,
    escalation_url: []const u8 = "",

    pub fn local() TwilioConfig {
        return .{
            .account_sid = "AC" ++ "0123456789abcdef0123456789abcdef",
            .auth_token = "0123456789abcdef0123456789abcdef",
            .transport = .local,
        };
    }
};

pub const ConversationRelayEventKind = relay.ConversationRelayEventKind;
pub const EscalationReason = relay.EscalationReason;
pub const ConversationMemory = relay.ConversationMemory;
pub const IntelligenceSignal = relay.IntelligenceSignal;
pub const ConversationRelayEvent = relay.ConversationRelayEvent;
pub const EscalationPayload = relay.EscalationPayload;
pub const ConversationRelayResponse = relay.ConversationRelayResponse;

pub const parseConversationRelayEvent = relay.parseConversationRelayEvent;
pub const buildConversationRelayJson = relay.buildConversationRelayJson;
pub const buildLocalConversationResponse = relay.buildLocalConversationResponse;
pub const buildEscalationPayload = relay.buildEscalationPayload;
pub const classifyEscalation = relay.classifyEscalation;

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
        if (self.config.transport != .live) return ConnectorError.LiveTransportUnavailable;
        try validateTwilioConfig(self.config);

        // Only classify escalation from actual user transcripts — mirror the
        // local handler's per-kind guard. Lifecycle events (`setup`/`disconnect`/
        // `interrupt`/`dtmf`) carry an empty transcript, which `classifyEscalation`
        // would otherwise read as `.empty_transcript` and wrongly redirect the
        // call on connect.
        const escalation = if (event.kind == .user_transcript) classifyEscalation(event) else null;
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

        var http_response = try http.httpPostForm(io, allocator, .{
            .api_key = self.config.account_sid,
            .base_url = self.config.base_url,
            .timeout_ms = self.config.timeout_ms,
            .transport = .live,
        }, path, form, &.{
            .{ .name = "authorization", .value = auth },
        });
        defer http_response.deinit(allocator);

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

        const log_summary = redactedLiveResponseSummary(http_response);
        std.log.info("Twilio live response status={d} body_bytes={d} body=redacted", .{ log_summary.status, log_summary.body_bytes });
        return result;
    }
};

pub fn validateTwilioConfig(config: TwilioConfig) ConnectorError!void {
    try validateTwilioAccountSid(config.account_sid);
    try validateTwilioAuthToken(config.auth_token);
    if (config.base_url.len == 0) return ConnectorError.ConnectionFailed;
    if (config.timeout_ms == 0) return ConnectorError.Timeout;
}

pub fn validateTwilioAccountSid(account_sid: []const u8) ConnectorError!void {
    if (account_sid.len != TWILIO_ACCOUNT_SID_BYTES) return ConnectorError.AuthenticationError;
    if (!std.mem.startsWith(u8, account_sid, "AC")) return ConnectorError.AuthenticationError;
    for (account_sid[2..]) |byte| {
        if (!std.ascii.isHex(byte)) return ConnectorError.AuthenticationError;
    }
}

pub fn validateTwilioAuthToken(auth_token: []const u8) ConnectorError!void {
    if (auth_token.len != TWILIO_AUTH_TOKEN_BYTES) return ConnectorError.AuthenticationError;
    for (auth_token) |byte| {
        if (!std.ascii.isHex(byte)) return ConnectorError.AuthenticationError;
    }
}

const RedactedLiveResponseSummary = struct {
    status: u16,
    body_bytes: usize,
};

fn redactedLiveResponseSummary(response: Response) RedactedLiveResponseSummary {
    return .{
        .status = response.status,
        .body_bytes = response.body.len,
    };
}

// --- Twilio XML/Form helpers ---

fn buildTwiMLSay(allocator: std.mem.Allocator, text: []const u8, escalate: bool, escalation_url: []const u8) ConnectorError![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response><Say>");
    try appendXmlText(&out, allocator, text);
    try out.appendSlice(allocator, "</Say>");
    if (escalate and escalation_url.len > 0) {
        try out.appendSlice(allocator, "<Redirect>");
        try appendXmlText(&out, allocator, escalation_url);
        try out.appendSlice(allocator, "</Redirect>");
    }
    try out.appendSlice(allocator, "</Response>");
    return try out.toOwnedSlice(allocator);
}

fn appendXmlText(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, text: []const u8) ConnectorError!void {
    for (text) |byte| {
        switch (byte) {
            '<' => try out.appendSlice(allocator, "&lt;"),
            '>' => try out.appendSlice(allocator, "&gt;"),
            '&' => try out.appendSlice(allocator, "&amp;"),
            '"' => try out.appendSlice(allocator, "&quot;"),
            else => try out.append(allocator, byte),
        }
    }
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

test "Twilio TwiML helper escapes text and redirect URL" {
    const twiml = try buildTwiMLSay(std.testing.allocator, "hello <ABI> & \"team\"", true, "https://example.com/escalate?a=1&b=<2>");
    defer std.testing.allocator.free(twiml);

    try std.testing.expect(std.mem.indexOf(u8, twiml, "hello &lt;ABI&gt; &amp; &quot;team&quot;") != null);
    try std.testing.expect(std.mem.indexOf(u8, twiml, "https://example.com/escalate?a=1&amp;b=&lt;2&gt;") != null);
}

test "Twilio form helper url encodes fields" {
    const form = try buildUrlEncodedForm(std.testing.allocator, &.{
        .{ "Twiml", "<Response>hello world</Response>" },
        .{ "customer_id", "+15551234567" },
    });
    defer std.testing.allocator.free(form);

    try std.testing.expect(std.mem.indexOf(u8, form, "Twiml=%3CResponse%3Ehello+world%3C%2FResponse%3E") != null);
    try std.testing.expect(std.mem.indexOf(u8, form, "customer_id=%2B15551234567") != null);
}

test "Twilio redacted live response summary omits body text" {
    const response = Response{
        .status = 201,
        .body = @constCast("provider secret body"),
        .owned = false,
    };
    const summary = redactedLiveResponseSummary(response);
    try std.testing.expectEqual(@as(u16, 201), summary.status);
    try std.testing.expectEqual(response.body.len, summary.body_bytes);
}

test {
    std.testing.refAllDecls(@This());
}
