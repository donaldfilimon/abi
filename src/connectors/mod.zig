const std = @import("std");

pub const ConnectorError = error{
    OutOfMemory,
    ConnectionFailed,
    AuthenticationError,
    RateLimited,
    InvalidResponse,
    Timeout,
    LiveTransportUnavailable,
};

pub const TransportMode = enum {
    local,
    live,
};

pub const ConnectorConfig = struct {
    api_key: []const u8,
    base_url: []const u8,
    timeout_ms: u32 = 30000,
    transport: TransportMode = .local,
};

pub fn transportModeName(mode: TransportMode) []const u8 {
    return switch (mode) {
        .local => "local",
        .live => "live",
    };
}

pub const Response = struct {
    status: u16,
    body: []u8,
    owned: bool = true,

    pub fn deinit(self: *Response, allocator: std.mem.Allocator) void {
        if (self.owned and self.body.len > 0) {
            allocator.free(self.body);
            self.body = "";
        }
    }
};

pub const openai = struct {
    pub const Client = struct {
        allocator: std.mem.Allocator,
        config: ConnectorConfig,

        pub fn init(allocator: std.mem.Allocator, config: ConnectorConfig) Client {
            return .{
                .allocator = allocator,
                .config = config,
            };
        }

        pub fn deinit(self: *Client) void {
            _ = self;
        }

        pub fn chatCompletion(
            self: *Client,
            allocator: std.mem.Allocator,
            model: []const u8,
            messages: []const u8,
        ) ConnectorError!Response {
            try validateConnectorConfig(self.config);
            if (self.config.transport == .live) return ConnectorError.LiveTransportUnavailable;
            const body = try buildOpenAiBody(allocator, model, messages, false);
            defer allocator.free(body);

            std.log.info("OpenAI-compatible local chat request for model {s} via {s}", .{ model, self.config.base_url });

            return .{
                .status = 200,
                .body = try openAiLocalResponse(allocator, model, messages, body.len),
                .owned = true,
            };
        }

        pub fn chatCompletionLive(
            self: *Client,
            io: std.Io,
            allocator: std.mem.Allocator,
            model: []const u8,
            messages: []const u8,
        ) ConnectorError!Response {
            try validateConnectorConfig(self.config);
            const body = try buildOpenAiBody(allocator, model, messages, false);
            defer allocator.free(body);
            const authorization = try bearerHeader(allocator, self.config.api_key);
            defer allocator.free(authorization);
            return httpPostJson(io, allocator, self.config, "/v1/chat/completions", body, &.{
                .{ .name = "authorization", .value = authorization },
            });
        }

        pub fn streamChatCompletion(
            self: *Client,
            allocator: std.mem.Allocator,
            model: []const u8,
            messages: []const u8,
        ) ConnectorError!Response {
            try validateConnectorConfig(self.config);
            if (self.config.transport == .live) return ConnectorError.LiveTransportUnavailable;
            const body = try buildOpenAiBody(allocator, model, messages, true);
            defer allocator.free(body);

            std.log.info("OpenAI-compatible local streaming request for model {s} via {s}", .{ model, self.config.base_url });

            return .{
                .status = 200,
                .body = try openAiLocalStream(allocator, model, messages, body.len),
                .owned = true,
            };
        }

        pub fn streamChatCompletionLive(
            self: *Client,
            io: std.Io,
            allocator: std.mem.Allocator,
            model: []const u8,
            messages: []const u8,
        ) ConnectorError!Response {
            try validateConnectorConfig(self.config);
            const body = try buildOpenAiBody(allocator, model, messages, true);
            defer allocator.free(body);
            const authorization = try bearerHeader(allocator, self.config.api_key);
            defer allocator.free(authorization);
            return httpPostJson(io, allocator, self.config, "/v1/chat/completions", body, &.{
                .{ .name = "authorization", .value = authorization },
            });
        }
    };
};

pub const anthropic = struct {
    pub const Client = struct {
        allocator: std.mem.Allocator,
        config: ConnectorConfig,

        pub fn init(allocator: std.mem.Allocator, config: ConnectorConfig) Client {
            return .{
                .allocator = allocator,
                .config = config,
            };
        }

        pub fn deinit(self: *Client) void {
            _ = self;
        }

        pub fn message(
            self: *Client,
            allocator: std.mem.Allocator,
            model: []const u8,
            prompt: []const u8,
            max_tokens: u32,
        ) ConnectorError!Response {
            try validateConnectorConfig(self.config);
            if (self.config.transport == .live) return ConnectorError.LiveTransportUnavailable;
            const body = try buildAnthropicBody(allocator, model, prompt, max_tokens, false);
            defer allocator.free(body);

            std.log.info("Anthropic-compatible local message request for model {s} via {s}", .{ model, self.config.base_url });

            return .{
                .status = 200,
                .body = try anthropicLocalResponse(allocator, model, prompt, max_tokens, body.len),
                .owned = true,
            };
        }

        pub fn messageLive(
            self: *Client,
            io: std.Io,
            allocator: std.mem.Allocator,
            model: []const u8,
            prompt: []const u8,
            max_tokens: u32,
        ) ConnectorError!Response {
            try validateConnectorConfig(self.config);
            const body = try buildAnthropicBody(allocator, model, prompt, max_tokens, false);
            defer allocator.free(body);
            return httpPostJson(io, allocator, self.config, "/v1/messages", body, &.{
                .{ .name = "x-api-key", .value = self.config.api_key },
                .{ .name = "anthropic-version", .value = "2023-06-01" },
            });
        }

        pub fn streamMessage(
            self: *Client,
            allocator: std.mem.Allocator,
            model: []const u8,
            prompt: []const u8,
            max_tokens: u32,
        ) ConnectorError!Response {
            try validateConnectorConfig(self.config);
            if (self.config.transport == .live) return ConnectorError.LiveTransportUnavailable;
            const body = try buildAnthropicBody(allocator, model, prompt, max_tokens, true);
            defer allocator.free(body);

            std.log.info("Anthropic-compatible local streaming request for model {s} via {s}", .{ model, self.config.base_url });

            return .{
                .status = 200,
                .body = try anthropicLocalStream(allocator, model, prompt, max_tokens, body.len),
                .owned = true,
            };
        }

        pub fn streamMessageLive(
            self: *Client,
            io: std.Io,
            allocator: std.mem.Allocator,
            model: []const u8,
            prompt: []const u8,
            max_tokens: u32,
        ) ConnectorError!Response {
            try validateConnectorConfig(self.config);
            const body = try buildAnthropicBody(allocator, model, prompt, max_tokens, true);
            defer allocator.free(body);
            return httpPostJson(io, allocator, self.config, "/v1/messages", body, &.{
                .{ .name = "x-api-key", .value = self.config.api_key },
                .{ .name = "anthropic-version", .value = "2023-06-01" },
            });
        }
    };
};

pub const discord = struct {
    pub const BotConfig = struct {
        token: []const u8,
        client_id: []const u8,
        intents: u32 = 0xFFFF,
        transport: TransportMode = .local,
    };

    pub const Bot = struct {
        allocator: std.mem.Allocator,
        config: BotConfig,
        connected: bool = false,

        pub fn init(allocator: std.mem.Allocator, config: BotConfig) Bot {
            return .{
                .allocator = allocator,
                .config = config,
                .connected = false,
            };
        }

        pub fn deinit(self: *Bot) void {
            if (self.connected) {
                std.log.info("Discord bot disconnecting", .{});
                self.connected = false;
            }
        }

        pub fn connect(self: *Bot) !void {
            if (self.config.token.len == 0) return ConnectorError.AuthenticationError;
            if (self.config.client_id.len == 0) return ConnectorError.AuthenticationError;
            if (self.config.transport == .live) return ConnectorError.LiveTransportUnavailable;
            std.log.info("Discord bot opened local session for client {s}", .{self.config.client_id});
            self.connected = true;
        }

        pub fn sendMessage(
            self: *Bot,
            allocator: std.mem.Allocator,
            channel_id: []const u8,
            content: []const u8,
        ) ![]u8 {
            if (!self.connected) return ConnectorError.ConnectionFailed;

            if (channel_id.len == 0 or content.len == 0) return ConnectorError.InvalidResponse;
            std.log.info("Discord local send to channel {s}: {s}", .{ channel_id, content });

            return try discordLocalAck(allocator, channel_id, content);
        }

        pub fn sendMessageLive(
            self: *Bot,
            io: std.Io,
            allocator: std.mem.Allocator,
            channel_id: []const u8,
            content: []const u8,
        ) ConnectorError!Response {
            if (self.config.token.len == 0) return ConnectorError.AuthenticationError;
            if (channel_id.len == 0 or content.len == 0) return ConnectorError.InvalidResponse;
            const body = try buildDiscordMessageBody(allocator, content);
            defer allocator.free(body);
            const authorization = try botHeader(allocator, self.config.token);
            defer allocator.free(authorization);
            const path = try std.fmt.allocPrint(allocator, "/api/v10/channels/{s}/messages", .{channel_id});
            defer allocator.free(path);
            return httpPostJson(io, allocator, .{
                .api_key = self.config.token,
                .base_url = "https://discord.com",
                .timeout_ms = 30000,
                .transport = .live,
            }, path, body, &.{
                .{ .name = "authorization", .value = authorization },
            });
        }

        pub fn handleMessage(self: *Bot, author: []const u8, content: []const u8) ![]const u8 {
            if (!self.connected) return ConnectorError.ConnectionFailed;

            if (author.len == 0 or content.len == 0) return ConnectorError.InvalidResponse;
            std.log.info("Discord local receive from {s}: {s}", .{ author, content });

            return try std.fmt.allocPrint(
                self.allocator,
                "processed message from {s}",
                .{author},
            );
        }
    };
};

pub const twilio = struct {
    pub const TwilioConfig = struct {
        account_sid: []const u8,
        auth_token: []const u8,
        base_url: []const u8 = "https://api.twilio.com",
        timeout_ms: u32 = 30000,
        transport: TransportMode = .local,
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
            const twiml = try buildTwiMLSay(allocator, agent_reply, escalation != null, escalation_url);
            defer allocator.free(twiml);

            const form = try buildUrlEncodedForm(allocator, &.{
                .{ "Twiml", twiml },
                .{ "conversation_id", event.conversation_id },
                .{ "customer_id", event.customer_id },
            });
            defer allocator.free(form);

            const auth = try basicAuthHeader(allocator, self.config.account_sid, self.config.auth_token);
            defer allocator.free(auth);

            const path = try std.fmt.allocPrint(allocator, "/2010-04-01/Accounts/{s}/Calls.json", .{self.config.account_sid});
            defer allocator.free(path);

            const http_response = try httpPostForm(io, allocator, .{
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
        try appendJsonString(&out, allocator, response.text);
        try out.appendSlice(allocator, ",\"escalation\":");
        if (response.escalation) |payload| {
            try out.append(allocator, '{');
            try out.appendSlice(allocator, "\"conversation_id\":");
            try appendJsonString(&out, allocator, payload.conversation_id);
            try out.appendSlice(allocator, ",\"customer_id\":");
            try appendJsonString(&out, allocator, payload.customer_id);
            try out.appendSlice(allocator, ",\"reason_code\":");
            try appendJsonString(&out, allocator, payload.reason_code);
            try out.appendSlice(allocator, ",\"summary\":");
            try appendJsonString(&out, allocator, payload.summary);
            try out.appendSlice(allocator, ",\"routing_hints\":");
            try appendJsonString(&out, allocator, payload.routing_hints);
            try out.append(allocator, '}');
        } else {
            try out.appendSlice(allocator, "null");
        }
        try out.append(allocator, '}');
        return try out.toOwnedSlice(allocator);
    }

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
};

fn validateConnectorConfig(config: ConnectorConfig) ConnectorError!void {
    if (config.api_key.len == 0) return ConnectorError.AuthenticationError;
    if (config.base_url.len == 0) return ConnectorError.ConnectionFailed;
    if (config.timeout_ms == 0) return ConnectorError.Timeout;
}

fn httpPostForm(
    io: std.Io,
    allocator: std.mem.Allocator,
    config: ConnectorConfig,
    path: []const u8,
    body: []const u8,
    extra_headers: []const std.http.Header,
) ConnectorError!Response {
    if (config.transport != .live) return ConnectorError.LiveTransportUnavailable;

    const url = try joinUrl(allocator, config.base_url, path);
    defer allocator.free(url);

    var client = std.http.Client{ .allocator = allocator, .io = io };
    defer client.deinit();

    var response_writer = std.Io.Writer.Allocating.init(allocator);
    defer response_writer.deinit();

    const result = client.fetch(.{
        .location = .{ .url = url },
        .method = .POST,
        .payload = body,
        .headers = .{ .content_type = .{ .override = "application/x-www-form-urlencoded" } },
        .extra_headers = extra_headers,
        .response_writer = &response_writer.writer,
        .redirect_behavior = .unhandled,
        .keep_alive = false,
    }) catch |err| return mapHttpError(err);

    try mapHttpStatus(result.status);
    return .{
        .status = @intCast(@intFromEnum(result.status)),
        .body = try response_writer.toOwnedSlice(),
        .owned = true,
    };
}

fn httpPostJson(
    io: std.Io,
    allocator: std.mem.Allocator,
    config: ConnectorConfig,
    path: []const u8,
    body: []const u8,
    extra_headers: []const std.http.Header,
) ConnectorError!Response {
    if (config.transport != .live) return ConnectorError.LiveTransportUnavailable;

    const url = try joinUrl(allocator, config.base_url, path);
    defer allocator.free(url);

    var client = std.http.Client{ .allocator = allocator, .io = io };
    defer client.deinit();

    var response_writer = std.Io.Writer.Allocating.init(allocator);
    defer response_writer.deinit();

    const result = client.fetch(.{
        .location = .{ .url = url },
        .method = .POST,
        .payload = body,
        .headers = .{ .content_type = .{ .override = "application/json" } },
        .extra_headers = extra_headers,
        .response_writer = &response_writer.writer,
        .redirect_behavior = .unhandled,
        .keep_alive = false,
    }) catch |err| return mapHttpError(err);

    try mapHttpStatus(result.status);
    return .{
        .status = @intCast(@intFromEnum(result.status)),
        .body = try response_writer.toOwnedSlice(),
        .owned = true,
    };
}

fn mapHttpError(err: anyerror) ConnectorError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.Timeout => error.Timeout,
        error.AuthenticationError => error.AuthenticationError,
        else => error.ConnectionFailed,
    };
}

fn mapHttpStatus(status: std.http.Status) ConnectorError!void {
    const code: u16 = @intCast(@intFromEnum(status));
    return switch (code) {
        200...299 => {},
        401, 403 => ConnectorError.AuthenticationError,
        408, 504 => ConnectorError.Timeout,
        429 => ConnectorError.RateLimited,
        else => ConnectorError.InvalidResponse,
    };
}

fn joinUrl(allocator: std.mem.Allocator, base_url: []const u8, path: []const u8) ConnectorError![]u8 {
    if (base_url.len == 0 or path.len == 0) return ConnectorError.ConnectionFailed;
    const base_has_slash = base_url[base_url.len - 1] == '/';
    const path_has_slash = path[0] == '/';
    if (base_has_slash and path_has_slash) return try std.fmt.allocPrint(allocator, "{s}{s}", .{ base_url, path[1..] });
    if (!base_has_slash and !path_has_slash) return try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base_url, path });
    return try std.fmt.allocPrint(allocator, "{s}{s}", .{ base_url, path });
}

fn bearerHeader(allocator: std.mem.Allocator, api_key: []const u8) ConnectorError![]u8 {
    return try std.fmt.allocPrint(allocator, "Bearer {s}", .{api_key});
}

fn botHeader(allocator: std.mem.Allocator, token: []const u8) ConnectorError![]u8 {
    return try std.fmt.allocPrint(allocator, "Bot {s}", .{token});
}

fn basicAuthHeader(allocator: std.mem.Allocator, username: []const u8, password: []const u8) ConnectorError![]u8 {
    const combined = try std.fmt.allocPrint(allocator, "{s}:{s}", .{ username, password });
    defer allocator.free(combined);

    var encoder = std.base64.standard.Encoder.init(.{});
    const encoded_len = encoder.calcLength(combined.len);
    const encoded = try allocator.alloc(u8, encoded_len);
    defer allocator.free(encoded);

    _ = encoder.encode(encoded, combined);
    return try std.fmt.allocPrint(allocator, "Basic {s}", .{encoded});
}

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

fn buildOpenAiBody(allocator: std.mem.Allocator, model: []const u8, messages: []const u8, stream: bool) ConnectorError![]u8 {
    try validateJsonValue(allocator, messages, .array);
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"model\":");
    try appendJsonString(&out, allocator, model);
    try out.appendSlice(allocator, ",\"messages\":");
    try out.appendSlice(allocator, messages);
    if (stream) try out.appendSlice(allocator, ",\"stream\":true");
    try out.append(allocator, '}');
    return try out.toOwnedSlice(allocator);
}

fn buildAnthropicBody(allocator: std.mem.Allocator, model: []const u8, prompt: []const u8, max_tokens: u32, stream: bool) ConnectorError![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"model\":");
    try appendJsonString(&out, allocator, model);
    try out.print(allocator, ",\"max_tokens\":{d}", .{max_tokens});
    if (stream) try out.appendSlice(allocator, ",\"stream\":true");
    try out.appendSlice(allocator, ",\"messages\":[{\"role\":\"user\",\"content\":");
    try appendJsonString(&out, allocator, prompt);
    try out.appendSlice(allocator, "}]}");
    return try out.toOwnedSlice(allocator);
}

fn buildDiscordMessageBody(allocator: std.mem.Allocator, content: []const u8) ConnectorError![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"content\":");
    try appendJsonString(&out, allocator, content);
    try out.append(allocator, '}');
    return try out.toOwnedSlice(allocator);
}

const ExpectedJsonKind = enum { array, object, any };

fn validateJsonValue(allocator: std.mem.Allocator, input: []const u8, expected: ExpectedJsonKind) ConnectorError!void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, input, .{}) catch return ConnectorError.InvalidResponse;
    defer parsed.deinit();
    switch (expected) {
        .any => {},
        .array => switch (parsed.value) {
            .array => {},
            else => return ConnectorError.InvalidResponse,
        },
        .object => switch (parsed.value) {
            .object => {},
            else => return ConnectorError.InvalidResponse,
        },
    }
}

fn openAiLocalResponse(allocator: std.mem.Allocator, model: []const u8, messages: []const u8, body_len: usize) ![]u8 {
    const text = try std.fmt.allocPrint(
        allocator,
        "OpenAI-compatible local response model={s} messages_bytes={d} request_bytes={d}",
        .{ model, messages.len, body_len },
    );
    defer allocator.free(text);
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":");
    try appendJsonString(&out, allocator, text);
    try out.appendSlice(allocator, "}}]}");
    return try out.toOwnedSlice(allocator);
}

fn openAiLocalStream(allocator: std.mem.Allocator, model: []const u8, messages: []const u8, body_len: usize) ![]u8 {
    const content = try std.fmt.allocPrint(allocator, "OpenAI-compatible local stream model={s} messages_bytes={d} request_bytes={d}", .{ model, messages.len, body_len });
    defer allocator.free(content);
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "data: ");
    try appendOpenAiDelta(&out, allocator, content);
    try out.appendSlice(allocator, "\n\ndata: [DONE]\n\n");
    return try out.toOwnedSlice(allocator);
}

fn appendOpenAiDelta(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, content: []const u8) !void {
    try out.appendSlice(allocator, "{\"choices\":[{\"delta\":{\"content\":");
    try appendJsonString(out, allocator, content);
    try out.appendSlice(allocator, "}}]}");
}

fn anthropicLocalResponse(allocator: std.mem.Allocator, model: []const u8, prompt: []const u8, max_tokens: u32, body_len: usize) ![]u8 {
    const text = try std.fmt.allocPrint(
        allocator,
        "Anthropic-compatible local response model={s} prompt_bytes={d} max_tokens={d} request_bytes={d}",
        .{ model, prompt.len, max_tokens, body_len },
    );
    defer allocator.free(text);
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"content\":[{\"type\":\"text\",\"text\":");
    try appendJsonString(&out, allocator, text);
    try out.appendSlice(allocator, "}]}");
    return try out.toOwnedSlice(allocator);
}

fn anthropicLocalStream(allocator: std.mem.Allocator, model: []const u8, prompt: []const u8, max_tokens: u32, body_len: usize) ![]u8 {
    const content = try std.fmt.allocPrint(allocator, "Anthropic-compatible local stream model={s} prompt_bytes={d} max_tokens={d} request_bytes={d}", .{ model, prompt.len, max_tokens, body_len });
    defer allocator.free(content);
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "event: content_block_delta\ndata: ");
    try appendAnthropicDelta(&out, allocator, content);
    try out.appendSlice(allocator, "\n\nevent: message_stop\n\n");
    return try out.toOwnedSlice(allocator);
}

fn appendAnthropicDelta(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, content: []const u8) !void {
    try out.appendSlice(allocator, "{\"delta\":{\"text\":");
    try appendJsonString(out, allocator, content);
    try out.appendSlice(allocator, "}}");
}

fn discordLocalAck(allocator: std.mem.Allocator, channel_id: []const u8, content: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\"status\":\"queued-local\",\"channel_id\":");
    try appendJsonString(&out, allocator, channel_id);
    try out.print(allocator, ",\"content_bytes\":{d}}}", .{content.len});
    return try out.toOwnedSlice(allocator);
}

fn appendJsonString(out: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, value: []const u8) !void {
    try out.append(allocator, '"');
    for (value) |byte| {
        switch (byte) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            0x00...0x07 => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            0x08 => try out.appendSlice(allocator, "\\b"),
            0x0c => try out.appendSlice(allocator, "\\f"),
            0x0b => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            0x0e...0x1f => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            else => try out.append(allocator, byte),
        }
    }
    try out.append(allocator, '"');
}

test {
    std.testing.refAllDecls(@This());
}

test "openai client init and deinit" {
    const allocator = std.testing.allocator;
    var client = openai.Client.init(allocator, .{
        .api_key = "test-key",
        .base_url = "https://api.openai.com",
    });
    defer client.deinit();
    try std.testing.expectEqualStrings("https://api.openai.com", client.config.base_url);
}

test "openai chatCompletion returns response" {
    const allocator = std.testing.allocator;
    var client = openai.Client.init(allocator, .{
        .api_key = "test-key",
        .base_url = "https://api.openai.com",
    });
    defer client.deinit();

    var response = try client.chatCompletion(allocator, "gpt-4", "[{\"role\":\"user\",\"content\":\"hello\"}]");
    defer response.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), response.status);
    try std.testing.expect(response.body.len > 0);
}

test "openai body builder validates messages json and escapes model" {
    const allocator = std.testing.allocator;
    const body = try buildOpenAiBody(allocator, "gpt-\"quoted\"", "[{\"role\":\"user\",\"content\":\"hello\"}]", true);
    defer allocator.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "gpt-\\\"quoted\\\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"stream\":true") != null);
    try std.testing.expectError(ConnectorError.InvalidResponse, buildOpenAiBody(allocator, "gpt", "{}", false));
}

test "openai live transport is explicit opt-in boundary" {
    const allocator = std.testing.allocator;
    var client = openai.Client.init(allocator, .{
        .api_key = "test-key",
        .base_url = "https://api.openai.com",
        .transport = .live,
    });
    defer client.deinit();

    try std.testing.expectError(
        ConnectorError.LiveTransportUnavailable,
        client.chatCompletion(allocator, "gpt-4", "[]"),
    );
}

test "anthropic client init and deinit" {
    const allocator = std.testing.allocator;
    var client = anthropic.Client.init(allocator, .{
        .api_key = "test-key",
        .base_url = "https://api.anthropic.com",
    });
    defer client.deinit();
    try std.testing.expectEqualStrings("https://api.anthropic.com", client.config.base_url);
}

test "anthropic message returns response" {
    const allocator = std.testing.allocator;
    var client = anthropic.Client.init(allocator, .{
        .api_key = "test-key",
        .base_url = "https://api.anthropic.com",
    });
    defer client.deinit();

    var response = try client.message(allocator, "claude-3", "hello", 1024);
    defer response.deinit(allocator);
    try std.testing.expectEqual(@as(u16, 200), response.status);
}

test "anthropic body builder escapes prompt" {
    const allocator = std.testing.allocator;
    const body = try buildAnthropicBody(allocator, "claude", "hello \"world\"", 128, false);
    defer allocator.free(body);
    try std.testing.expect(std.mem.indexOf(u8, body, "hello \\\"world\\\"") != null);
}

test "live url helpers build expected values" {
    const allocator = std.testing.allocator;
    const url = try joinUrl(allocator, "https://api.openai.com/", "/v1/chat/completions");
    defer allocator.free(url);
    try std.testing.expectEqualStrings("https://api.openai.com/v1/chat/completions", url);

    const bearer = try bearerHeader(allocator, "key");
    defer allocator.free(bearer);
    try std.testing.expectEqualStrings("Bearer key", bearer);
}

test "discord bot connect fails without token" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "",
        .client_id = "test",
    });
    defer bot.deinit();
    try std.testing.expectError(ConnectorError.AuthenticationError, bot.connect());
}

test "discord bot connect succeeds with token" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "valid-token",
        .client_id = "test",
    });
    defer bot.deinit();
    try bot.connect();
    try std.testing.expect(bot.connected);
}

test "discord bot send message requires connection" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "valid-token",
        .client_id = "test",
    });
    defer bot.deinit();
    try std.testing.expectError(
        ConnectorError.ConnectionFailed,
        bot.sendMessage(allocator, "channel-1", "hello"),
    );
}

test "discord live transport is explicit opt-in boundary" {
    const allocator = std.testing.allocator;
    var bot = discord.Bot.init(allocator, .{
        .token = "valid-token",
        .client_id = "test",
        .transport = .live,
    });
    defer bot.deinit();
    try std.testing.expectError(ConnectorError.LiveTransportUnavailable, bot.connect());
}

test "twilio config validation requires credentials" {
    try std.testing.expectError(ConnectorError.AuthenticationError, twilio.validateTwilioConfig(.{
        .account_sid = "",
        .auth_token = "token",
    }));
    try std.testing.expectError(ConnectorError.AuthenticationError, twilio.validateTwilioConfig(.{
        .account_sid = "AC123",
        .auth_token = "",
    }));
    try std.testing.expectError(ConnectorError.Timeout, twilio.validateTwilioConfig(.{
        .account_sid = "AC123",
        .auth_token = "token",
        .timeout_ms = 0,
    }));
}

test "twilio parses ConversationRelay transcript event" {
    const allocator = std.testing.allocator;
    var event = try twilio.parseConversationRelayEvent(allocator,
        \\{"type":"transcript","conversationId":"CA123","customerId":"customer-1","transcript":"I need help","memory":{"profile_id":"profile-1","recall_summary":"prefers SMS"}}
    );
    defer event.deinit(allocator);

    try std.testing.expectEqual(.user_transcript, event.kind);
    try std.testing.expectEqualStrings("CA123", event.conversation_id);
    try std.testing.expectEqualStrings("customer-1", event.customer_id);
    try std.testing.expectEqualStrings("I need help", event.transcript);
    try std.testing.expect(event.memory != null);
    try std.testing.expectEqualStrings("prefers SMS", event.memory.?.recall_summary);
}

test "twilio local response includes memory context" {
    const allocator = std.testing.allocator;
    var client = twilio.Client.init(allocator, twilio.TwilioConfig.local());
    defer client.deinit();

    var response = try client.handleConversationRelayEvent(allocator, .{
        .kind = .user_transcript,
        .conversation_id = "CA123",
        .customer_id = "customer-1",
        .transcript = "I need order help",
        .memory = .{ .recall_summary = "customer has an open shipping case" },
    }, "Abi can help with that order.");
    defer response.deinit(allocator);

    try std.testing.expect(response.escalation == null);
    try std.testing.expect(std.mem.indexOf(u8, response.text, "open shipping case") != null);
}

test "twilio local response builds escalation payload" {
    const allocator = std.testing.allocator;
    var client = twilio.Client.init(allocator, twilio.TwilioConfig.local());
    defer client.deinit();

    var response = try client.handleConversationRelayEvent(allocator, .{
        .kind = .user_transcript,
        .conversation_id = "CA456",
        .customer_id = "customer-2",
        .transcript = "Please connect me to a human representative",
    }, "Abi can help.");
    defer response.deinit(allocator);

    try std.testing.expect(response.escalation != null);
    try std.testing.expectEqualStrings("human_requested", response.escalation.?.reason_code);

    const json = try twilio.buildConversationRelayJson(allocator, response);
    defer allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"escalation\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "human_requested") != null);
}

test "twilio live transport is explicit boundary" {
    const allocator = std.testing.allocator;
    var client = twilio.Client.init(allocator, .{
        .account_sid = "AC123",
        .auth_token = "token",
        .transport = .live,
    });
    defer client.deinit();

    try std.testing.expectError(ConnectorError.LiveTransportUnavailable, client.handleConversationRelayEvent(allocator, .{
        .kind = .user_transcript,
        .conversation_id = "CA789",
        .customer_id = "customer-3",
        .transcript = "hello",
    }, "hello"));
}
