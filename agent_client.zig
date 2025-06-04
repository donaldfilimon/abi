const std = @import("std");

const Message = struct {
    role: []const u8,
    content: []const u8,
};

const Persona = struct {
    name: []const u8,
    prompt: []const u8,
};

pub const PersonaType = enum { EmpatheticAnalyst, DirectExpert, AdaptiveModerator };

/// Configuration for persona embedding and response filtering
pub const Config = struct {
    persona_weight: f32 = 1.0,
    risk_threshold: i32 = 50,
    max_response_length: usize = 1024,
};

pub const AbiError = error{
    InvalidQuery,
    EthicsViolation,
    AllocationFailed,
    ResponseTooLong,
};

pub fn personaEmbedding(input: []const u8, p: PersonaType, cfg: Config, alloc: std.mem.Allocator) ![]u8 {
    _ = cfg; // unused for now
    var list = std.ArrayList(u8).init(alloc);
    try list.appendSlice(input);
    const tag = switch (p) {
        .EmpatheticAnalyst => "[EA]",
        .DirectExpert => "[DE]",
        .AdaptiveModerator => "[AM]",
    };
    try list.appendSlice(tag);
    return list.toOwnedSlice();
}

pub fn evaluateRisk(response: []const u8, config: Config) i32 {
    var risk: i32 = 0;
    if (std.mem.indexOf(u8, response, "banned") != null) {
        risk += 100;
    }
    if (response.len > config.max_response_length) {
        risk += 25;
    }
    return risk;
}

pub fn respond(p: PersonaType, query: []const u8, alloc: std.mem.Allocator) AbiError![]u8 {
    if (query.len == 0) return AbiError.InvalidQuery;
    const base = switch (p) {
        .EmpatheticAnalyst => "I understand how you feel. Let's see how I can help.",
        .DirectExpert => "Here is the direct answer to your question.",
        .AdaptiveModerator => "I'll route your request appropriately.",
    };
    return personaEmbedding(base, p, .{}, alloc) catch |err| {
        return switch (err) {
            error.OutOfMemory => AbiError.AllocationFailed,
            else => err,
        };
    };
}

fn router(query: []const u8) PersonaType {
    if (std.mem.indexOf(u8, query, "help") != null) {
        return .EmpatheticAnalyst;
    } else if (std.mem.indexOf(u8, query, "explain") != null) {
        return .DirectExpert;
    } else {
        return .AdaptiveModerator;
    }
}

fn ethicalFilter(text: []const u8) []const u8 {
    if (std.mem.indexOf(u8, text, "banned") != null) {
        return "Content removed due to policy.";
    }
    return text;
}

test "persona routing" {
    const query1 = "help me understand";
    const query2 = "explain this concept";
    const query3 = "general question";
    try std.testing.expectEqual(router(query1), PersonaType.EmpatheticAnalyst);
    try std.testing.expectEqual(router(query2), PersonaType.DirectExpert);
    try std.testing.expectEqual(router(query3), PersonaType.AdaptiveModerator);
}

test "ethical filter" {
    const safe_text = "hello world";
    const unsafe_text = "this is banned content";
    try std.testing.expectEqualStrings(ethicalFilter(safe_text), safe_text);
    try std.testing.expectEqualStrings(ethicalFilter(unsafe_text), "Content removed due to policy.");
}

pub const personas = [_]Persona{
    .{ .name = "Abbey", .prompt = "You are Abbey, an empathetic polymath who provides detailed yet supportive answers. Blend compassion with technical knowledge when assisting the user." },
    .{ .name = "Aviva", .prompt = "You are Aviva, an unfiltered expert. Provide concise factual answers without unnecessary embellishment." },
    .{ .name = "Abi", .prompt = "You are Abi, an adaptive moderator. Decide whether a request is best answered empathetically or factually and respond accordingly." },
};

fn findPersona(name: []const u8) ?Persona {
    for (personas) |p| {
        if (std.ascii.eqlIgnoreCase(p.name, name)) return p;
    }
    return null;
}

fn buildMessages(allocator: std.mem.Allocator, persona_prompt: []const u8, history: []Message, user_input: []const u8) ![]u8 {
    var list = std.ArrayList(u8).init(allocator);
    const w = list.writer();
    try w.writeAll("[");
    try w.print("{{\"role\":\"system\",\"content\":\"{s}\"}}", .{persona_prompt});
    for (history) |msg| {
        try w.print(",{{\"role\":\"{s}\",\"content\":\"{s}\"}}", .{ msg.role, msg.content });
    }
    try w.print(",{{\"role\":\"user\",\"content\":\"{s}\"}}]", .{user_input});
    return list.toOwnedSlice();
}

fn generateResponse(allocator: std.mem.Allocator, persona: Persona, api_key: []const u8, history: *std.ArrayList(Message), user_input: []const u8) ![]u8 {
    const msg_json = try buildMessages(allocator, persona.prompt, history.items, user_input);
    defer allocator.free(msg_json);
    const payload = try std.fmt.allocPrint(allocator, "{{\"model\":\"gpt-3.5-turbo\",\"messages\":{s}}}", .{msg_json});
    defer allocator.free(payload);

    var auth_header_buf: [256]u8 = undefined;
    const auth_header = try std.fmt.bufPrint(&auth_header_buf, "Authorization: Bearer {s}", .{api_key});

    const result = try std.ChildProcess.run(.{
        .allocator = allocator,
        .argv = &.{
            "curl",                                       "-sS",
            "-H",                                         auth_header,
            "-H",                                         "Content-Type: application/json",
            "-d",                                         payload,
            "https://api.openai.com/v1/chat/completions",
        },
    });

    if (result.stderr.len != 0) {
        std.debug.print("curl error: {s}\n", .{result.stderr});
    }

    const root = try std.json.parseFromSlice(std.json.Value, allocator, result.stdout, .{});
    defer root.deinit();
    const choices_val = root.value.object.get("choices") orelse return error.InvalidResponse;
    const first = choices_val.array.items[0];
    const message = first.object.get("message") orelse return error.InvalidResponse;
    const content = message.object.get("content") orelse return error.InvalidResponse;
    const text = content.string;
    return allocator.dupe(u8, text);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = std.process.args();
    _ = args.next();

    var persona_name: []const u8 = "Abbey";
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--persona")) {
            persona_name = args.next() orelse {
                std.log.err("--persona requires a value", .{});
                return;
            };
        } else {
            std.log.err("Unknown argument: {s}", .{arg});
            return;
        }
    }

    const persona = findPersona(persona_name) orelse {
        std.log.err("Unknown persona: {s}", .{persona_name});
        return;
    };

    const api_key = std.process.getEnvVarOwned(allocator, "OPENAI_API_KEY") catch |err| {
        if (err == error.EnvironmentVariableNotFound) {
            std.log.err("OPENAI_API_KEY environment variable not set", .{});
            return;
        }
        return err;
    };
    defer allocator.free(api_key);

    var history = std.ArrayList(Message).init(allocator);
    defer history.deinit();

    const stdin = std.io.getStdIn().reader();
    const stdout = std.io.getStdOut().writer();

    try stdout.print("Starting session with {s}. Type 'quit' to exit.\n", .{persona.name});
    var buf: [1024]u8 = undefined;
    while (true) {
        try stdout.writeAll("you> ");
        const line = (try stdin.readUntilDelimiterOrEof(&buf, '\n')) orelse break;
        const trimmed = std.mem.trimRight(u8, line, " \t\r\n");
        if (std.ascii.eqlIgnoreCase(trimmed, "quit")) break;

        defer history.append(.{ .role = "user", .content = allocator.dupe(u8, trimmed) catch trimmed }) catch {};
        const reply = generateResponse(allocator, persona, api_key, &history, trimmed) catch |err| {
            try stdout.print("error: {s}\n", .{@errorName(err)});
            continue;
        };
        defer allocator.free(reply);
        try history.append(.{ .role = "assistant", .content = reply });
        try stdout.print("{s}> {s}\n", .{ persona.name, reply });
    }
}
