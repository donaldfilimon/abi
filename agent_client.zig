const std = @import("std");

const Message = struct {
    role: []const u8,
    content: []const u8,
};

const Persona = struct {
    name: []const u8,
    prompt: []const u8,
};

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
        try w.print(",{{\"role\":\"{s}\",\"content\":\"{s}\"}}", .{msg.role, msg.content});
    }
    try w.print(",{{\"role\":\"user\",\"content\":\"{s}\"}}]", .{user_input});
    return list.toOwnedSlice();
}

fn generateResponse(allocator: std.mem.Allocator, persona: Persona, api_key: []const u8, history: *std.ArrayList(Message), user_input: []const u8) ![]u8 {
    const msg_json = try buildMessages(allocator, persona.prompt, history.items, user_input);
    defer allocator.free(msg_json);
    const payload = try std.fmt.allocPrint(allocator,
        "{{\"model\":\"gpt-3.5-turbo\",\"messages\":{s}}}", .{msg_json});
    defer allocator.free(payload);

    var auth_header_buf: [256]u8 = undefined;
    const auth_header = try std.fmt.bufPrint(&auth_header_buf, "Authorization: Bearer {s}", .{api_key});

    var child = std.process.Child.init(.{
        .allocator = allocator,
        .argv = &.{
            "curl", "-sS",
            "-H", auth_header,
            "-H", "Content-Type: application/json",
            "-d", payload,
            "https://api.openai.com/v1/chat/completions",
        },
    });
    var result = try child.run();

    if (result.stderr.len != 0) {
        std.debug.print("curl error: {s}\n", .{result.stderr});
    }

    const root = try std.json.parseFromSlice(std.json.Value, allocator, result.stdout, .{});
    defer root.deinit();
    const choices = root.object.get("choices") orelse return error.InvalidResponse;
    const first = choices.array[0];
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

    const api_key = std.process.getenv("OPENAI_API_KEY") orelse {
        std.log.err("OPENAI_API_KEY environment variable not set", .{});
        return;
    };

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
        try stdout.print("{s}> {s}\n", .{persona.name, reply});
    }
}
