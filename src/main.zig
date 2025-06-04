const std = @import("std");

pub const PersonaType = enum {
    EmpatheticAnalyst,
    DirectExpert,
    AdaptiveModerator,
};

pub const Config = struct {
    persona_weight: f32 = 1.0,
};

pub fn personaEmbedding(input: []const u8, p: PersonaType, cfg: Config, alloc: std.mem.Allocator) []u8 {
    _ = cfg; // unused for now
    var list = std.ArrayList(u8).init(alloc);
    list.appendSlice(input) catch {};
    const tag = switch (p) {
        .EmpatheticAnalyst => "[EA]",
        .DirectExpert => "[DE]",
        .AdaptiveModerator => "[AM]",
    };
    list.appendSlice(tag) catch {};
    return list.toOwnedSlice() catch unreachable;
}

fn evaluateRisk(response: []const u8) i32 {
    if (std.mem.indexOf(u8, response, "banned") != null) {
        return 100;
    }
    return 0;
}

pub fn ethicalFilter(response: []const u8) []const u8 {
    if (evaluateRisk(response) > 0) {
        return "Content removed due to policy.";
    }
    return response;
}

pub fn router(query: []const u8) PersonaType {
    if (std.mem.indexOf(u8, query, "help") != null)
        return PersonaType.EmpatheticAnalyst;
    if (std.mem.indexOf(u8, query, "explain") != null)
        return PersonaType.DirectExpert;
    return PersonaType.AdaptiveModerator;
}

fn respond(p: PersonaType, _: []const u8, alloc: std.mem.Allocator) []u8 {
    const base = switch (p) {
        .EmpatheticAnalyst => "I understand how you feel. Let's see how I can help.",
        .DirectExpert => "Here is the direct answer to your question.",
        .AdaptiveModerator => "I'll route your request appropriately.",
    };
    return personaEmbedding(base, p, .{}, alloc);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();
    const args = std.os.argv;
    if (args.len < 2) {
        std.debug.print("Usage: {s} <query>\n", .{args[0]});
        return;
    }
    const query = std.mem.span(args[1]);
    const persona = router(query);
    const raw_response = respond(persona, query, alloc);
    defer alloc.free(raw_response);
    const safe_response = ethicalFilter(raw_response);
    std.debug.print("{s}\n", .{safe_response});
}
