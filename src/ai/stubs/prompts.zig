const std = @import("std");

const Self = @This();

pub const PersonaType = enum {
    assistant,
    coder,
    writer,
    analyst,
    companion,
    docs,
    reviewer,
    minimal,
    abbey,
    ralph,
};

pub const Persona = struct {
    name: []const u8 = "",
    description: []const u8 = "",
    system_prompt: []const u8 = "",
    suggested_temperature: f32 = 0.7,
    include_examples: bool = false,
};

pub const PromptBuilder = struct {
    const Builder = @This();

    pub fn init(_: std.mem.Allocator, _: Self.PersonaType) Builder {
        return .{};
    }
    pub fn deinit(_: *Builder) void {}

    pub fn addUserMessage(_: *Builder, _: []const u8) error{AiDisabled}!void {
        return error.AiDisabled;
    }

    pub fn addSystemMessage(_: *Builder, _: []const u8) error{AiDisabled}!void {
        return error.AiDisabled;
    }

    pub fn addAssistantMessage(_: *Builder, _: []const u8) error{AiDisabled}!void {
        return error.AiDisabled;
    }

    pub fn build(_: *Builder, _: Self.PromptFormat) error{AiDisabled}![]u8 {
        return error.AiDisabled;
    }

    pub fn exportDebug(_: *Builder) error{AiDisabled}![]u8 {
        return error.AiDisabled;
    }

    pub fn addMessage(_: *Builder, _: Self.Role, _: []const u8) error{AiDisabled}!void {
        return error.AiDisabled;
    }
};

pub const PromptFormat = enum { plain, chat, text };
pub const Message = struct {};
pub const Role = enum { system, user, assistant, tool };

pub fn getPersona(_: Self.PersonaType) Self.Persona {
    return .{};
}

pub fn listPersonas() []const Self.PersonaType {
    return &[_]Self.PersonaType{
        .assistant,
        .coder,
        .writer,
        .analyst,
        .companion,
        .docs,
        .reviewer,
        .minimal,
        .abbey,
        .ralph,
    };
}

pub fn createBuilder(allocator: std.mem.Allocator) Self.PromptBuilder {
    return Self.PromptBuilder.init(allocator, .assistant);
}

pub fn createBuilderWithPersona(allocator: std.mem.Allocator, persona_type: Self.PersonaType) Self.PromptBuilder {
    return Self.PromptBuilder.init(allocator, persona_type);
}
