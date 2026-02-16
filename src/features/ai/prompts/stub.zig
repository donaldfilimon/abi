//! Prompts stub — active when AI feature is disabled.

const std = @import("std");

// Sub-module stubs
pub const personas = struct {
    pub const Persona = OuterPersona;
    pub const PersonaType = OuterPersonaType;
    pub const getPersona = outerGetPersona;
    pub const listPersonas = outerListPersonas;

    const OuterPersona = struct {
        name: []const u8 = "disabled",
        description: []const u8 = "",
        system_prompt: []const u8 = "",
        suggested_temperature: f32 = 0.7,
        include_examples: bool = false,
    };

    const OuterPersonaType = enum { assistant, coder, writer, analyst, companion, docs, reviewer, minimal, abbey, ralph, aviva, abi, ava };

    fn outerGetPersona(_: OuterPersonaType) OuterPersona {
        return .{};
    }
    fn outerListPersonas() []const OuterPersonaType {
        return &.{};
    }
};

pub const builder = struct {
    pub const PromptBuilder = OuterPromptBuilder;
    pub const Message = OuterMessage;
    pub const Role = OuterRole;
    pub const PromptFormat = OuterPromptFormat;

    const OuterRole = enum { system, user, assistant, tool };
    const OuterMessage = struct { role: OuterRole = .user, content: []const u8 = "" };
    const OuterPromptFormat = enum { text, chatml, alpaca, llama, vicuna };

    const OuterPromptBuilder = struct {
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, _: personas.OuterPersonaType) OuterPromptBuilder {
            return .{ .allocator = allocator };
        }
        pub fn initCustom(allocator: std.mem.Allocator, _: personas.OuterPersona) OuterPromptBuilder {
            return .{ .allocator = allocator };
        }
        pub fn deinit(_: *OuterPromptBuilder) void {}
        pub fn addUserMessage(_: *OuterPromptBuilder, _: []const u8) !void {
            return error.FeatureDisabled;
        }
        pub fn addMessage(_: *OuterPromptBuilder, _: OuterRole, _: []const u8) !void {
            return error.FeatureDisabled;
        }
        pub fn build(_: *OuterPromptBuilder, _: OuterPromptFormat) ![]u8 {
            return error.FeatureDisabled;
        }
        pub fn exportDebug(_: *OuterPromptBuilder) ![]u8 {
            return error.FeatureDisabled;
        }
    };
};

pub const ralph = struct {};

// Re-export main types
pub const Persona = personas.OuterPersona;
pub const PersonaType = personas.OuterPersonaType;
pub const PromptBuilder = builder.OuterPromptBuilder;
pub const Message = builder.OuterMessage;
pub const Role = builder.OuterRole;
pub const PromptFormat = builder.OuterPromptFormat;

pub fn getPersona(_: PersonaType) Persona {
    return .{};
}
pub fn listPersonas() []const PersonaType {
    return &.{};
}
pub fn createBuilder(allocator: std.mem.Allocator) PromptBuilder {
    return PromptBuilder.init(allocator, .assistant);
}
pub fn createBuilderWithPersona(allocator: std.mem.Allocator, persona_type: PersonaType) PromptBuilder {
    return PromptBuilder.init(allocator, persona_type);
}
pub fn createBuilderWithCustomPersona(allocator: std.mem.Allocator, persona: Persona) PromptBuilder {
    return PromptBuilder.initCustom(allocator, persona);
}
