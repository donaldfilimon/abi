//! Prompts stub — active when AI feature is disabled.

const std = @import("std");

// Sub-module stubs
pub const personas = struct {
    pub const Persona = @This().OuterPersona;
    pub const PersonaType = @This().OuterPersonaType;
    pub const getPersona = @This().outerGetPersona;
    pub const listPersonas = @This().outerListPersonas;

    const OuterPersona = struct {
        name: []const u8 = "disabled",
        description: []const u8 = "",
        system_prompt: []const u8 = "",
        suggested_temperature: f32 = 0.7,
        include_examples: bool = false,
    };

    const OuterPersonaType = enum {
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
        aviva,
        abi,
        ava,
    };

    fn outerGetPersona(_: OuterPersonaType) OuterPersona {
        return .{};
    }

    fn outerListPersonas() []const OuterPersonaType {
        return &.{};
    }
};

pub const builder = struct {
    pub const PromptBuilder = @This().OuterPromptBuilder;
    pub const Message = @This().OuterMessage;
    pub const Role = @This().OuterRole;
    pub const PromptFormat = @This().OuterPromptFormat;

    const OuterRole = enum {
        system,
        user,
        assistant,
        tool,
    };

    const OuterMessage = struct {
        role: OuterRole = .user,
        content: []const u8 = "",
    };

    const OuterPromptFormat = enum {
        text,
        chatml,
        alpaca,
        llama,
        vicuna,
    };

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

/// Get a persona by type.
pub fn getPersona(persona_type: PersonaType) Persona {
    _ = persona_type;
    return .{};
}

/// List all available personas.
pub fn listPersonas() []const PersonaType {
    return &.{};
}

/// Create a prompt builder with default assistant persona.
pub fn createBuilder(allocator: std.mem.Allocator) PromptBuilder {
    return PromptBuilder.init(allocator, .assistant);
}

/// Create a prompt builder with a specific persona.
pub fn createBuilderWithPersona(allocator: std.mem.Allocator, persona_type: PersonaType) PromptBuilder {
    return PromptBuilder.init(allocator, persona_type);
}

/// Create a prompt builder with a custom persona.
pub fn createBuilderWithCustomPersona(allocator: std.mem.Allocator, persona: Persona) PromptBuilder {
    return PromptBuilder.initCustom(allocator, persona);
}
