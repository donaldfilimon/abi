//! Centralized Prompt Templates and Builder
//!
//! Provides consistent, well-documented prompt templates for all AI/LLM features.
//! All prompts are exportable for inspection and debugging via --show-prompt flag.

const std = @import("std");
pub const types = @import("types");
pub const builder = @import("builder.zig");
pub const ralph = @import("ralph.zig");
pub const personas = @import("personas.zig");

// Re-export main types
pub const PersonaType = types.PersonaType;
pub const Persona = personas.Persona;
pub const PromptBuilder = builder.PromptBuilder;
pub const Message = builder.Message;
pub const Role = builder.Role;
pub const PromptFormat = builder.PromptFormat;

/// Get a persona definition by type.
/// Maps from the canonical PersonaType to the prompts-internal PersonaType
/// and returns the corresponding prompt definition.
pub fn getPersona(persona_type: PersonaType) Persona {
    const prompts_type = std.meta.stringToEnum(
        personas.PersonaType,
        @tagName(persona_type),
    ) orelse return personas.getPersona(.assistant);
    return personas.getPersona(prompts_type);
}

/// List all available personas
pub fn listPersonas() []const PersonaType {
    return &[_]PersonaType{ .assistant, .coder, .writer, .analyst };
}

/// Create a prompt builder with default assistant persona
pub fn createBuilder(allocator: std.mem.Allocator) PromptBuilder {
    return PromptBuilder.init(allocator, .assistant);
}

/// Create a prompt builder with a specific persona
pub fn createBuilderWithPersona(allocator: std.mem.Allocator, persona_type: PersonaType) PromptBuilder {
    return PromptBuilder.init(allocator, persona_type);
}

test "prompt module basics" {
    const allocator = std.testing.allocator;

    var b = createBuilder(allocator);
    defer b.deinit();

    try b.addUserMessage("Hello");
    const prompt = try b.build(.text);
    defer allocator.free(prompt);

    try std.testing.expect(prompt.len > 0);
}

test {
    std.testing.refAllDecls(@This());
}
