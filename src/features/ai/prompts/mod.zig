//! Centralized Prompt Templates and Builder
//!
//! Provides consistent, well-documented prompt templates for all AI/LLM features.
//! All prompts are exportable for inspection and debugging via --show-prompt flag.
//!
//! ## Usage
//! ```zig
//! const prompts = @import("prompts");
//!
//! // Create a prompt builder with persona
//! var builder = prompts.PromptBuilder.init(allocator, .assistant);
//! defer builder.deinit();
//!
//! // Build a chat prompt
//! try builder.addUserMessage("Hello!");
//! const prompt = try builder.build();
//!
//! // Export for debugging
//! if (show_prompt) {
//!     std.debug.print("{s}\n", .{builder.export()});
//! }
//! ```

const std = @import("std");
pub const personas = @import("personas.zig");
pub const builder = @import("builder.zig");

// Re-export main types
pub const Persona = personas.Persona;
pub const PersonaType = personas.PersonaType;
pub const PromptBuilder = builder.PromptBuilder;
pub const Message = builder.Message;
pub const Role = builder.Role;
pub const PromptFormat = builder.PromptFormat;

/// Get a persona by type
pub fn getPersona(persona_type: PersonaType) Persona {
    return personas.getPersona(persona_type);
}

/// List all available personas
pub fn listPersonas() []const PersonaType {
    return personas.listPersonas();
}

/// Create a prompt builder with default assistant persona
pub fn createBuilder(allocator: std.mem.Allocator) PromptBuilder {
    return PromptBuilder.init(allocator, .assistant);
}

/// Create a prompt builder with a specific persona
pub fn createBuilderWithPersona(allocator: std.mem.Allocator, persona_type: PersonaType) PromptBuilder {
    return PromptBuilder.init(allocator, persona_type);
}

/// Create a prompt builder with a custom persona
pub fn createBuilderWithCustomPersona(allocator: std.mem.Allocator, persona: Persona) PromptBuilder {
    return PromptBuilder.initCustom(allocator, persona);
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
