//! End-to-End Multi-Persona Integration Tests
//!
//! Comprehensive tests for the multi-persona AI assistant system covering:
//! - Persona definitions and retrieval
//! - System prompt generation
//! - Persona configuration (temperature, examples)
//! - All persona types (assistant, coder, writer, etc.)
//! - Edge cases: invalid personas, empty prompts
//! - Integration with AI module
//!
//! These tests verify the persona system works correctly without
//! requiring external AI services.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

// Use the prompts module which has the simple getPersona/listPersonas API
const prompts = abi.ai.prompts;

// Skip all tests if AI feature is disabled
fn skipIfAiDisabled() !void {
    if (!build_options.enable_ai) return error.SkipZigTest;
}

// ============================================================================
// Persona Definition Tests
// ============================================================================

// Test retrieving the assistant persona.
// Verifies default helpful assistant configuration.
test "persona: assistant definition" {
    try skipIfAiDisabled();

    const persona = prompts.getPersona(.assistant);

    try std.testing.expectEqualStrings("assistant", persona.name);
    try std.testing.expect(persona.description.len > 0);
    try std.testing.expect(persona.system_prompt.len > 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), persona.suggested_temperature, 0.001);
    try std.testing.expect(!persona.include_examples);

    // Should contain key phrases
    try std.testing.expect(std.mem.indexOf(u8, persona.system_prompt, "helpful") != null);
}

// Test retrieving the coder persona.
// Verifies programming-focused configuration.
test "persona: coder definition" {
    try skipIfAiDisabled();

    const persona = prompts.getPersona(.coder);

    try std.testing.expectEqualStrings("coder", persona.name);
    try std.testing.expect(persona.description.len > 0);
    try std.testing.expect(persona.system_prompt.len > 0);

    // Coder should have lower temperature for more deterministic output
    try std.testing.expect(persona.suggested_temperature < 0.5);

    // Should include coding-related content
    try std.testing.expect(persona.include_examples);
    try std.testing.expect(std.mem.indexOf(u8, persona.system_prompt, "code") != null);
}

// Test retrieving the writer persona.
// Verifies creative writing configuration.
test "persona: writer definition" {
    try skipIfAiDisabled();

    const persona = prompts.getPersona(.writer);

    try std.testing.expectEqualStrings("writer", persona.name);
    try std.testing.expect(persona.description.len > 0);
    try std.testing.expect(persona.system_prompt.len > 0);

    // Writer should have higher temperature for creativity
    try std.testing.expect(persona.suggested_temperature > 0.7);

    // Should contain writing-related content
    try std.testing.expect(std.mem.indexOf(u8, persona.system_prompt, "creative") != null or
        std.mem.indexOf(u8, persona.system_prompt, "writing") != null);
}

// Test retrieving the analyst persona.
// Verifies data analysis configuration.
test "persona: analyst definition" {
    try skipIfAiDisabled();

    const persona = prompts.getPersona(.analyst);

    try std.testing.expectEqualStrings("analyst", persona.name);
    try std.testing.expect(persona.description.len > 0);
    try std.testing.expect(persona.system_prompt.len > 0);

    // Analyst should have moderate temperature
    try std.testing.expect(persona.suggested_temperature >= 0.3);
    try std.testing.expect(persona.suggested_temperature <= 0.6);

    // Should contain analysis-related content
    try std.testing.expect(std.mem.indexOf(u8, persona.system_prompt, "analysis") != null or
        std.mem.indexOf(u8, persona.system_prompt, "data") != null);
}

// Test retrieving the companion persona.
// Verifies friendly conversational configuration.
test "persona: companion definition" {
    try skipIfAiDisabled();

    const persona = prompts.getPersona(.companion);

    try std.testing.expectEqualStrings("companion", persona.name);
    try std.testing.expect(persona.description.len > 0);
    try std.testing.expect(persona.system_prompt.len > 0);

    // Companion should have higher temperature for natural conversation
    try std.testing.expect(persona.suggested_temperature >= 0.7);

    // Should contain friendly-related content
    try std.testing.expect(std.mem.indexOf(u8, persona.system_prompt, "friendly") != null or
        std.mem.indexOf(u8, persona.system_prompt, "conversational") != null);
}

// Test retrieving the docs persona.
// Verifies documentation specialist configuration.
test "persona: docs definition" {
    try skipIfAiDisabled();

    const persona = prompts.getPersona(.docs);

    try std.testing.expectEqualStrings("docs", persona.name);
    try std.testing.expect(persona.description.len > 0);
    try std.testing.expect(persona.system_prompt.len > 0);

    // Docs should have lower temperature for consistency
    try std.testing.expect(persona.suggested_temperature < 0.5);

    // Should include examples for documentation
    try std.testing.expect(persona.include_examples);

    // Should contain documentation-related content
    try std.testing.expect(std.mem.indexOf(u8, persona.system_prompt, "documentation") != null or
        std.mem.indexOf(u8, persona.system_prompt, "technical") != null);
}

// Test retrieving the reviewer persona.
// Verifies code review specialist configuration.
test "persona: reviewer definition" {
    try skipIfAiDisabled();

    const persona = prompts.getPersona(.reviewer);

    try std.testing.expectEqualStrings("reviewer", persona.name);
    try std.testing.expect(persona.description.len > 0);
    try std.testing.expect(persona.system_prompt.len > 0);

    // Reviewer should have low temperature for precise feedback
    try std.testing.expect(persona.suggested_temperature <= 0.3);

    // Should contain review-related content
    try std.testing.expect(std.mem.indexOf(u8, persona.system_prompt, "review") != null or
        std.mem.indexOf(u8, persona.system_prompt, "code") != null);
}

// Test retrieving the minimal persona.
// Verifies direct response configuration.
test "persona: minimal definition" {
    try skipIfAiDisabled();

    const persona = prompts.getPersona(.minimal);

    try std.testing.expectEqualStrings("minimal", persona.name);
    try std.testing.expect(persona.description.len > 0);
    try std.testing.expect(persona.system_prompt.len > 0);

    // Minimal prompt should be short
    try std.testing.expect(persona.system_prompt.len < 200);

    // Should not include examples (minimal mode)
    try std.testing.expect(!persona.include_examples);
}

// ============================================================================
// Abbey Persona Tests
// ============================================================================

// Test retrieving the Abbey persona.
// Verifies emotionally intelligent polymath configuration.
test "persona: abbey definition" {
    try skipIfAiDisabled();

    const persona = prompts.getPersona(.abbey);

    try std.testing.expectEqualStrings("abbey", persona.name);
    try std.testing.expect(persona.description.len > 0);
    try std.testing.expect(persona.system_prompt.len > 0);

    // Abbey should have moderate temperature balancing creativity and precision
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), persona.suggested_temperature, 0.1);

    // Abbey should include examples
    try std.testing.expect(persona.include_examples);

    // Should contain Abbey-specific content
    try std.testing.expect(std.mem.indexOf(u8, persona.system_prompt, "Abbey") != null);

    // Should have emotional intelligence content
    try std.testing.expect(std.mem.indexOf(u8, persona.system_prompt, "emotional") != null or
        std.mem.indexOf(u8, persona.system_prompt, "Emotional") != null);
}

// ============================================================================
// Ralph Persona Tests
// ============================================================================

// Test retrieving the Ralph persona.
// Verifies iterative worker configuration.
test "persona: ralph definition" {
    try skipIfAiDisabled();

    const persona = prompts.getPersona(.ralph);

    try std.testing.expectEqualStrings("ralph", persona.name);
    try std.testing.expect(persona.description.len > 0);
    try std.testing.expect(persona.system_prompt.len > 0);

    // Ralph should have low temperature for precise iteration
    try std.testing.expect(persona.suggested_temperature <= 0.3);

    // Should contain Ralph-specific content
    try std.testing.expect(std.mem.indexOf(u8, persona.system_prompt, "Ralph") != null);

    // Should have iteration-related content
    try std.testing.expect(std.mem.indexOf(u8, persona.system_prompt, "ITERATE") != null or
        std.mem.indexOf(u8, persona.system_prompt, "iterate") != null);
}

// ============================================================================
// Aviva Persona Tests
// ============================================================================

// Test retrieving the Aviva persona.
// Verifies direct expert configuration.
test "persona: aviva definition" {
    try skipIfAiDisabled();

    const persona = prompts.getPersona(.aviva);

    try std.testing.expectEqualStrings("aviva", persona.name);
    try std.testing.expect(persona.description.len > 0);
    try std.testing.expect(persona.system_prompt.len > 0);

    // Aviva should have low temperature for precise, factual output
    try std.testing.expect(persona.suggested_temperature <= 0.3);

    // Should contain Aviva-specific content
    try std.testing.expect(std.mem.indexOf(u8, persona.system_prompt, "Aviva") != null);

    // Should have direct/concise content
    try std.testing.expect(std.mem.indexOf(u8, persona.system_prompt, "direct") != null or
        std.mem.indexOf(u8, persona.system_prompt, "concise") != null);
}

// ============================================================================
// Abi Persona Tests
// ============================================================================

// Test retrieving the Abi persona.
// Verifies adaptive moderator configuration.
test "persona: abi definition" {
    try skipIfAiDisabled();

    const persona = prompts.getPersona(.abi);

    try std.testing.expectEqualStrings("abi", persona.name);
    try std.testing.expect(persona.description.len > 0);
    try std.testing.expect(persona.system_prompt.len > 0);

    // Abi should have moderate temperature for adaptive routing
    try std.testing.expect(persona.suggested_temperature >= 0.3);
    try std.testing.expect(persona.suggested_temperature <= 0.7);

    // Should contain Abi-specific content
    try std.testing.expect(std.mem.indexOf(u8, persona.system_prompt, "Abi") != null);

    // Should have routing/moderation content
    try std.testing.expect(std.mem.indexOf(u8, persona.system_prompt, "route") != null or
        std.mem.indexOf(u8, persona.system_prompt, "routing") != null or
        std.mem.indexOf(u8, persona.system_prompt, "moderator") != null);
}

// ============================================================================
// Persona Listing Tests
// ============================================================================

// Test listing all available personas.
// Verifies complete persona enumeration.
test "persona: list all" {
    try skipIfAiDisabled();

    const all_personas = prompts.listPersonas();

    // Should have at least 8 personas
    try std.testing.expect(all_personas.len >= 8);

    // Verify expected personas are present
    var found_assistant = false;
    var found_coder = false;
    var found_abbey = false;
    var found_ralph = false;

    for (all_personas) |persona_type| {
        if (persona_type == .assistant) found_assistant = true;
        if (persona_type == .coder) found_coder = true;
        if (persona_type == .abbey) found_abbey = true;
        if (persona_type == .ralph) found_ralph = true;
    }

    try std.testing.expect(found_assistant);
    try std.testing.expect(found_coder);
    try std.testing.expect(found_abbey);
    try std.testing.expect(found_ralph);
}

// ============================================================================
// Temperature Configuration Tests
// ============================================================================

// Test temperature values are within valid range.
// All temperatures should be between 0.0 and 2.0.
test "persona: temperature ranges" {
    try skipIfAiDisabled();

    const all_types = prompts.listPersonas();

    for (all_types) |persona_type| {
        const persona = prompts.getPersona(persona_type);

        // Temperature should be in valid range
        try std.testing.expect(persona.suggested_temperature >= 0.0);
        try std.testing.expect(persona.suggested_temperature <= 2.0);
    }
}

// Test temperature ordering by persona purpose.
// Creative personas should have higher temperatures than analytical.
test "persona: temperature ordering" {
    try skipIfAiDisabled();

    const writer = prompts.getPersona(.writer);
    const coder = prompts.getPersona(.coder);
    const reviewer = prompts.getPersona(.reviewer);

    // Writer (creative) should have higher temp than coder (precise)
    try std.testing.expect(writer.suggested_temperature > coder.suggested_temperature);

    // Coder should have higher or equal temp to reviewer (very precise)
    try std.testing.expect(coder.suggested_temperature >= reviewer.suggested_temperature);
}

// ============================================================================
// System Prompt Quality Tests
// ============================================================================

// Test system prompts have sufficient length.
// Prompts should have meaningful content, not just placeholders.
test "persona: prompt length requirements" {
    try skipIfAiDisabled();

    const all_types = prompts.listPersonas();

    for (all_types) |persona_type| {
        const persona = prompts.getPersona(persona_type);

        // Minimal persona can be short, others should be substantial
        if (persona_type == .minimal) {
            try std.testing.expect(persona.system_prompt.len >= 10);
        } else {
            try std.testing.expect(persona.system_prompt.len >= 50);
        }
    }
}

// Test system prompts contain no control characters.
// Prompts should be clean text suitable for display.
test "persona: prompt character validity" {
    try skipIfAiDisabled();

    const all_types = prompts.listPersonas();

    for (all_types) |persona_type| {
        const persona = prompts.getPersona(persona_type);

        for (persona.system_prompt) |c| {
            // Should not contain null bytes or bell characters
            try std.testing.expect(c != 0);
            try std.testing.expect(c != 7); // Bell
            try std.testing.expect(c != 8); // Backspace

            // Allowed: printable ASCII, newline, tab, carriage return
            const is_printable = (c >= 32 and c <= 126);
            const is_whitespace = (c == '\n' or c == '\r' or c == '\t');
            const is_extended = (c >= 128); // UTF-8 continuation
            const is_backslash = (c == '\\');

            try std.testing.expect(is_printable or is_whitespace or is_extended or is_backslash);
        }
    }
}

// ============================================================================
// Persona Type Enumeration Tests
// ============================================================================

// Test PersonaType enum values.
// Verifies enum is properly defined and accessible.
test "persona type: enum values" {
    try skipIfAiDisabled();

    // All persona types should be valid enum values
    const types = [_]prompts.PersonaType{
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
        .aviva,
        .abi,
    };

    // Each type should be distinct
    for (types, 0..) |t1, i| {
        for (types[i + 1 ..]) |t2| {
            try std.testing.expect(t1 != t2);
        }
    }
}

// Test PersonaType can be converted to/from integers.
// Useful for serialization and indexing.
test "persona type: integer conversion" {
    try skipIfAiDisabled();

    // Convert to int and back
    const original = prompts.PersonaType.coder;
    const as_int = @intFromEnum(original);
    const back = @as(prompts.PersonaType, @enumFromInt(as_int));

    try std.testing.expectEqual(original, back);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

// Test that persona retrieval is consistent.
// Same persona type should always return same definition.
test "edge case: retrieval consistency" {
    try skipIfAiDisabled();

    const persona1 = prompts.getPersona(.abbey);
    const persona2 = prompts.getPersona(.abbey);

    // Should return identical data
    try std.testing.expectEqualStrings(persona1.name, persona2.name);
    try std.testing.expectEqualStrings(persona1.description, persona2.description);
    try std.testing.expectEqualStrings(persona1.system_prompt, persona2.system_prompt);
    try std.testing.expectEqual(persona1.suggested_temperature, persona2.suggested_temperature);
    try std.testing.expectEqual(persona1.include_examples, persona2.include_examples);
}

// Test rapid persona switching.
// Should handle fast switches without issues.
test "edge case: rapid persona switching" {
    try skipIfAiDisabled();

    const types = prompts.listPersonas();

    // Switch between personas rapidly
    for (0..100) |_| {
        for (types) |persona_type| {
            const persona = prompts.getPersona(persona_type);
            _ = persona.name;
            _ = persona.system_prompt;
        }
    }
}

// ============================================================================
// Prompt Builder Integration Tests
// ============================================================================

// Test creating a prompt builder with a persona.
// Verifies integration with the builder pattern.
test "prompt builder: with persona" {
    try skipIfAiDisabled();

    const allocator = std.testing.allocator;

    var builder = prompts.PromptBuilder.init(allocator, .coder);
    defer builder.deinit();

    try builder.addUserMessage("Write a hello world program");
    const prompt = try builder.build(.text);
    defer allocator.free(prompt);

    // Should contain the coder persona's system prompt
    try std.testing.expect(prompt.len > 0);
}

// Test creating a prompt builder with custom persona.
// Verifies custom persona can be provided.
test "prompt builder: custom persona" {
    try skipIfAiDisabled();

    const allocator = std.testing.allocator;

    const custom = prompts.Persona{
        .name = "custom",
        .description = "Custom test persona",
        .system_prompt = "You are a custom test assistant.",
        .suggested_temperature = 0.5,
        .include_examples = false,
    };

    var builder = prompts.createBuilderWithCustomPersona(allocator, custom);
    defer builder.deinit();

    try builder.addUserMessage("Hello");
    const prompt = try builder.build(.text);
    defer allocator.free(prompt);

    // Should contain the custom system prompt
    try std.testing.expect(std.mem.indexOf(u8, prompt, "custom test assistant") != null);
}
