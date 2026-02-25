//! Agent Roles and Personas
//!
//! Provides identity, specialization, and behavioral constraints for agents
//! in multi-agent workflows. Each agent can be assigned a role that defines
//! its expertise domain, capabilities, and interaction style.
//!
//! Roles enable:
//! - **Intelligent routing**: Coordinator assigns tasks to the best-suited agent
//! - **Behavioral constraints**: Temperature, token limits, system prompts per role
//! - **Capability matching**: Workflow steps require capabilities that roles provide
//! - **Team composition**: Build balanced teams from complementary roles

const std = @import("std");

// ============================================================================
// Role Definitions
// ============================================================================

/// Domain of expertise for an agent role.
pub const Domain = enum {
    /// Code analysis, generation, and review.
    coding,
    /// Logical reasoning and problem decomposition.
    reasoning,
    /// Creative writing, brainstorming, ideation.
    creative,
    /// Data analysis, statistics, interpretation.
    analysis,
    /// Research, fact-finding, information synthesis.
    research,
    /// Planning, scheduling, project management.
    planning,
    /// Testing, validation, quality assurance.
    testing,
    /// Security analysis, vulnerability assessment.
    security,
    /// Documentation, technical writing.
    documentation,
    /// General-purpose, no specific domain.
    general,

    pub fn toString(self: Domain) []const u8 {
        return @tagName(self);
    }
};

/// Capability that a role provides. Used for matching roles to workflow steps.
pub const Capability = enum {
    /// Can generate code in various languages.
    code_generation,
    /// Can review code for bugs and style.
    code_review,
    /// Can refactor existing code.
    refactoring,
    /// Can write unit/integration tests.
    test_writing,
    /// Can analyze data and produce insights.
    data_analysis,
    /// Can summarize long text.
    summarization,
    /// Can translate between languages.
    translation,
    /// Can brainstorm and generate ideas.
    ideation,
    /// Can decompose complex problems.
    problem_decomposition,
    /// Can evaluate and critique other agents' work.
    critique,
    /// Can synthesize multiple sources into a coherent whole.
    synthesis,
    /// Can identify security vulnerabilities.
    security_audit,
    /// Can write documentation and guides.
    doc_writing,
    /// Can plan multi-step processes.
    planning,
    /// Can verify correctness of results.
    verification,

    pub fn toString(self: Capability) []const u8 {
        return @tagName(self);
    }
};

/// Interaction style that shapes how the agent communicates.
pub const InteractionStyle = enum {
    /// Terse, factual, minimal decoration.
    concise,
    /// Balanced detail with explanations.
    balanced,
    /// Thorough with examples and alternatives.
    detailed,
    /// Step-by-step with reasoning shown.
    methodical,
    /// Direct, opinionated, decisive.
    assertive,
    /// Exploratory, considers many angles.
    exploratory,
};

/// Behavioral constraints applied to an agent based on its role.
pub const BehaviorConstraints = struct {
    /// Minimum temperature for this role (0.0 = deterministic).
    min_temperature: f32 = 0.0,
    /// Maximum temperature for this role.
    max_temperature: f32 = 1.0,
    /// Suggested default temperature.
    default_temperature: f32 = 0.7,
    /// Maximum tokens this role should typically produce.
    max_output_tokens: u32 = 4096,
    /// Whether this role should show reasoning steps.
    show_reasoning: bool = false,
    /// Whether this role can delegate to other agents.
    can_delegate: bool = false,
    /// Whether this role can veto/reject other agents' output.
    can_veto: bool = false,
    /// Maximum retry attempts for this role.
    max_retries: u32 = 2,
};

// ============================================================================
// Persona
// ============================================================================

/// A persona defines the complete identity of an agent within a multi-agent team.
/// It combines a role, behavioral constraints, and a system prompt template.
pub const Persona = struct {
    /// Unique identifier for this persona.
    id: []const u8,
    /// Human-readable name.
    name: []const u8,
    /// Primary domain of expertise.
    domain: Domain,
    /// Capabilities this persona provides.
    capabilities: []const Capability,
    /// Interaction style.
    style: InteractionStyle,
    /// Behavioral constraints.
    constraints: BehaviorConstraints,
    /// System prompt that shapes the agent's behavior.
    system_prompt: []const u8,
    /// Optional description of what this persona does.
    description: []const u8 = "",

    /// Check if this persona has a specific capability.
    pub fn hasCapability(self: Persona, cap: Capability) bool {
        for (self.capabilities) |c| {
            if (c == cap) return true;
        }
        return false;
    }

    /// Count how many of the required capabilities this persona satisfies.
    pub fn matchScore(self: Persona, required: []const Capability) u32 {
        var score: u32 = 0;
        for (required) |req| {
            if (self.hasCapability(req)) score += 1;
        }
        return score;
    }

    /// Check if this persona satisfies all required capabilities.
    pub fn satisfiesAll(self: Persona, required: []const Capability) bool {
        return self.matchScore(required) == @as(u32, @intCast(required.len));
    }
};

// ============================================================================
// Preset Personas
// ============================================================================

/// Built-in persona presets for common agent roles.
pub const presets = struct {
    pub const code_reviewer = Persona{
        .id = "code-reviewer",
        .name = "Code Reviewer",
        .domain = .coding,
        .capabilities = &.{ .code_review, .critique, .security_audit },
        .style = .methodical,
        .constraints = .{
            .default_temperature = 0.3,
            .max_temperature = 0.5,
            .show_reasoning = true,
            .can_veto = true,
        },
        .system_prompt = "You are a thorough code reviewer. Analyze code for bugs, security issues, " ++
            "performance problems, and style violations. Be specific about line numbers and provide fixes.",
        .description = "Reviews code for correctness, security, and quality",
    };

    pub const architect = Persona{
        .id = "architect",
        .name = "Software Architect",
        .domain = .planning,
        .capabilities = &.{ .problem_decomposition, .planning, .synthesis },
        .style = .detailed,
        .constraints = .{
            .default_temperature = 0.5,
            .max_output_tokens = 8192,
            .can_delegate = true,
            .show_reasoning = true,
        },
        .system_prompt = "You are a software architect. Design systems with clear boundaries, " ++
            "identify components and their interactions, and produce implementation plans.",
        .description = "Designs system architecture and creates implementation plans",
    };

    pub const implementer = Persona{
        .id = "implementer",
        .name = "Code Implementer",
        .domain = .coding,
        .capabilities = &.{ .code_generation, .refactoring, .test_writing },
        .style = .concise,
        .constraints = .{
            .default_temperature = 0.2,
            .max_temperature = 0.4,
            .max_output_tokens = 8192,
        },
        .system_prompt = "You are a focused code implementer. Write clean, correct, well-tested code. " ++
            "Follow existing patterns and conventions. Minimize unnecessary changes.",
        .description = "Implements features and writes code",
    };

    pub const researcher = Persona{
        .id = "researcher",
        .name = "Research Analyst",
        .domain = .research,
        .capabilities = &.{ .data_analysis, .summarization, .synthesis },
        .style = .exploratory,
        .constraints = .{
            .default_temperature = 0.6,
            .max_output_tokens = 6144,
            .show_reasoning = true,
        },
        .system_prompt = "You are a research analyst. Gather information, analyze sources, " ++
            "identify patterns, and synthesize findings into clear reports.",
        .description = "Researches topics and synthesizes findings",
    };

    pub const critic = Persona{
        .id = "critic",
        .name = "Quality Critic",
        .domain = .testing,
        .capabilities = &.{ .critique, .verification, .test_writing },
        .style = .assertive,
        .constraints = .{
            .default_temperature = 0.3,
            .can_veto = true,
            .show_reasoning = true,
        },
        .system_prompt = "You are a quality critic. Evaluate outputs for correctness, completeness, " ++
            "and quality. Identify gaps, errors, and improvements. Be direct and specific.",
        .description = "Evaluates and critiques outputs for quality",
    };

    pub const writer = Persona{
        .id = "writer",
        .name = "Technical Writer",
        .domain = .documentation,
        .capabilities = &.{ .doc_writing, .summarization, .synthesis },
        .style = .balanced,
        .constraints = .{
            .default_temperature = 0.5,
            .max_output_tokens = 8192,
        },
        .system_prompt = "You are a technical writer. Produce clear, well-structured documentation " ++
            "with examples. Adapt tone to the audience. Use consistent formatting.",
        .description = "Writes documentation and technical content",
    };

    /// All preset personas for iteration.
    pub const all = [_]Persona{
        code_reviewer,
        architect,
        implementer,
        researcher,
        critic,
        writer,
    };
};

// ============================================================================
// PersonaRegistry
// ============================================================================

/// Registry that maps agent IDs to personas and enables capability-based lookup.
pub const PersonaRegistry = struct {
    allocator: std.mem.Allocator,
    /// Map from persona ID to persona.
    personas: std.StringHashMapUnmanaged(Persona),

    pub fn init(allocator: std.mem.Allocator) PersonaRegistry {
        return .{
            .allocator = allocator,
            .personas = .{},
        };
    }

    pub fn deinit(self: *PersonaRegistry) void {
        self.personas.deinit(self.allocator);
    }

    /// Register a persona. Overwrites if ID already exists.
    pub fn register(self: *PersonaRegistry, persona: Persona) !void {
        try self.personas.put(self.allocator, persona.id, persona);
    }

    /// Load all preset personas into the registry.
    pub fn loadPresets(self: *PersonaRegistry) !void {
        for (presets.all) |p| {
            try self.register(p);
        }
    }

    /// Look up a persona by ID.
    pub fn get(self: *const PersonaRegistry, id: []const u8) ?Persona {
        return self.personas.get(id);
    }

    /// Find the best persona for a set of required capabilities.
    /// Returns the persona with the highest match score, or null if none match.
    pub fn findBestMatch(self: *const PersonaRegistry, required: []const Capability) ?Persona {
        var best: ?Persona = null;
        var best_score: u32 = 0;

        var iter = self.personas.iterator();
        while (iter.next()) |entry| {
            const score = entry.value_ptr.matchScore(required);
            if (score > best_score) {
                best_score = score;
                best = entry.value_ptr.*;
            }
        }

        return best;
    }

    /// Find all personas that have a specific capability.
    pub fn findByCapability(
        self: *const PersonaRegistry,
        allocator: std.mem.Allocator,
        cap: Capability,
    ) ![]const Persona {
        var results: std.ArrayListUnmanaged(Persona) = .empty;
        errdefer results.deinit(allocator);

        var iter = self.personas.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.hasCapability(cap)) {
                try results.append(allocator, entry.value_ptr.*);
            }
        }

        return results.toOwnedSlice(allocator);
    }

    /// Find all personas in a specific domain.
    pub fn findByDomain(
        self: *const PersonaRegistry,
        allocator: std.mem.Allocator,
        domain: Domain,
    ) ![]const Persona {
        var results: std.ArrayListUnmanaged(Persona) = .empty;
        errdefer results.deinit(allocator);

        var iter = self.personas.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.domain == domain) {
                try results.append(allocator, entry.value_ptr.*);
            }
        }

        return results.toOwnedSlice(allocator);
    }

    /// Number of registered personas.
    pub fn count(self: *const PersonaRegistry) usize {
        return self.personas.count();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "persona capability matching" {
    const p = presets.code_reviewer;
    try std.testing.expect(p.hasCapability(.code_review));
    try std.testing.expect(p.hasCapability(.critique));
    try std.testing.expect(!p.hasCapability(.code_generation));
}

test "persona match score" {
    const p = presets.implementer;
    const required = [_]Capability{ .code_generation, .refactoring, .test_writing };
    try std.testing.expectEqual(@as(u32, 3), p.matchScore(&required));
    try std.testing.expect(p.satisfiesAll(&required));

    const partial = [_]Capability{ .code_generation, .security_audit };
    try std.testing.expectEqual(@as(u32, 1), p.matchScore(&partial));
    try std.testing.expect(!p.satisfiesAll(&partial));
}

test "persona registry basics" {
    var reg = PersonaRegistry.init(std.testing.allocator);
    defer reg.deinit();

    try reg.loadPresets();
    try std.testing.expectEqual(@as(usize, 6), reg.count());

    const reviewer = reg.get("code-reviewer");
    try std.testing.expect(reviewer != null);
    try std.testing.expectEqualStrings("Code Reviewer", reviewer.?.name);
}

test "persona registry find best match" {
    var reg = PersonaRegistry.init(std.testing.allocator);
    defer reg.deinit();

    try reg.loadPresets();

    const required = [_]Capability{ .code_review, .critique };
    const best = reg.findBestMatch(&required);
    try std.testing.expect(best != null);
    try std.testing.expectEqualStrings("code-reviewer", best.?.id);
}

test "persona registry find by capability" {
    var reg = PersonaRegistry.init(std.testing.allocator);
    defer reg.deinit();

    try reg.loadPresets();

    const results = try reg.findByCapability(std.testing.allocator, .synthesis);
    defer std.testing.allocator.free(results);

    // architect, researcher, and writer all have synthesis
    try std.testing.expectEqual(@as(usize, 3), results.len);
}

test "persona registry find by domain" {
    var reg = PersonaRegistry.init(std.testing.allocator);
    defer reg.deinit();

    try reg.loadPresets();

    const coders = try reg.findByDomain(std.testing.allocator, .coding);
    defer std.testing.allocator.free(coders);

    // code_reviewer and implementer are in the coding domain
    try std.testing.expectEqual(@as(usize, 2), coders.len);
}

test "preset persona constraints" {
    const reviewer = presets.code_reviewer;
    try std.testing.expect(reviewer.constraints.can_veto);
    try std.testing.expect(!reviewer.constraints.can_delegate);
    try std.testing.expect(reviewer.constraints.show_reasoning);

    const architect = presets.architect;
    try std.testing.expect(architect.constraints.can_delegate);
    try std.testing.expect(!architect.constraints.can_veto);
}

test {
    std.testing.refAllDecls(@This());
}
