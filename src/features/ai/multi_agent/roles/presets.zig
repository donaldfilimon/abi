//! Preset agent profiles.

const types = @import("types.zig");
const Profile = types.Profile;

pub const presets = struct {
    pub const code_reviewer = Profile{
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

    pub const architect = Profile{
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

    pub const implementer = Profile{
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

    pub const researcher = Profile{
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

    pub const critic = Profile{
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

    pub const writer = Profile{
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

    pub const all = [_]Profile{
        code_reviewer,
        architect,
        implementer,
        researcher,
        critic,
        writer,
    };
};

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
