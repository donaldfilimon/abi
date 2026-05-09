//! Preset workflow templates.

const definition = @import("definition.zig");
const roles = @import("../roles.zig");
const WorkflowDef = definition.WorkflowDef;

pub const preset_workflows = struct {
    pub const code_review = WorkflowDef{
        .id = "code-review",
        .name = "Multi-Perspective Code Review",
        .description = "Analyzes code from multiple angles then synthesizes findings",
        .steps = &.{
            .{
                .id = "analyze",
                .description = "Analyze code structure and identify areas of concern",
                .depends_on = &.{},
                .required_capabilities = &.{.code_review},
                .input_keys = &.{"task:input"},
                .output_key = "analyze:output",
                .prompt_template = "Analyze the following code and identify structural concerns:\n\n{input}",
            },
            .{
                .id = "security",
                .description = "Check for security vulnerabilities",
                .depends_on = &.{},
                .required_capabilities = &.{.security_audit},
                .input_keys = &.{"task:input"},
                .output_key = "security:output",
                .prompt_template = "Review the following code for security vulnerabilities:\n\n{input}",
            },
            .{
                .id = "quality",
                .description = "Evaluate code quality and suggest improvements",
                .depends_on = &.{},
                .required_capabilities = &.{ .critique, .code_review },
                .input_keys = &.{"task:input"},
                .output_key = "quality:output",
                .prompt_template = "Evaluate the quality of this code and suggest improvements:\n\n{input}",
            },
            .{
                .id = "synthesize",
                .description = "Combine all review findings into a final report",
                .depends_on = &.{ "analyze", "security", "quality" },
                .required_capabilities = &.{.synthesis},
                .input_keys = &.{ "analyze:output", "security:output", "quality:output" },
                .output_key = "review:final",
                .prompt_template = "Synthesize these code review findings into a final report:\n\n" ++
                    "Structure Analysis:\n{input}\n\n" ++
                    "Security Review:\n{input}\n\n" ++
                    "Quality Assessment:\n{input}",
            },
        },
    };

    pub const research = WorkflowDef{
        .id = "research",
        .name = "Research and Analysis",
        .description = "Researches a topic, analyzes findings, and produces a report",
        .steps = &.{
            .{
                .id = "gather",
                .description = "Gather information about the topic",
                .depends_on = &.{},
                .required_capabilities = &.{.summarization},
                .input_keys = &.{"task:input"},
                .output_key = "gather:output",
                .prompt_template = "Research and gather key information about:\n\n{input}",
            },
            .{
                .id = "analyze",
                .description = "Analyze the gathered information",
                .depends_on = &.{"gather"},
                .required_capabilities = &.{.data_analysis},
                .input_keys = &.{"gather:output"},
                .output_key = "analyze:output",
                .prompt_template = "Analyze the following research findings:\n\n{input}",
            },
            .{
                .id = "report",
                .description = "Write a clear report from the analysis",
                .depends_on = &.{"analyze"},
                .required_capabilities = &.{.doc_writing},
                .input_keys = &.{"analyze:output"},
                .output_key = "report:final",
                .prompt_template = "Write a clear, well-structured report based on this analysis:\n\n{input}",
            },
        },
    };

    pub const implement_feature = WorkflowDef{
        .id = "implement-feature",
        .name = "Feature Implementation",
        .description = "Plans, implements, tests, and reviews a feature",
        .steps = &.{
            .{
                .id = "plan",
                .description = "Design the implementation plan",
                .depends_on = &.{},
                .required_capabilities = &.{ .problem_decomposition, .planning },
                .input_keys = &.{"task:input"},
                .output_key = "plan:output",
                .prompt_template = "Design an implementation plan for:\n\n{input}",
            },
            .{
                .id = "implement",
                .description = "Write the implementation code",
                .depends_on = &.{"plan"},
                .required_capabilities = &.{.code_generation},
                .input_keys = &.{ "task:input", "plan:output" },
                .output_key = "implement:output",
                .prompt_template = "Implement the following based on the plan:\n\n{input}",
            },
            .{
                .id = "test",
                .description = "Write tests for the implementation",
                .depends_on = &.{"implement"},
                .required_capabilities = &.{.test_writing},
                .input_keys = &.{"implement:output"},
                .output_key = "test:output",
                .prompt_template = "Write comprehensive tests for:\n\n{input}",
            },
            .{
                .id = "review",
                .description = "Review the implementation and tests",
                .depends_on = &.{ "implement", "test" },
                .required_capabilities = &.{ .code_review, .critique },
                .input_keys = &.{ "implement:output", "test:output" },
                .output_key = "review:final",
                .prompt_template = "Review the implementation and tests for quality:\n\n{input}",
            },
        },
    };
};

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
