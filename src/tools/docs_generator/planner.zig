const std = @import("std");

const config = @import("config.zig");
const module_docs = @import("generators/module_docs.zig");
const api_reference = @import("generators/api_reference.zig");
const examples = @import("generators/examples.zig");
const performance_guide = @import("generators/performance_guide.zig");
const definitions_reference = @import("generators/definitions_reference.zig");
const code_index = @import("generators/code_index.zig");
const search_index = @import("generators/search_index.zig");
const docs_index = @import("generators/docs_index.zig");
const readme_redirect = @import("generators/readme_redirect.zig");
const native_docs = @import("generators/native_docs.zig");

pub const StepCategory = enum {
    configuration,
    content,
    assets,
    workflow,
    native_docs,
};

pub const Context = struct {
    allocator: std.mem.Allocator,
};

pub const StepFn = *const fn (*Context) anyerror!void;

pub const GenerationStep = struct {
    name: []const u8,
    category: StepCategory,
    run: StepFn,
};

pub const GenerationPlan = struct {
    context: Context,
    steps: []const GenerationStep,

    pub fn execute(self: GenerationPlan) !void {
        var context = self.context;
        for (self.steps) |step| {
            std.log.info(
                "Running documentation step ({s}): {s}",
                .{ @tagName(step.category), step.name },
            );
            try step.run(&context);
        }
    }
};

pub fn buildDefaultPlan(allocator: std.mem.Allocator) GenerationPlan {
    return GenerationPlan{
        .context = .{ .allocator = allocator },
        .steps = &default_steps,
    };
}

pub fn defaultSteps() []const GenerationStep {
    return &default_steps;
}

fn allocatorStep(
    comptime name: []const u8,
    category: StepCategory,
    comptime func: fn (std.mem.Allocator) anyerror!void,
) GenerationStep {
    return GenerationStep{
        .name = name,
        .category = category,
        .run = struct {
            fn call(ctx: *Context) anyerror!void {
                try func(ctx.allocator);
            }
        }.call,
    };
}

const default_steps = [_]GenerationStep{
    allocatorStep("Disable Jekyll", .configuration, config.generateNoJekyll),
    allocatorStep("Write Jekyll Config", .configuration, config.generateJekyllConfig),
    allocatorStep("Write Documentation Layout", .configuration, config.generateGitHubPagesLayout),
    allocatorStep("Write Navigation Data", .configuration, config.generateNavigationData),
    allocatorStep("Write SEO Metadata", .configuration, config.generateSEOMetadata),
    allocatorStep("Generate Module Docs", .content, module_docs.generateModuleDocs),
    allocatorStep("Generate API Reference", .content, api_reference.generateApiReference),
    allocatorStep("Generate Examples", .content, examples.generateExamples),
    allocatorStep("Generate Performance Guide", .content, performance_guide.generatePerformanceGuide),
    allocatorStep("Generate Definitions Reference", .content, definitions_reference.generateDefinitionsReference),
    allocatorStep("Generate Code API Index", .content, code_index.generateCodeApiIndex),
    allocatorStep("Generate Search Index", .content, search_index.generateSearchIndex),
    allocatorStep("Write GitHub Pages Assets", .assets, config.generateGitHubPagesAssets),
    allocatorStep("Write Docs Index", .assets, docs_index.generateDocsIndexHtml),
    allocatorStep("Write README Redirect", .assets, readme_redirect.generateReadmeRedirect),
    allocatorStep("Write GitHub Actions Workflow", .workflow, config.generateGitHubActionsWorkflow),
    allocatorStep("Generate Zig Native Docs", .native_docs, native_docs.generateZigNativeDocs),
};
