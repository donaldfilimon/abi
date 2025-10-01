const Agent = @import("mod.zig");
const Schema = @import("schema.zig");

pub fn runSummarize(controller: *Agent.Controller, input: Schema.SummarizeInput) !Schema.SummarizeOutput {
    return controller.summarize(input);
}
