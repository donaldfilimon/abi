const Agent = @import("../../../agent/mod.zig");
const Schema = @import("../../../agent/schema.zig");

pub fn run(controller: *Agent.Controller, input: Schema.SummarizeInput) !Schema.SummarizeOutput {
    return controller.summarize(input);
}
