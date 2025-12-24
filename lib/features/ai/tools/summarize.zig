const Agent = @import("../agent.zig");
const Schema = @import("../schema.zig");

pub fn run(controller: *Agent.Controller, input: Schema.SummarizeInput) !Schema.SummarizeOutput {
    return controller.summarize(input);
}
