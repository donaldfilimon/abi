const Agent = @import("../agent.zig");
const common = @import("../config/common.zig");

pub fn run(controller: *Agent.Controller, input: common.Schema.SummarizeInput) !common.Schema.SummarizeOutput {
    return controller.summarize(input);
}
