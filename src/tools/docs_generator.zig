const docgen = @import("docs_generator/main.zig");

pub fn main() !void {
    try docgen.main();
}
