//! CLI entrypoint wrapper (preferred path for build.zig).
const cli = @import("cli");

pub fn main() !void {
    return cli.main();
}
