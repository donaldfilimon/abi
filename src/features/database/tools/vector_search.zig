pub const Input = struct {
    query: []const f32,
    k: u16 = 10,
};

pub const Match = struct {
    id: []const u8,
    score: f32,
};

pub fn run(allocator: std.mem.Allocator, in: Input) ![]Match {
    if (in.query.len == 0) return error.EmptyQuery;
    _ = allocator;
    return &.{};
}
