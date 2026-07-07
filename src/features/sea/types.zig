const std = @import("std");

/// Typed memory taxonomy: every durable memory record carries a kind that
/// classifies what sort of knowledge it holds (design reference:
/// `docs/spec/sea-design-extract.mdx` §2). The kind feeds task-fit scoring,
/// cluster-diversity budgeting, and default importance at insert time.
pub const MemoryKind = enum(u8) {
    note,
    user_preference,
    project_decision,
    code_fact,
    tool_output,
    benchmark,
    constraint,
    contradiction,
    summary,

    pub fn parse(s: []const u8) ?MemoryKind {
        return std.meta.stringToEnum(MemoryKind, s);
    }

    pub fn text(self: MemoryKind) []const u8 {
        return @tagName(self);
    }
};

/// Provenance/trust ladder: every memory record carries an authority that maps
/// provenance to a scalar trust weight (design reference:
/// `docs/spec/sea-design-extract.mdx` §3). The ladder encodes a clear policy —
/// inference is least trusted, user statements rank above inference, tool and
/// file verification rank higher, and a system-pinned invariant is absolute.
pub const Authority = enum(u8) {
    inferred,
    user_stated,
    tool_verified,
    file_verified,
    system_pinned,

    pub fn parse(s: []const u8) ?Authority {
        return std.meta.stringToEnum(Authority, s);
    }

    pub fn text(self: Authority) []const u8 {
        return @tagName(self);
    }

    /// Deterministic scalar trust weight for this authority rung.
    pub fn score(self: Authority) f32 {
        return switch (self) {
            .inferred => 0.30,
            .user_stated => 0.78,
            .tool_verified => 0.86,
            .file_verified => 0.90,
            .system_pinned => 1.00,
        };
    }
};

test "MemoryKind round-trips through parse and text" {
    const kinds = std.meta.tags(MemoryKind);
    for (kinds) |kind| {
        const parsed = MemoryKind.parse(kind.text());
        try std.testing.expect(parsed != null);
        try std.testing.expectEqual(kind, parsed.?);
    }
}

test "MemoryKind.parse returns null for unknown names" {
    try std.testing.expect(MemoryKind.parse("bogus") == null);
}

test "Authority round-trips through parse and text" {
    const authorities = std.meta.tags(Authority);
    for (authorities) |auth| {
        const parsed = Authority.parse(auth.text());
        try std.testing.expect(parsed != null);
        try std.testing.expectEqual(auth, parsed.?);
    }
}

test "Authority.score maps each rung to its trust weight" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.30), Authority.inferred.score(), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.78), Authority.user_stated.score(), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.86), Authority.tool_verified.score(), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.90), Authority.file_verified.score(), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.00), Authority.system_pinned.score(), 1e-5);
}

test {
    std.testing.refAllDecls(@This());
}
