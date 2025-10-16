pub fn validateNonEmpty(s: []const u8) !void {
    if (s.len == 0) return error.Empty;
}

pub const SummarizeInput = struct {
    doc_id: []const u8,
    max_tokens: u16 = 512,

    pub fn validate(self: *const SummarizeInput) !void {
        try validateNonEmpty(self.doc_id);
        if (self.max_tokens == 0) return error.InvalidMaxTokens;
    }
};

pub const SummarizeOutput = struct {
    summary: []const u8,
    tokens_used: u32,

    pub fn validate(self: *const SummarizeOutput) !void {
        try validateNonEmpty(self.summary);
    }
};
