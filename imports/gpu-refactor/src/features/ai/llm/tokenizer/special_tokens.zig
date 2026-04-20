//! Special token handling for LLM tokenizers.
//!
//! Defines standard special tokens like BOS, EOS, PAD, and UNK.

const std = @import("std");

/// Standard special token types.
pub const SpecialTokenType = enum {
    bos, // Beginning of sequence
    eos, // End of sequence
    pad, // Padding
    unk, // Unknown token
    sep, // Separator
    cls, // Classification
    mask, // Mask for MLM
    newline, // Explicit newline
};

/// Special tokens configuration.
pub const SpecialTokens = struct {
    /// Beginning of sequence token ID
    bos_id: u32,
    /// End of sequence token ID
    eos_id: u32,
    /// Padding token ID
    pad_id: u32,
    /// Unknown token ID
    unk_id: u32,
    /// Separator token ID (optional)
    sep_id: ?u32,
    /// Newline token ID (optional)
    nl_id: ?u32,

    /// String representations
    bos_str: []const u8,
    eos_str: []const u8,
    pad_str: []const u8,
    unk_str: []const u8,

    /// Initialize with default LLaMA-style special tokens.
    pub fn initDefault() SpecialTokens {
        return .{
            .bos_id = 1,
            .eos_id = 2,
            .pad_id = 0,
            .unk_id = 0,
            .sep_id = null,
            .nl_id = null,
            .bos_str = "<s>",
            .eos_str = "</s>",
            .pad_str = "<pad>",
            .unk_str = "<unk>",
        };
    }

    /// Initialize with ChatML-style special tokens.
    pub fn initChatML() SpecialTokens {
        return .{
            .bos_id = 1,
            .eos_id = 2,
            .pad_id = 0,
            .unk_id = 0,
            .sep_id = 3,
            .nl_id = 13,
            .bos_str = "<|im_start|>",
            .eos_str = "<|im_end|>",
            .pad_str = "<pad>",
            .unk_str = "<unk>",
        };
    }

    /// Initialize with specific token IDs.
    pub fn initCustom(
        bos_id: u32,
        eos_id: u32,
        pad_id: u32,
        unk_id: u32,
    ) SpecialTokens {
        return .{
            .bos_id = bos_id,
            .eos_id = eos_id,
            .pad_id = pad_id,
            .unk_id = unk_id,
            .sep_id = null,
            .nl_id = null,
            .bos_str = "<s>",
            .eos_str = "</s>",
            .pad_str = "<pad>",
            .unk_str = "<unk>",
        };
    }

    /// Check if a token ID is a special token.
    pub fn isSpecial(self: *const SpecialTokens, token_id: u32) bool {
        return token_id == self.bos_id or
            token_id == self.eos_id or
            token_id == self.pad_id or
            token_id == self.unk_id or
            (self.sep_id != null and token_id == self.sep_id.?) or
            (self.nl_id != null and token_id == self.nl_id.?);
    }

    /// Get the type of a special token.
    pub fn getType(self: *const SpecialTokens, token_id: u32) ?SpecialTokenType {
        if (token_id == self.bos_id) return .bos;
        if (token_id == self.eos_id) return .eos;
        if (token_id == self.pad_id) return .pad;
        if (token_id == self.unk_id) return .unk;
        if (self.sep_id != null and token_id == self.sep_id.?) return .sep;
        if (self.nl_id != null and token_id == self.nl_id.?) return .newline;
        return null;
    }

    /// Get string name of a special token.
    pub fn getName(self: *const SpecialTokens, token_id: u32) ?[]const u8 {
        if (token_id == self.bos_id) return self.bos_str;
        if (token_id == self.eos_id) return self.eos_str;
        if (token_id == self.pad_id) return self.pad_str;
        if (token_id == self.unk_id) return self.unk_str;
        if (self.sep_id != null and token_id == self.sep_id.?) return "<sep>";
        if (self.nl_id != null and token_id == self.nl_id.?) return "\n";
        return null;
    }

    /// Check if token is EOS.
    pub fn isEos(self: *const SpecialTokens, token_id: u32) bool {
        return token_id == self.eos_id;
    }

    /// Check if token is BOS.
    pub fn isBos(self: *const SpecialTokens, token_id: u32) bool {
        return token_id == self.bos_id;
    }

    /// Get all special token IDs as a slice.
    pub fn allIds(self: *const SpecialTokens) [6]?u32 {
        return .{
            self.bos_id,
            self.eos_id,
            self.pad_id,
            self.unk_id,
            self.sep_id,
            self.nl_id,
        };
    }
};

/// Chat template for formatting conversations.
pub const ChatTemplate = struct {
    /// System message prefix
    system_prefix: []const u8,
    /// System message suffix
    system_suffix: []const u8,
    /// User message prefix
    user_prefix: []const u8,
    /// User message suffix
    user_suffix: []const u8,
    /// Assistant message prefix
    assistant_prefix: []const u8,
    /// Assistant message suffix
    assistant_suffix: []const u8,

    /// LLaMA 2 chat template.
    pub fn llama2() ChatTemplate {
        return .{
            .system_prefix = "[INST] <<SYS>>\n",
            .system_suffix = "\n<</SYS>>\n\n",
            .user_prefix = "",
            .user_suffix = " [/INST] ",
            .assistant_prefix = "",
            .assistant_suffix = " </s><s>[INST] ",
        };
    }

    /// ChatML template (Mistral, etc.).
    pub fn chatml() ChatTemplate {
        return .{
            .system_prefix = "<|im_start|>system\n",
            .system_suffix = "<|im_end|>\n",
            .user_prefix = "<|im_start|>user\n",
            .user_suffix = "<|im_end|>\n",
            .assistant_prefix = "<|im_start|>assistant\n",
            .assistant_suffix = "<|im_end|>\n",
        };
    }

    /// Alpaca template.
    pub fn alpaca() ChatTemplate {
        return .{
            .system_prefix = "",
            .system_suffix = "\n\n",
            .user_prefix = "### Instruction:\n",
            .user_suffix = "\n\n",
            .assistant_prefix = "### Response:\n",
            .assistant_suffix = "\n\n",
        };
    }

    /// Format a conversation into a prompt string.
    pub fn format(
        self: *const ChatTemplate,
        allocator: std.mem.Allocator,
        system: ?[]const u8,
        messages: []const Message,
    ) ![]u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(allocator);

        // Add system message if present
        if (system) |sys| {
            try result.appendSlice(allocator, self.system_prefix);
            try result.appendSlice(allocator, sys);
            try result.appendSlice(allocator, self.system_suffix);
        }

        // Add conversation messages
        for (messages) |msg| {
            switch (msg.role) {
                .user => {
                    try result.appendSlice(allocator, self.user_prefix);
                    try result.appendSlice(allocator, msg.content);
                    try result.appendSlice(allocator, self.user_suffix);
                },
                .assistant => {
                    try result.appendSlice(allocator, self.assistant_prefix);
                    try result.appendSlice(allocator, msg.content);
                    try result.appendSlice(allocator, self.assistant_suffix);
                },
                .system => {
                    // System messages mid-conversation not standard
                    try result.appendSlice(allocator, msg.content);
                },
            }
        }

        // Add assistant prefix for generation
        try result.appendSlice(allocator, self.assistant_prefix);

        return result.toOwnedSlice(allocator);
    }
};

/// Message role in a conversation.
pub const Role = enum {
    user,
    assistant,
    system,
};

/// A single message in a conversation.
pub const Message = struct {
    role: Role,
    content: []const u8,
};

test "special tokens default" {
    const special = SpecialTokens.initDefault();

    try std.testing.expect(special.isSpecial(1)); // BOS
    try std.testing.expect(special.isSpecial(2)); // EOS
    try std.testing.expect(!special.isSpecial(100));

    try std.testing.expectEqual(SpecialTokenType.bos, special.getType(1).?);
    try std.testing.expectEqual(SpecialTokenType.eos, special.getType(2).?);
}

test "chat template formatting" {
    const allocator = std.testing.allocator;
    const template = ChatTemplate.llama2();

    const messages = [_]Message{
        .{ .role = .user, .content = "Hello!" },
    };

    const result = try template.format(allocator, "You are helpful.", &messages);
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "<<SYS>>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Hello!") != null);
}

test {
    std.testing.refAllDecls(@This());
}
