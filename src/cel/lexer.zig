const std = @import("std");

// ── Token types ───────────────────────────────────────────────────────

pub const Loc = struct {
    start: u32 = 0,
    end: u32 = 0,
};

pub const Token = struct {
    tag: Tag,
    loc: Loc,
};

pub const Tag = enum {
    // Literals
    integer_literal,
    float_literal,
    string_literal,
    char_literal,

    // Identifiers
    identifier,
    builtin_identifier, // @import, @intCast, etc.
    doc_comment, // /// ...

    // Keywords
    kw_fn,
    kw_return,
    kw_const,
    kw_var,
    kw_if,
    kw_else,
    kw_while,
    kw_for,
    kw_break,
    kw_continue,
    kw_switch,
    kw_struct,
    kw_enum,
    kw_union,
    kw_pub,
    kw_extern,
    kw_export,
    kw_inline,
    kw_comptime,
    kw_try,
    kw_catch,
    kw_orelse,
    kw_and,
    kw_or,
    kw_true,
    kw_false,
    kw_null,
    kw_undefined,
    kw_unreachable,
    kw_defer,
    kw_errdefer,
    kw_test,
    kw_import,
    kw_type,
    kw_void,
    kw_noreturn,
    kw_error,
    kw_mut,
    kw_let,
    kw_as,
    kw_impl,
    kw_trait,
    kw_self,
    kw_match,

    // Single-character operators / delimiters
    l_paren, // (
    r_paren, // )
    l_brace, // {
    r_brace, // }
    l_bracket, // [
    r_bracket, // ]
    semicolon, // ;
    colon, // :
    comma, // ,
    dot, // .
    at, // @ (bare)
    hash, // #
    tilde, // ~
    question, // ?

    // Operators
    plus, // +
    minus, // -
    star, // *
    slash, // /
    percent, // %
    ampersand, // &
    pipe, // |
    caret, // ^
    bang, // !
    less, // <
    greater, // >
    equal, // =

    // Multi-character operators
    equal_equal, // ==
    bang_equal, // !=
    less_equal, // <=
    greater_equal, // >=
    ampersand_ampersand, // &&
    pipe_pipe, // ||
    less_less, // <<
    greater_greater, // >>
    plus_equal, // +=
    minus_equal, // -=
    star_equal, // *=
    slash_equal, // /=
    percent_equal, // %=
    ampersand_equal, // &=
    pipe_equal, // |=
    caret_equal, // ^=
    less_less_equal, // <<=
    greater_greater_equal, // >>=
    dot_dot, // ..
    dot_dot_dot, // ...
    dot_star, // .*
    dot_question, // .?
    arrow, // ->
    fat_arrow, // =>

    // Special
    eof,
    invalid,

    /// Look up whether an identifier is a keyword. Returns null if not.
    pub fn keyword(ident: []const u8) ?Tag {
        const map = std.StaticStringMap(Tag).initComptime(.{
            .{ "fn", .kw_fn },
            .{ "return", .kw_return },
            .{ "const", .kw_const },
            .{ "var", .kw_var },
            .{ "if", .kw_if },
            .{ "else", .kw_else },
            .{ "while", .kw_while },
            .{ "for", .kw_for },
            .{ "break", .kw_break },
            .{ "continue", .kw_continue },
            .{ "switch", .kw_switch },
            .{ "struct", .kw_struct },
            .{ "enum", .kw_enum },
            .{ "union", .kw_union },
            .{ "pub", .kw_pub },
            .{ "extern", .kw_extern },
            .{ "export", .kw_export },
            .{ "inline", .kw_inline },
            .{ "comptime", .kw_comptime },
            .{ "try", .kw_try },
            .{ "catch", .kw_catch },
            .{ "orelse", .kw_orelse },
            .{ "and", .kw_and },
            .{ "or", .kw_or },
            .{ "true", .kw_true },
            .{ "false", .kw_false },
            .{ "null", .kw_null },
            .{ "undefined", .kw_undefined },
            .{ "unreachable", .kw_unreachable },
            .{ "defer", .kw_defer },
            .{ "errdefer", .kw_errdefer },
            .{ "test", .kw_test },
            .{ "import", .kw_import },
            .{ "type", .kw_type },
            .{ "void", .kw_void },
            .{ "noreturn", .kw_noreturn },
            .{ "error", .kw_error },
            .{ "mut", .kw_mut },
            .{ "let", .kw_let },
            .{ "as", .kw_as },
            .{ "impl", .kw_impl },
            .{ "trait", .kw_trait },
            .{ "self", .kw_self },
            .{ "match", .kw_match },
        });
        return map.get(ident);
    }
};

// ── Lexer ─────────────────────────────────────────────────────────────

pub const Lexer = struct {
    source: []const u8,
    index: u32,

    pub fn init(src: []const u8) Lexer {
        return .{ .source = src, .index = 0 };
    }

    pub fn next(self: *Lexer) Token {
        self.skipWhitespace();

        if (self.index >= self.source.len) {
            return .{ .tag = .eof, .loc = .{ .start = self.index, .end = self.index } };
        }

        const start = self.index;
        const c = self.advance();

        const tag: Tag = switch (c) {
            '(' => .l_paren,
            ')' => .r_paren,
            '{' => .l_brace,
            '}' => .r_brace,
            '[' => .l_bracket,
            ']' => .r_bracket,
            ';' => .semicolon,
            ':' => .colon,
            ',' => .comma,
            '#' => .hash,
            '~' => .tilde,
            '?' => .question,

            '.' => blk: {
                if (self.peek()) |p| {
                    if (p == '.') {
                        _ = self.advance();
                        if (self.peek() == @as(u8, '.')) {
                            _ = self.advance();
                            break :blk .dot_dot_dot;
                        }
                        break :blk .dot_dot;
                    }
                    if (p == '*') {
                        _ = self.advance();
                        break :blk .dot_star;
                    }
                    if (p == '?') {
                        _ = self.advance();
                        break :blk .dot_question;
                    }
                }
                break :blk .dot;
            },

            '+' => if (self.peek() == @as(u8, '=')) b: {
                _ = self.advance();
                break :b .plus_equal;
            } else .plus,

            '-' => blk: {
                if (self.peek() == @as(u8, '=')) {
                    _ = self.advance();
                    break :blk .minus_equal;
                }
                if (self.peek() == @as(u8, '>')) {
                    _ = self.advance();
                    break :blk .arrow;
                }
                break :blk .minus;
            },

            '*' => if (self.peek() == @as(u8, '=')) b: {
                _ = self.advance();
                break :b .star_equal;
            } else .star,

            '/' => blk: {
                if (self.peek() == @as(u8, '/')) {
                    // Comment — check for doc comment (///)
                    _ = self.advance(); // consume second /
                    if (self.peek() == @as(u8, '/')) {
                        // Doc comment
                        _ = self.advance(); // consume third /
                        const doc_start = self.index;
                        self.skipToEndOfLine();
                        return .{
                            .tag = .doc_comment,
                            .loc = .{ .start = start, .end = self.index },
                        };
                        _ = doc_start; // suppress unused
                    }
                    // Regular line comment — skip
                    self.skipToEndOfLine();
                    return self.next();
                }
                if (self.peek() == @as(u8, '=')) {
                    _ = self.advance();
                    break :blk .slash_equal;
                }
                break :blk .slash;
            },

            '%' => if (self.peek() == @as(u8, '=')) b: {
                _ = self.advance();
                break :b .percent_equal;
            } else .percent,

            '&' => blk: {
                if (self.peek() == @as(u8, '&')) {
                    _ = self.advance();
                    break :blk .ampersand_ampersand;
                }
                if (self.peek() == @as(u8, '=')) {
                    _ = self.advance();
                    break :blk .ampersand_equal;
                }
                break :blk .ampersand;
            },

            '|' => blk: {
                if (self.peek() == @as(u8, '|')) {
                    _ = self.advance();
                    break :blk .pipe_pipe;
                }
                if (self.peek() == @as(u8, '=')) {
                    _ = self.advance();
                    break :blk .pipe_equal;
                }
                break :blk .pipe;
            },

            '^' => if (self.peek() == @as(u8, '=')) b: {
                _ = self.advance();
                break :b .caret_equal;
            } else .caret,

            '!' => if (self.peek() == @as(u8, '=')) b: {
                _ = self.advance();
                break :b .bang_equal;
            } else .bang,

            '<' => blk: {
                if (self.peek() == @as(u8, '=')) {
                    _ = self.advance();
                    break :blk .less_equal;
                }
                if (self.peek() == @as(u8, '<')) {
                    _ = self.advance();
                    if (self.peek() == @as(u8, '=')) {
                        _ = self.advance();
                        break :blk .less_less_equal;
                    }
                    break :blk .less_less;
                }
                break :blk .less;
            },

            '>' => blk: {
                if (self.peek() == @as(u8, '=')) {
                    _ = self.advance();
                    break :blk .greater_equal;
                }
                if (self.peek() == @as(u8, '>')) {
                    _ = self.advance();
                    if (self.peek() == @as(u8, '=')) {
                        _ = self.advance();
                        break :blk .greater_greater_equal;
                    }
                    break :blk .greater_greater;
                }
                break :blk .greater;
            },

            '=' => blk: {
                if (self.peek() == @as(u8, '=')) {
                    _ = self.advance();
                    break :blk .equal_equal;
                }
                if (self.peek() == @as(u8, '>')) {
                    _ = self.advance();
                    break :blk .fat_arrow;
                }
                break :blk .equal;
            },

            '@' => {
                // Builtin identifier: @name
                if (self.peek()) |p| {
                    if (isIdentStart(p)) {
                        self.readIdentBody();
                        return .{
                            .tag = .builtin_identifier,
                            .loc = .{ .start = start, .end = self.index },
                        };
                    }
                }
                return .{ .tag = .at, .loc = .{ .start = start, .end = self.index } };
            },

            '\'' => {
                return self.readCharLiteral(start);
            },

            '"' => {
                return self.readStringLiteral(start);
            },

            '0' => {
                // Check for 0x, 0b, 0o prefixes
                if (self.peek()) |p| {
                    if (p == 'x' or p == 'X') {
                        _ = self.advance();
                        self.readDigits(isHexDigit);
                        return .{ .tag = .integer_literal, .loc = .{ .start = start, .end = self.index } };
                    }
                    if (p == 'b' or p == 'B') {
                        _ = self.advance();
                        self.readDigits(isBinDigit);
                        return .{ .tag = .integer_literal, .loc = .{ .start = start, .end = self.index } };
                    }
                    if (p == 'o' or p == 'O') {
                        _ = self.advance();
                        self.readDigits(isOctDigit);
                        return .{ .tag = .integer_literal, .loc = .{ .start = start, .end = self.index } };
                    }
                }
                return self.readNumberRest(start);
            },

            else => {
                if (c >= '1' and c <= '9') {
                    return self.readNumberRest(start);
                }
                if (isIdentStart(c)) {
                    self.readIdentBody();
                    const text = self.source[start..self.index];
                    const kw = Tag.keyword(text);
                    return .{
                        .tag = kw orelse .identifier,
                        .loc = .{ .start = start, .end = self.index },
                    };
                }
                return .{ .tag = .invalid, .loc = .{ .start = start, .end = self.index } };
            },
        };

        return .{ .tag = tag, .loc = .{ .start = start, .end = self.index } };
    }

    // ── Helpers ───────────────────────────────────────────────────────

    fn skipWhitespace(self: *Lexer) void {
        while (self.index < self.source.len) {
            const c = self.source[self.index];
            if (c == ' ' or c == '\t' or c == '\n' or c == '\r') {
                self.index += 1;
            } else {
                break;
            }
        }
    }

    fn skipToEndOfLine(self: *Lexer) void {
        while (self.index < self.source.len and self.source[self.index] != '\n') {
            self.index += 1;
        }
    }

    fn advance(self: *Lexer) u8 {
        const c = self.source[self.index];
        self.index += 1;
        return c;
    }

    fn peek(self: *Lexer) ?u8 {
        if (self.index >= self.source.len) return null;
        return self.source[self.index];
    }

    fn peekNext(self: *Lexer) ?u8 {
        if (self.index + 1 >= self.source.len) return null;
        return self.source[self.index + 1];
    }

    fn readIdentBody(self: *Lexer) void {
        while (self.index < self.source.len) {
            const c = self.source[self.index];
            if (isIdentContinue(c)) {
                self.index += 1;
            } else {
                break;
            }
        }
    }

    fn readDigits(self: *Lexer, comptime pred: fn (u8) bool) void {
        while (self.index < self.source.len) {
            const c = self.source[self.index];
            if (pred(c) or c == '_') {
                self.index += 1;
            } else {
                break;
            }
        }
    }

    fn readNumberRest(self: *Lexer, start: u32) Token {
        // Continue reading decimal digits
        self.readDigits(isDecDigit);

        // Check for float: decimal point (but not .. range)
        if (self.peek() == @as(u8, '.') and self.peekNext() != @as(u8, '.')) {
            _ = self.advance(); // consume '.'
            self.readDigits(isDecDigit);
            // Scientific notation
            if (self.peek()) |p| {
                if (p == 'e' or p == 'E') {
                    _ = self.advance();
                    if (self.peek()) |s| {
                        if (s == '+' or s == '-') _ = self.advance();
                    }
                    self.readDigits(isDecDigit);
                }
            }
            return .{ .tag = .float_literal, .loc = .{ .start = start, .end = self.index } };
        }

        // Scientific notation on integer makes it a float
        if (self.peek()) |p| {
            if (p == 'e' or p == 'E') {
                _ = self.advance();
                if (self.peek()) |s| {
                    if (s == '+' or s == '-') _ = self.advance();
                }
                self.readDigits(isDecDigit);
                return .{ .tag = .float_literal, .loc = .{ .start = start, .end = self.index } };
            }
        }

        return .{ .tag = .integer_literal, .loc = .{ .start = start, .end = self.index } };
    }

    fn readStringLiteral(self: *Lexer, start: u32) Token {
        while (self.index < self.source.len) {
            const c = self.advance();
            if (c == '\\') {
                // Skip the escaped character
                if (self.index < self.source.len) self.index += 1;
            } else if (c == '"') {
                return .{ .tag = .string_literal, .loc = .{ .start = start, .end = self.index } };
            } else if (c == '\n') {
                // Unterminated string
                break;
            }
        }
        return .{ .tag = .invalid, .loc = .{ .start = start, .end = self.index } };
    }

    fn readCharLiteral(self: *Lexer, start: u32) Token {
        if (self.index >= self.source.len) {
            return .{ .tag = .invalid, .loc = .{ .start = start, .end = self.index } };
        }
        const c = self.advance();
        if (c == '\\') {
            // Escape sequence
            if (self.index < self.source.len) {
                const esc = self.advance();
                if (esc == 'x') {
                    // \xNN — two hex digits
                    if (self.index + 1 < self.source.len) {
                        self.index += 2;
                    }
                }
            }
        }
        // Expect closing quote
        if (self.peek() == @as(u8, '\'')) {
            _ = self.advance();
            return .{ .tag = .char_literal, .loc = .{ .start = start, .end = self.index } };
        }
        return .{ .tag = .invalid, .loc = .{ .start = start, .end = self.index } };
    }

    // ── Character classification ──────────────────────────────────────

    fn isIdentStart(c: u8) bool {
        return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c == '_';
    }

    fn isIdentContinue(c: u8) bool {
        return isIdentStart(c) or (c >= '0' and c <= '9');
    }

    fn isDecDigit(c: u8) bool {
        return c >= '0' and c <= '9';
    }

    fn isHexDigit(c: u8) bool {
        return (c >= '0' and c <= '9') or (c >= 'a' and c <= 'f') or (c >= 'A' and c <= 'F');
    }

    fn isBinDigit(c: u8) bool {
        return c == '0' or c == '1';
    }

    fn isOctDigit(c: u8) bool {
        return c >= '0' and c <= '7';
    }

    /// Collect all tokens into a slice (for testing).
    pub fn tokenize(allocator: std.mem.Allocator, src: []const u8) ![]Token {
        var lex = Lexer.init(src);
        var list = std.ArrayList(Token).init(allocator);
        while (true) {
            const tok = lex.next();
            try list.append(tok);
            if (tok.tag == .eof) break;
        }
        return list.toOwnedSlice();
    }

    /// Return the source text of a token.
    pub fn slice(self: Lexer, tok: Token) []const u8 {
        return self.source[tok.loc.start..tok.loc.end];
    }
};

// ── Tests ─────────────────────────────────────────────────────────────

fn expectTags(src: []const u8, expected: []const Tag) !void {
    var lex = Lexer.init(src);
    for (expected) |exp| {
        const tok = lex.next();
        try std.testing.expectEqual(exp, tok.tag);
    }
}

fn expectTagsSlice(src: []const u8, expected: []const Tag) !void {
    try expectTags(src, expected);
}

test "simple expression: 1 + 2" {
    try expectTags("1 + 2", &.{ .integer_literal, .plus, .integer_literal, .eof });
}

test "variable declaration: const x: i32 = 42;" {
    try expectTags("const x: i32 = 42;", &.{
        .kw_const, .identifier, .colon, .identifier, .equal, .integer_literal, .semicolon, .eof,
    });
}

test "function declaration" {
    try expectTags("fn add(a: i32, b: i32) i32 { return a + b; }", &.{
        .kw_fn,      .identifier, .l_paren,
        .identifier, .colon,      .identifier,
        .comma,      .identifier, .colon,
        .identifier, .r_paren,    .identifier,
        .l_brace,    .kw_return,  .identifier,
        .plus,       .identifier, .semicolon,
        .r_brace,    .eof,
    });
}

test "comparison operators" {
    try expectTags("== != <= >= < >", &.{
        .equal_equal, .bang_equal, .less_equal, .greater_equal, .less, .greater, .eof,
    });
}

test "logical operators" {
    try expectTags("&& ||", &.{ .ampersand_ampersand, .pipe_pipe, .eof });
}

test "shift operators" {
    try expectTags("<< >> <<= >>=", &.{ .less_less, .greater_greater, .less_less_equal, .greater_greater_equal, .eof });
}

test "assignment operators" {
    try expectTags("+= -= *= /= %= &= |= ^=", &.{
        .plus_equal,    .minus_equal,     .star_equal, .slash_equal,
        .percent_equal, .ampersand_equal, .pipe_equal, .caret_equal,
        .eof,
    });
}

test "dot operators" {
    try expectTags(".. ... .* .?", &.{ .dot_dot, .dot_dot_dot, .dot_star, .dot_question, .eof });
}

test "arrow operators" {
    try expectTags("-> =>", &.{ .arrow, .fat_arrow, .eof });
}

test "hex, binary, octal literals" {
    try expectTags("0xFF 0b1010 0o77", &.{ .integer_literal, .integer_literal, .integer_literal, .eof });
}

test "float literals" {
    try expectTags("3.14 1.0e10 2.5E-3", &.{ .float_literal, .float_literal, .float_literal, .eof });
}

test "underscore separators in numbers" {
    try expectTags("1_000_000 0xFF_FF", &.{ .integer_literal, .integer_literal, .eof });
}

test "string literals with escapes" {
    var lex = Lexer.init("\"hello\\nworld\" \"tab\\there\"");
    const t1 = lex.next();
    try std.testing.expectEqual(Tag.string_literal, t1.tag);
    const t2 = lex.next();
    try std.testing.expectEqual(Tag.string_literal, t2.tag);
    try std.testing.expectEqual(Tag.eof, lex.next().tag);
}

test "char literals" {
    try expectTags("'a' '\\n' '\\0'", &.{ .char_literal, .char_literal, .char_literal, .eof });
}

test "doc comments" {
    var lex = Lexer.init("/// A doc comment\nconst x = 1;");
    const doc = lex.next();
    try std.testing.expectEqual(Tag.doc_comment, doc.tag);
    try std.testing.expectEqual(Tag.kw_const, lex.next().tag);
}

test "regular comments are skipped" {
    try expectTags("// this is a comment\n42", &.{ .integer_literal, .eof });
}

test "builtin identifiers" {
    var lex = Lexer.init("@import @intCast");
    const t1 = lex.next();
    try std.testing.expectEqual(Tag.builtin_identifier, t1.tag);
    const t2 = lex.next();
    try std.testing.expectEqual(Tag.builtin_identifier, t2.tag);
    try std.testing.expectEqual(Tag.eof, lex.next().tag);
}

test "invalid character" {
    var lex = Lexer.init("$");
    const tok = lex.next();
    try std.testing.expectEqual(Tag.invalid, tok.tag);
}
