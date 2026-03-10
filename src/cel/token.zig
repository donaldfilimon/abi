///! CEL Token definitions.
///! Comprehensive token types for the CEL systems programming language.
const std = @import("std");

pub const Token = struct {
    tag: Tag,
    loc: Loc,

    pub const Loc = struct {
        start: u32,
        end: u32,

        pub fn slice(self: Loc, source: []const u8) []const u8 {
            return source[self.start..self.end];
        }
    };

    pub const Tag = enum {
        // ── Keywords ──────────────────────────────────────────────
        keyword_fn,
        keyword_var,
        keyword_const,
        keyword_comptime,
        keyword_pub,
        keyword_return,
        keyword_if,
        keyword_else,
        keyword_for,
        keyword_while,
        keyword_switch,
        keyword_break,
        keyword_continue,
        keyword_defer,
        keyword_errdefer,
        keyword_try,
        keyword_catch,
        keyword_orelse,
        keyword_struct,
        keyword_enum,
        keyword_union,
        keyword_error,
        keyword_test,
        keyword_unreachable,
        keyword_undefined,
        keyword_null,
        keyword_true,
        keyword_false,
        keyword_inline,
        keyword_noalias,
        keyword_threadlocal,
        keyword_volatile,
        keyword_extern,
        keyword_export,
        keyword_align,
        keyword_linksection,
        keyword_nosuspend,
        keyword_async,
        keyword_await,
        keyword_suspend,
        keyword_resume,
        keyword_and,
        keyword_or,

        // ── Built-in scalar types ─────────────────────────────────
        keyword_void,
        keyword_bool,
        keyword_type,
        keyword_anytype,
        keyword_anyerror,
        keyword_anyframe,
        keyword_comptime_int,
        keyword_comptime_float,
        keyword_i8,
        keyword_i16,
        keyword_i32,
        keyword_i64,
        keyword_i128,
        keyword_u8,
        keyword_u16,
        keyword_u32,
        keyword_u64,
        keyword_u128,
        keyword_f16,
        keyword_f32,
        keyword_f64,
        keyword_usize,
        keyword_isize,

        // ── Operators ─────────────────────────────────────────────
        plus,
        minus,
        star,
        slash,
        percent,
        equal,
        equal_equal,
        bang_equal,
        less,
        less_equal,
        greater,
        greater_equal,
        ampersand,
        pipe,
        caret,
        tilde,
        less_less,
        greater_greater,
        ampersand_ampersand,
        pipe_pipe,
        plus_equal,
        minus_equal,
        star_equal,
        slash_equal,
        percent_equal,
        ampersand_equal,
        pipe_equal,
        caret_equal,
        less_less_equal,
        greater_greater_equal,
        dot,
        dot_star,
        dot_question,
        dot_dot,
        dot_dot_dot,
        arrow,
        fat_arrow,
        bang,

        // ── Delimiters ───────────────────────────────────────────
        l_paren,
        r_paren,
        l_brace,
        r_brace,
        l_bracket,
        r_bracket,
        semicolon,
        colon,
        comma,
        at_sign,

        // ── Literals and identifiers ─────────────────────────────
        identifier,
        builtin_identifier,
        integer_literal,
        float_literal,
        string_literal,
        char_literal,

        // ── Special ──────────────────────────────────────────────
        doc_comment,
        comment,
        eof,
        invalid,

        /// Return the fixed lexeme for this tag, or null if the token
        /// has variable content (identifiers, literals, etc.).
        pub fn lexeme(self: Tag) ?[]const u8 {
            return switch (self) {
                // operators
                .plus => "+",
                .minus => "-",
                .star => "*",
                .slash => "/",
                .percent => "%",
                .equal => "=",
                .equal_equal => "==",
                .bang_equal => "!=",
                .less => "<",
                .less_equal => "<=",
                .greater => ">",
                .greater_equal => ">=",
                .ampersand => "&",
                .pipe => "|",
                .caret => "^",
                .tilde => "~",
                .less_less => "<<",
                .greater_greater => ">>",
                .ampersand_ampersand => "&&",
                .pipe_pipe => "||",
                .plus_equal => "+=",
                .minus_equal => "-=",
                .star_equal => "*=",
                .slash_equal => "/=",
                .percent_equal => "%=",
                .ampersand_equal => "&=",
                .pipe_equal => "|=",
                .caret_equal => "^=",
                .less_less_equal => "<<=",
                .greater_greater_equal => ">>=",
                .dot => ".",
                .dot_star => ".*",
                .dot_question => ".?",
                .dot_dot => "..",
                .dot_dot_dot => "...",
                .arrow => "->",
                .fat_arrow => "=>",
                .bang => "!",
                // delimiters
                .l_paren => "(",
                .r_paren => ")",
                .l_brace => "{",
                .r_brace => "}",
                .l_bracket => "[",
                .r_bracket => "]",
                .semicolon => ";",
                .colon => ":",
                .comma => ",",
                .at_sign => "@",
                // keywords
                .keyword_fn => "fn",
                .keyword_var => "var",
                .keyword_const => "const",
                .keyword_comptime => "comptime",
                .keyword_pub => "pub",
                .keyword_return => "return",
                .keyword_if => "if",
                .keyword_else => "else",
                .keyword_for => "for",
                .keyword_while => "while",
                .keyword_switch => "switch",
                .keyword_break => "break",
                .keyword_continue => "continue",
                .keyword_defer => "defer",
                .keyword_errdefer => "errdefer",
                .keyword_try => "try",
                .keyword_catch => "catch",
                .keyword_orelse => "orelse",
                .keyword_struct => "struct",
                .keyword_enum => "enum",
                .keyword_union => "union",
                .keyword_error => "error",
                .keyword_test => "test",
                .keyword_unreachable => "unreachable",
                .keyword_undefined => "undefined",
                .keyword_null => "null",
                .keyword_true => "true",
                .keyword_false => "false",
                .keyword_inline => "inline",
                .keyword_noalias => "noalias",
                .keyword_threadlocal => "threadlocal",
                .keyword_volatile => "volatile",
                .keyword_extern => "extern",
                .keyword_export => "export",
                .keyword_align => "align",
                .keyword_linksection => "linksection",
                .keyword_nosuspend => "nosuspend",
                .keyword_async => "async",
                .keyword_await => "await",
                .keyword_suspend => "suspend",
                .keyword_resume => "resume",
                .keyword_and => "and",
                .keyword_or => "or",
                // scalar types
                .keyword_void => "void",
                .keyword_bool => "bool",
                .keyword_type => "type",
                .keyword_anytype => "anytype",
                .keyword_anyerror => "anyerror",
                .keyword_anyframe => "anyframe",
                .keyword_comptime_int => "comptime_int",
                .keyword_comptime_float => "comptime_float",
                .keyword_i8 => "i8",
                .keyword_i16 => "i16",
                .keyword_i32 => "i32",
                .keyword_i64 => "i64",
                .keyword_i128 => "i128",
                .keyword_u8 => "u8",
                .keyword_u16 => "u16",
                .keyword_u32 => "u32",
                .keyword_u64 => "u64",
                .keyword_u128 => "u128",
                .keyword_f16 => "f16",
                .keyword_f32 => "f32",
                .keyword_f64 => "f64",
                .keyword_usize => "usize",
                .keyword_isize => "isize",
                // variable-content tokens
                .identifier,
                .builtin_identifier,
                .integer_literal,
                .float_literal,
                .string_literal,
                .char_literal,
                .doc_comment,
                .comment,
                .eof,
                .invalid,
                => null,
            };
        }

        /// Look up a keyword by its string representation.
        /// Returns null if the string is not a keyword.
        pub fn keyword(bytes: []const u8) ?Tag {
            const map = comptime blk: {
                const kvs = [_]struct { []const u8, Tag }{
                    .{ "fn", .keyword_fn },
                    .{ "var", .keyword_var },
                    .{ "const", .keyword_const },
                    .{ "comptime", .keyword_comptime },
                    .{ "pub", .keyword_pub },
                    .{ "return", .keyword_return },
                    .{ "if", .keyword_if },
                    .{ "else", .keyword_else },
                    .{ "for", .keyword_for },
                    .{ "while", .keyword_while },
                    .{ "switch", .keyword_switch },
                    .{ "break", .keyword_break },
                    .{ "continue", .keyword_continue },
                    .{ "defer", .keyword_defer },
                    .{ "errdefer", .keyword_errdefer },
                    .{ "try", .keyword_try },
                    .{ "catch", .keyword_catch },
                    .{ "orelse", .keyword_orelse },
                    .{ "struct", .keyword_struct },
                    .{ "enum", .keyword_enum },
                    .{ "union", .keyword_union },
                    .{ "error", .keyword_error },
                    .{ "test", .keyword_test },
                    .{ "unreachable", .keyword_unreachable },
                    .{ "undefined", .keyword_undefined },
                    .{ "null", .keyword_null },
                    .{ "true", .keyword_true },
                    .{ "false", .keyword_false },
                    .{ "inline", .keyword_inline },
                    .{ "noalias", .keyword_noalias },
                    .{ "threadlocal", .keyword_threadlocal },
                    .{ "volatile", .keyword_volatile },
                    .{ "extern", .keyword_extern },
                    .{ "export", .keyword_export },
                    .{ "align", .keyword_align },
                    .{ "linksection", .keyword_linksection },
                    .{ "nosuspend", .keyword_nosuspend },
                    .{ "async", .keyword_async },
                    .{ "await", .keyword_await },
                    .{ "suspend", .keyword_suspend },
                    .{ "resume", .keyword_resume },
                    .{ "and", .keyword_and },
                    .{ "or", .keyword_or },
                    .{ "void", .keyword_void },
                    .{ "bool", .keyword_bool },
                    .{ "type", .keyword_type },
                    .{ "anytype", .keyword_anytype },
                    .{ "anyerror", .keyword_anyerror },
                    .{ "anyframe", .keyword_anyframe },
                    .{ "comptime_int", .keyword_comptime_int },
                    .{ "comptime_float", .keyword_comptime_float },
                    .{ "i8", .keyword_i8 },
                    .{ "i16", .keyword_i16 },
                    .{ "i32", .keyword_i32 },
                    .{ "i64", .keyword_i64 },
                    .{ "i128", .keyword_i128 },
                    .{ "u8", .keyword_u8 },
                    .{ "u16", .keyword_u16 },
                    .{ "u32", .keyword_u32 },
                    .{ "u64", .keyword_u64 },
                    .{ "u128", .keyword_u128 },
                    .{ "f16", .keyword_f16 },
                    .{ "f32", .keyword_f32 },
                    .{ "f64", .keyword_f64 },
                    .{ "usize", .keyword_usize },
                    .{ "isize", .keyword_isize },
                };
                break :blk std.StaticStringMap(Tag).initComptime(kvs);
            };
            return map.get(bytes);
        }
    };
};

// ── Tests ────────────────────────────────────────────────────────────

test "keyword lookup" {
    const Tag = Token.Tag;
    try std.testing.expectEqual(Tag.keyword_fn, Tag.keyword("fn").?);
    try std.testing.expectEqual(Tag.keyword_const, Tag.keyword("const").?);
    try std.testing.expectEqual(Tag.keyword_return, Tag.keyword("return").?);
    try std.testing.expectEqual(Tag.keyword_struct, Tag.keyword("struct").?);
    try std.testing.expectEqual(Tag.keyword_u32, Tag.keyword("u32").?);
    try std.testing.expectEqual(Tag.keyword_usize, Tag.keyword("usize").?);
    try std.testing.expectEqual(Tag.keyword_and, Tag.keyword("and").?);
    try std.testing.expect(Tag.keyword("foobar") == null);
    try std.testing.expect(Tag.keyword("") == null);
}

test "lexeme round-trip for operators" {
    const Tag = Token.Tag;
    try std.testing.expectEqualStrings("+", Tag.plus.lexeme().?);
    try std.testing.expectEqualStrings("==", Tag.equal_equal.lexeme().?);
    try std.testing.expectEqualStrings("!=", Tag.bang_equal.lexeme().?);
    try std.testing.expectEqualStrings("<<=", Tag.less_less_equal.lexeme().?);
    try std.testing.expectEqualStrings("...", Tag.dot_dot_dot.lexeme().?);
    try std.testing.expectEqualStrings("=>", Tag.fat_arrow.lexeme().?);
    try std.testing.expectEqualStrings("->", Tag.arrow.lexeme().?);
    try std.testing.expect(Tag.identifier.lexeme() == null);
}

test "Loc.slice" {
    const source = "fn main() void";
    const loc = Token.Loc{ .start = 3, .end = 7 };
    try std.testing.expectEqualStrings("main", loc.slice(source));
}
