const std = @import("std");

pub const TokenType = enum {
    identifier,
    number,
    plus,
    minus,
    star,
    slash,
    assign,
    semicolon,
    l_paren,
    r_paren,
    l_brace,
    r_brace,
    error_scope,
    handle_kw,
    arrow,
    eof,
};

pub const Token = struct {
    ttype: TokenType,
    lexeme: []const u8,
};
