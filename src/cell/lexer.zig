const std = @import("std");
const token = @import("token.zig");

pub const Lexer = struct {
    input: []const u8,
    index: usize = 0,

    pub fn init(input: []const u8) Lexer {
        return Lexer{ .input = input };
    }

    fn peek(self: *Lexer) ?u8 {
        if (self.index >= self.input.len) return null;
        return self.input[self.index];
    }

    fn advance(self: *Lexer) ?u8 {
        if (self.index >= self.input.len) return null;
        const ch = self.input[self.index];
        self.index += 1;
        return ch;
    }

    fn skipWhitespace(self: *Lexer) void {
        while (self.peek()) |c| {
            if (std.ascii.isWhitespace(c)) {
                _ = self.advance();
                continue;
            }
            break;
        }
    }

    pub fn nextToken(self: *Lexer) token.Token {
        self.skipWhitespace();
        const start = self.index;
        if (self.peek() == null) {
            return .{ .ttype = .eof, .lexeme = self.input[start..start] };
        }
        const ch = self.advance().?;
        switch (ch) {
            '+' => return .{ .ttype = .plus, .lexeme = self.input[start..self.index] },
            '-' => return .{ .ttype = .minus, .lexeme = self.input[start..self.index] },
            '*' => return .{ .ttype = .star, .lexeme = self.input[start..self.index] },
            '/' => return .{ .ttype = .slash, .lexeme = self.input[start..self.index] },
            '=' => if (self.peek() == '>') {
                _ = self.advance();
                return .{ .ttype = .arrow, .lexeme = self.input[start..self.index] };
            } else {
                return .{ .ttype = .assign, .lexeme = self.input[start..self.index] };
            },
            ';' => return .{ .ttype = .semicolon, .lexeme = self.input[start..self.index] },
            '(' => return .{ .ttype = .l_paren, .lexeme = self.input[start..self.index] },
            ')' => return .{ .ttype = .r_paren, .lexeme = self.input[start..self.index] },
            '{' => return .{ .ttype = .l_brace, .lexeme = self.input[start..self.index] },
            '}' => return .{ .ttype = .r_brace, .lexeme = self.input[start..self.index] },
            else => {
                if (std.ascii.isDigit(ch)) {
                    while (self.peek()) |p| if (std.ascii.isDigit(p)) _ = self.advance() else break;
                    return .{ .ttype = .number, .lexeme = self.input[start..self.index] };
                }
                if (std.ascii.isAlphabetic(ch)) {
                    while (self.peek()) |p| if (std.ascii.isAlphanumeric(p)) _ = self.advance() else break;
                    const lex = self.input[start..self.index];
                    if (std.mem.eql(u8, lex, "error_scope")) return .{ .ttype = .error_scope, .lexeme = lex };
                    if (std.mem.eql(u8, lex, "handle")) return .{ .ttype = .handle_kw, .lexeme = lex };
                    return .{ .ttype = .identifier, .lexeme = lex };
                }
                return .{ .ttype = .eof, .lexeme = self.input[start..start] };
            },
        }
    }
};
