const std = @import("std");
const token = @import("token.zig");
const lexer = @import("lexer.zig");
const ast = @import("ast.zig");

pub const Parser = struct {
    lexer: lexer.Lexer,
    current: token.Token,
    arena: std.heap.ArenaAllocator,

    pub fn init(allocator: std.mem.Allocator, input: []const u8) Parser {
        var arena = std.heap.ArenaAllocator.init(allocator);
        var lex = lexer.Lexer.init(input);
        const first = lex.nextToken();
        return Parser{ .lexer = lex, .current = first, .arena = arena };
    }

    fn advance(self: *Parser) void {
        self.current = self.lexer.nextToken();
    }

    fn expect(self: *Parser, t: token.TokenType) !void {
        if (self.current.ttype != t) return error.UnexpectedToken;
        self.advance();
    }

    pub fn parseProgram(self: *Parser) ![]ast.Node {
        var list = std.ArrayList(ast.Node).init(self.arena.allocator());
        while (self.current.ttype != .eof) {
            const stmt = try self.parseStmt();
            try list.append(stmt.*);
        }
        return list.toOwnedSlice();
    }

    fn parseStmt(self: *Parser) !*ast.Node {
        if (self.current.ttype == .identifier and std.mem.eql(u8, self.current.lexeme, "let")) {
            return try self.parseVarDecl();
        } else if (self.current.ttype == .identifier and std.mem.eql(u8, self.current.lexeme, "print")) {
            return try self.parsePrint();
        } else if (self.current.ttype == .error_scope) {
            return try self.parseErrorScope();
        } else {
            return error.InvalidStatement;
        }
    }

    fn parseVarDecl(self: *Parser) !*ast.Node {
        // consume 'let'
        self.advance();
        const name = self.current.lexeme;
        try self.expect(.identifier);
        try self.expect(.assign);
        const expr = try self.parseExpr();
        try self.expect(.semicolon);
        const node = self.arena.allocator().create(ast.Node) catch unreachable;
        node.* = ast.Node{ .var_decl = .{ .name = name, .value = expr } };
        return node;
    }

    fn parsePrint(self: *Parser) !*ast.Node {
        // consume 'print'
        self.advance();
        const expr = try self.parseExpr();
        try self.expect(.semicolon);
        const node = self.arena.allocator().create(ast.Node) catch unreachable;
        node.* = ast.Node{ .print_stmt = expr };
        return node;
    }

    fn parseErrorScope(self: *Parser) !*ast.Node {
        self.advance(); // consume error_scope
        try self.expect(.l_brace);
        var body = std.ArrayList(ast.Node).init(self.arena.allocator());
        while (self.current.ttype != .r_brace) {
            const stmt = try self.parseStmt();
            try body.append(stmt.*);
        }
        try self.expect(.r_brace);
        try self.expect(.handle_kw);
        try self.expect(.l_brace);
        var handlers = std.ArrayList(ast.Handler).init(self.arena.allocator());
        while (self.current.ttype != .r_brace) {
            const name = self.current.lexeme;
            try self.expect(.identifier);
            try self.expect(.arrow);
            try self.expect(.l_brace);
            var hbody = std.ArrayList(ast.Node).init(self.arena.allocator());
            while (self.current.ttype != .r_brace) {
                const s = try self.parseStmt();
                try hbody.append(s.*);
            }
            try self.expect(.r_brace);
            try handlers.append(.{ .name = name, .block = hbody.toOwnedSlice() });
        }
        try self.expect(.r_brace);
        const node = self.arena.allocator().create(ast.Node) catch unreachable;
        node.* = ast.Node{ .error_scope = .{ .body = body.toOwnedSlice(), .handlers = handlers.toOwnedSlice() } };
        return node;
    }

    fn parseExpr(self: *Parser) !*ast.Node {
        return self.parseAdd();
    }

    fn parseAdd(self: *Parser) !*ast.Node {
        var node = try self.parsePrimary();
        while (self.current.ttype == .plus or self.current.ttype == .minus) {
            const op = self.current.lexeme;
            self.advance();
            const right = try self.parsePrimary();
            const new = self.arena.allocator().create(ast.Node) catch unreachable;
            new.* = ast.Node{ .binary = .{ .op = op, .left = node, .right = right } };
            node = new;
        }
        return node;
    }

    fn parsePrimary(self: *Parser) !*ast.Node {
        if (self.current.ttype == .number) {
            const val = try std.fmt.parseInt(i64, self.current.lexeme, 10);
            const node = self.arena.allocator().create(ast.Node) catch unreachable;
            node.* = ast.Node{ .integer = val };
            self.advance();
            return node;
        } else if (self.current.ttype == .identifier) {
            const name = self.current.lexeme;
            const node = self.arena.allocator().create(ast.Node) catch unreachable;
            node.* = ast.Node{ .identifier = name };
            self.advance();
            return node;
        } else {
            return error.UnexpectedToken;
        }
    }
};

pub const ParseError = error{
    UnexpectedToken,
    InvalidStatement,
};
