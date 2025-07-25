const std = @import("std");

pub const NodeTag = enum {
    integer,
    identifier,
    binary,
    var_decl,
    print_stmt,
    block,
    error_scope,
};

pub const Node = union(NodeTag) {
    integer: i64,
    identifier: []const u8,
    binary: Binary,
    var_decl: VarDecl,
    print_stmt: Expr,
    block: []Node,
    error_scope: ErrorScope,
};

pub const Binary = struct {
    op: []const u8,
    left: *Node,
    right: *Node,
};

pub const VarDecl = struct {
    name: []const u8,
    value: *Node,
};

pub const Expr = *Node;

pub const ErrorScope = struct {
    body: []Node,
    handlers: []Handler,
};

pub const Handler = struct {
    name: []const u8,
    block: []Node,
};
