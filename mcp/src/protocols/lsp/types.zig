//! Minimal LSP types used by the built-in ZLS client.

const std = @import("std");

pub const Position = struct {
    line: u32,
    character: u32,
};

pub const Range = struct {
    start: Position,
    end: Position,
};

pub const TextDocumentIdentifier = struct {
    uri: []const u8,
};

pub const TextDocumentItem = struct {
    uri: []const u8,
    languageId: []const u8 = "zig",
    version: i32 = 1,
    text: []const u8,
};

pub const DidOpenTextDocumentParams = struct {
    textDocument: TextDocumentItem,
};

pub const TextDocumentPositionParams = struct {
    textDocument: TextDocumentIdentifier,
    position: Position,
};

pub const ReferencesContext = struct {
    includeDeclaration: bool = true,
};

pub const ReferencesParams = struct {
    textDocument: TextDocumentIdentifier,
    position: Position,
    context: ReferencesContext,
};

pub const RenameParams = struct {
    textDocument: TextDocumentIdentifier,
    position: Position,
    newName: []const u8,
};

pub const FormattingOptions = struct {
    tabSize: u32 = 4,
    insertSpaces: bool = true,
};

pub const DocumentFormattingParams = struct {
    textDocument: TextDocumentIdentifier,
    options: FormattingOptions,
};

test {
    std.testing.refAllDecls(@This());
}
