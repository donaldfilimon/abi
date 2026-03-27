//! Documents stub — disabled at compile time.

const std = @import("std");
pub const types = @import("types.zig");

pub const html = struct {
    pub const DomNode = struct {
        tag: []const u8 = "",
        text_content: ?[]const u8 = null,
        children: std.ArrayListUnmanaged(DomNode) = .empty,

        pub fn deinit(_: *DomNode, _: std.mem.Allocator) void {}
    };

    pub const HtmlParser = struct {
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) HtmlParser {
            return .{ .allocator = allocator };
        }

        pub fn parse(_: *HtmlParser, _: []const u8) !DomNode {
            return .{};
        }
    };
};

pub const pdf = struct {
    pub const PdfError = error{
        InvalidPdfHeader,
        MissingXref,
        MalformedXref,
        NoTextContent,
        StreamDecompressFailure,
        OutOfMemory,
    };

    pub const PdfDocument = struct {
        pages: usize = 0,
        extracted_text: []const u8 = "",

        pub fn deinit(_: *PdfDocument, _: std.mem.Allocator) void {}
    };

    pub const PdfParser = struct {
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) PdfParser {
            return .{ .allocator = allocator };
        }

        pub fn parseBinaryStream(_: *PdfParser, _: []const u8) PdfError!PdfDocument {
            return error.InvalidPdfHeader;
        }
    };
};

pub const DocumentsError = types.DocumentsError;
pub const Error = types.Error;

pub const Context = struct {
    allocator: std.mem.Allocator,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator) Context {
        return .{ .allocator = allocator, .initialized = false };
    }

    pub fn deinit(self: *Context) void {
        _ = self;
    }
};

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
