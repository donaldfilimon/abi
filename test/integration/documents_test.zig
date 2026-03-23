//! Integration Tests: Documents Feature
//!
//! Tests the documents module exports, lifecycle queries, error types,
//! context management, and sub-module accessibility through the public
//! `abi.documents` surface.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const documents = abi.documents;

// ============================================================================
// Feature gate
// ============================================================================

test "documents: isEnabled reflects feature flag" {
    if (build_options.feat_documents) {
        try std.testing.expect(documents.isEnabled());
    } else {
        try std.testing.expect(!documents.isEnabled());
    }
}

test "documents: isInitialized reflects feature flag" {
    if (build_options.feat_documents) {
        try std.testing.expect(documents.isInitialized());
    } else {
        try std.testing.expect(!documents.isInitialized());
    }
}

// ============================================================================
// Types
// ============================================================================

test "documents: DocumentsError type is accessible" {
    const E = documents.DocumentsError;
    const err: E = error.ParseFailed;
    try std.testing.expect(err == error.ParseFailed);
}

test "documents: Error alias matches DocumentsError" {
    try std.testing.expect(documents.Error == documents.DocumentsError);
}

test "documents: DocumentsError includes expected variants" {
    const e1: documents.DocumentsError = error.UnsupportedFormat;
    const e2: documents.DocumentsError = error.InvalidInput;
    const e3: documents.DocumentsError = error.OutOfMemory;
    try std.testing.expect(e1 == error.UnsupportedFormat);
    try std.testing.expect(e2 == error.InvalidInput);
    try std.testing.expect(e3 == error.OutOfMemory);
}

// ============================================================================
// Context lifecycle
// ============================================================================

test "documents: Context init and deinit" {
    var ctx = documents.Context.init(std.testing.allocator);
    defer ctx.deinit();

    if (build_options.feat_documents) {
        try std.testing.expect(ctx.initialized);
    } else {
        try std.testing.expect(!ctx.initialized);
    }
}

test "documents: Context stores allocator" {
    var ctx = documents.Context.init(std.testing.allocator);
    defer ctx.deinit();
    _ = ctx.allocator;
}

// ============================================================================
// Sub-modules
// ============================================================================

test "documents: html sub-module is accessible" {
    const html_ns = documents.html;
    _ = html_ns;
}

test "documents: pdf sub-module is accessible" {
    const pdf_ns = documents.pdf;
    _ = pdf_ns;
}

test "documents: types sub-module is accessible" {
    const T = documents.types;
    const E = T.DocumentsError;
    const err: E = error.ParseFailed;
    try std.testing.expect(err == error.ParseFailed);
}

test {
    std.testing.refAllDecls(@This());
}
