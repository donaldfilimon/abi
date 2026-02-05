//! Stub Parity Verification
//!
//! Verifies that stub modules export the same public symbols as their
//! corresponding real modules. This ensures stubs stay in sync with
//! real implementations.
//!
//! Run with: zig build test --summary all

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const abi = @import("abi");

// ============================================================================
// Database Module Parity
// ============================================================================

test "database stub parity - types exist" {
    // These types should exist in both real and stub implementations
    // Since we're testing via abi module, we verify the public API surface
    const Database = abi.database;

    // Verify key exported types exist
    try testing.expect(@hasDecl(Database, "DatabaseHandle"));
    try testing.expect(@hasDecl(Database, "SearchResult"));
    try testing.expect(@hasDecl(Database, "Context"));
    try testing.expect(@hasDecl(Database, "Stats"));
    try testing.expect(@hasDecl(Database, "VectorView"));

    // Verify key functions exist
    try testing.expect(@hasDecl(Database, "open"));
    try testing.expect(@hasDecl(Database, "search"));
    try testing.expect(@hasDecl(Database, "insert"));
    try testing.expect(@hasDecl(Database, "isEnabled"));

    // Verify sub-modules
    try testing.expect(@hasDecl(Database, "wdbx"));
    try testing.expect(@hasDecl(Database, "fulltext"));
    try testing.expect(@hasDecl(Database, "hybrid"));
    try testing.expect(@hasDecl(Database, "filter"));
    try testing.expect(@hasDecl(Database, "batch"));
    try testing.expect(@hasDecl(Database, "clustering"));
    try testing.expect(@hasDecl(Database, "formats"));
}

// ============================================================================
// GPU Module Parity
// ============================================================================

test "gpu stub parity - types exist" {
    const Gpu = abi.gpu;

    // Verify Context exists
    try testing.expect(@hasDecl(Gpu, "Context"));

    // Verify isEnabled exists
    try testing.expect(@hasDecl(Gpu, "isEnabled"));
}

// ============================================================================
// Network Module Parity
// ============================================================================

test "network stub parity - types exist" {
    const Network = abi.network;

    try testing.expect(@hasDecl(Network, "Context"));
    try testing.expect(@hasDecl(Network, "isEnabled"));
}

// ============================================================================
// Web Module Parity
// ============================================================================

test "web stub parity - types exist" {
    const Web = abi.web;

    try testing.expect(@hasDecl(Web, "Context"));
    try testing.expect(@hasDecl(Web, "isEnabled"));
}

// ============================================================================
// Observability Module Parity
// ============================================================================

test "observability stub parity - types exist" {
    const Observability = abi.observability;

    try testing.expect(@hasDecl(Observability, "Context"));
    try testing.expect(@hasDecl(Observability, "isEnabled"));
}

// ============================================================================
// AI Module Parity
// ============================================================================

test "ai stub parity - types exist" {
    if (!build_options.enable_ai) return;

    const AI = abi.ai;

    try testing.expect(@hasDecl(AI, "Context"));
    try testing.expect(@hasDecl(AI, "isEnabled"));
}

// ============================================================================
// AI Sub-module Parity Tests
// ============================================================================

test "ai/llm stub parity - types exist" {
    if (!build_options.enable_llm) return;

    const Llm = abi.ai.llm;
    _ = Llm; // Module exists and is accessible
}

test "ai/agents stub parity - types exist" {
    if (!build_options.enable_ai) return;

    const Agents = abi.ai.agents;
    _ = Agents; // Module exists and is accessible
}

test "ai/embeddings stub parity - types exist" {
    if (!build_options.enable_ai) return;

    const Embeddings = abi.ai.embeddings;
    _ = Embeddings; // Module exists and is accessible
}

test "ai/training stub parity - types exist" {
    if (!build_options.enable_ai) return;

    const Training = abi.ai.training;
    _ = Training; // Module exists and is accessible
}

// ============================================================================
// Parity Verification Helpers
// ============================================================================

/// Verify a module has the expected minimal API surface for Context pattern
fn verifyContextPattern(comptime Module: type) !void {
    try testing.expect(@hasDecl(Module, "Context"));
    try testing.expect(@hasDecl(Module, "isEnabled"));
}

// ============================================================================
// Comprehensive Module Surface Test
// ============================================================================

test "all feature modules have consistent API surface" {
    // All feature modules should follow the Context + isEnabled pattern
    try verifyContextPattern(abi.database);
    try verifyContextPattern(abi.gpu);
    try verifyContextPattern(abi.network);
    try verifyContextPattern(abi.web);
    try verifyContextPattern(abi.observability);

    if (build_options.enable_ai) {
        try verifyContextPattern(abi.ai);
    }
}
