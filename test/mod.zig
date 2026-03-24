//! Integration test root.
//!
//! This module is the entry point for integration tests that use the abi package.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

// Integration test modules
const persona_pipeline = @import("integration/persona_pipeline_test.zig");
const database_tests = @import("integration/database_test.zig");
const database_surface_tests = @import("integration/database_surface_test.zig");
const feature_boundary_tests = @import("integration/feature_boundary_test.zig");
const inference_tests = @import("integration/inference_test.zig");
const security_tests = @import("integration/security_test.zig");
const gpu_tests = @import("integration/gpu_test.zig");
const web_tests = @import("integration/web_test.zig");
const network_tests = @import("integration/network_test.zig");
const auth_tests = @import("integration/auth_test.zig");
const gateway_tests = @import("integration/gateway_test.zig");
const analytics_tests = @import("integration/analytics_test.zig");
const desktop_tests = @import("integration/desktop_test.zig");
const mobile_tests = @import("integration/mobile_test.zig");
const messaging_tests = @import("integration/messaging_test.zig");
const cloud_tests = @import("integration/cloud_test.zig");
const compute_tests = @import("integration/compute_test.zig");
const storage_tests = @import("integration/storage_test.zig");
const search_tests = @import("integration/search_test.zig");
const cache_tests = @import("integration/cache_test.zig");
const connector_tests = @import("integration/connector_test.zig");
const connector_errors_tests = @import("integration/connector_errors_test.zig");
const mcp_tests = @import("integration/mcp_test.zig");
const observability_tests = @import("integration/observability_test.zig");
const cli_tests = @import("integration/cli_test.zig");
const cli_e2e_tests = @import("integration/cli_e2e_test.zig");
const tui_tests = @import("integration/tui_test.zig");
const documents_tests = @import("integration/documents_test.zig");
const lsp_tests = @import("integration/lsp_test.zig");
const lsp_protocol_tests = @import("integration/lsp_protocol_test.zig");
const pages_tests = @import("integration/pages_test.zig");
const training_tests = @import("integration/training_test.zig");
const benchmarks_tests = @import("integration/benchmarks_test.zig");
const ha_tests = @import("integration/ha_test.zig");
const acp_tests = @import("integration/acp_test.zig");
const plugin_registry_tests = @import("integration/plugin_registry_test.zig");
const database_core_tests = @import("integration/database_core_test.zig");

test {
    std.testing.refAllDecls(@This());
}
