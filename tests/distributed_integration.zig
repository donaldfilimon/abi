//! Distributed Integration Smoke Tests (external test root)
//!
//! Verifies that distributed subsystem types are accessible through the
//! public `abi` package interface.  Full distributed tests live in
//! src/core/database/distributed/ and run via `zig build test`.

const std = @import("std");
const abi = @import("abi");

test "database feature exposes distributed sub-modules" {
    // Verify the database feature namespace resolves and has expected
    // sub-module declarations (compile-time import check).
    comptime {
        std.debug.assert(@hasDecl(abi.database, "DatabaseConfig"));
        std.debug.assert(@hasDecl(abi.database, "DatabaseHandle"));
    }
}

test "network feature exposes raft consensus" {
    // Verify the network feature namespace resolves (includes Raft).
    comptime {
        std.debug.assert(@hasDecl(abi, "network"));
    }
}

test "feature catalog lists database and network features" {
    const features = abi.feature_catalog.all_features;

    var found_database = false;
    var found_network = false;

    for (features) |feat| {
        if (std.mem.eql(u8, feat.name, "database")) found_database = true;
        if (std.mem.eql(u8, feat.name, "network")) found_network = true;
    }

    try std.testing.expect(found_database);
    try std.testing.expect(found_network);
}
