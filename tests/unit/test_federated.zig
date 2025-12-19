// Tests for Federated Learning Module

const std = @import("std");
const ai = @import("abi").ai;

/// Test basic federated coordinator initialization and client management
test "federated coordinator basic operations" {
    const testing = std.testing;
    var coordinator = try ai.federated.FederatedCoordinator.init(testing.allocator, 42);
    defer coordinator.deinit();

    // Test initial state
    try testing.expectEqual(@as(usize, 0), coordinator.clients.items.len);
    try testing.expectEqual(@as(usize, 0), coordinator.rounds);

    // Add a client
    const client = ai.federated.FederatedCoordinator.Client{
        .id = 1,
        .model = 100,
        .data_size = 1000,
    };

    try coordinator.addClient(client);
    try testing.expectEqual(@as(usize, 1), coordinator.clients.items.len);
    try testing.expectEqual(@as(usize, 1), coordinator.clients.items[0].id);
}

/// Test federated model aggregation
test "federated model aggregation" {
    const testing = std.testing;
    var coordinator = try ai.federated.FederatedCoordinator.init(testing.allocator, 42);
    defer coordinator.deinit();

    // Add multiple clients
    const clients = [_]ai.federated.FederatedCoordinator.Client{
        .{ .id = 1, .model = 100, .data_size = 500 },
        .{ .id = 2, .model = 200, .data_size = 800 },
        .{ .id = 3, .model = 300, .data_size = 300 },
    };

    for (clients) |client| {
        try coordinator.addClient(client);
    }

    try testing.expectEqual(@as(usize, 3), coordinator.clients.items.len);

    // Perform aggregation (currently a no-op, but should not crash)
    try coordinator.aggregateModels();
    try testing.expectEqual(@as(usize, 1), coordinator.rounds);
}

/// Test federated coordinator with no clients
test "federated coordinator empty state" {
    const testing = std.testing;
    var coordinator = try ai.federated.FederatedCoordinator.init(testing.allocator, 42);
    defer coordinator.deinit();

    // Aggregation with no clients should succeed
    try coordinator.aggregateModels();
    try testing.expectEqual(@as(usize, 1), coordinator.rounds);
}

/// Test federated coordinator memory management
test "federated coordinator memory management" {
    const testing = std.testing;
    var coordinator = try ai.federated.FederatedCoordinator.init(testing.allocator, 1000);
    defer coordinator.deinit();

    // Add many clients to test memory handling
    for (0..10) |i| {
        const client = ai.federated.FederatedCoordinator.Client{
            .id = i,
            .model = i * 100,
            .data_size = 100 + i * 50,
        };
        try coordinator.addClient(client);
    }

    try testing.expectEqual(@as(usize, 10), coordinator.clients.items.len);

    // Verify client data integrity
    for (0..10) |i| {
        try testing.expectEqual(i, coordinator.clients.items[i].id);
        try testing.expectEqual(i * 100, coordinator.clients.items[i].model);
        try testing.expectEqual(100 + i * 50, coordinator.clients.items[i].data_size);
    }
}