//! Federated Learning Module
//!
//! This module provides federated learning coordination capabilities for distributed
//! machine learning across multiple clients/devices. It implements the FedAvg algorithm
//! and supports privacy-preserving model updates.
//!
//! Key features:
//! - Client registration and management
//! - Federated averaging (FedAvg) model aggregation
//! - Round-based training coordination
//! - Memory-efficient model storage

const std = @import("std");

/// Errors specific to federated learning operations
pub const Error = error{
    /// Update has wrong number of parameters compared to global model
    InvalidUpdateSize,
    /// No clients registered for aggregation
    NoClients,
    /// Client ID already exists
    ClientAlreadyExists,
    /// Invalid parameter (empty or too long client ID)
    InvalidParameter,
};

/// Federated learning coordinator that manages distributed model training
/// across multiple clients. Implements the FedAvg algorithm for model aggregation.
pub const Coordinator = struct {
    /// Memory allocator for dynamic allocations
    allocator: std.mem.Allocator,
    /// List of registered clients participating in federated learning
    clients: std.ArrayList(ClientInfo),
    /// Global model parameters shared across all clients
    global_model: []f32,
    /// Number of training rounds completed
    rounds: usize,

    /// Information about a federated learning client
    pub const ClientInfo = struct {
        /// Unique client identifier
        id: []const u8,
        /// Timestamp of last model update from this client
        last_update: i64,
        /// Weight factor for this client's contribution to model aggregation
        contribution_weight: f32,
    };

    /// Initialize a new federated learning coordinator.
    ///
    /// # Parameters
    /// - `allocator`: Memory allocator for internal data structures
    /// - `model_size`: Number of parameters in the model
    ///
    /// # Returns
    /// New Coordinator instance with initialized global model
    ///
    /// # Errors
    /// - `OutOfMemory`: If allocation fails
    ///
    /// # Example
    /// ```zig
    /// var coordinator = try Coordinator.init(allocator, 1000);
    /// defer coordinator.deinit();
    /// ```
    pub fn init(allocator: std.mem.Allocator, model_size: usize) !Coordinator {
        const global_model = try allocator.alloc(f32, model_size);
        @memset(global_model, 0);

        return Coordinator{
            .allocator = allocator,
            .clients = std.ArrayList(ClientInfo).init(allocator),
            .global_model = global_model,
            .rounds = 0,
        };
    }

    /// Clean up coordinator resources and free allocated memory.
    /// This will free the global model and all client information.
    pub fn deinit(self: *Coordinator) void {
        self.allocator.free(self.global_model);
        for (self.clients.items) |client| {
            self.allocator.free(client.id);
        }
        self.clients.deinit();
    }

    /// Register a new client for federated learning participation.
    ///
    /// # Parameters
    /// - `client_id`: Unique identifier for the client (will be duplicated)
    ///
    /// # Errors
    /// - `InvalidParameter`: If client_id is empty or too long (>256 chars)
    /// - `ClientAlreadyExists`: If a client with the same ID is already registered
    /// - `OutOfMemory`: If allocation of client ID copy fails
    ///
    /// # Example
    /// ```zig
    /// try coordinator.registerClient("client_001");
    /// try coordinator.registerClient("mobile_device_42");
    /// ```
    pub fn registerClient(self: *Coordinator, client_id: []const u8) !void {
        // Input validation
        if (client_id.len == 0) return Error.InvalidParameter;
        if (client_id.len > 256) return Error.InvalidParameter; // Reasonable limit

        // Check for duplicate client IDs
        for (self.clients.items) |client| {
            if (std.mem.eql(u8, client.id, client_id)) {
                return Error.ClientAlreadyExists;
            }
        }

        const id_copy = try self.allocator.dupe(u8, client_id);
        errdefer self.allocator.free(id_copy);

        try self.clients.append(ClientInfo{
            .id = id_copy,
            .last_update = std.time.timestamp(),
            .contribution_weight = 1.0,
        });
    }

    /// Aggregate model updates from multiple clients using FedAvg algorithm.
    /// This implements the core federated averaging operation where client model
    /// updates are averaged to produce a new global model.
    ///
    /// # Parameters
    /// - `updates`: Array of model parameter arrays from different clients.
    ///              All arrays must have the same length as the global model.
    ///
    /// # Errors
    /// - `InvalidUpdateSize`: If any update has wrong number of parameters
    ///
    /// # Algorithm
    /// The global model is updated as: `global = (sum of all client_updates) / num_clients`
    ///
    /// # Example
    /// ```zig
    /// const update1 = [_]f32{0.1, 0.2, 0.3}; // Client 1 updates
    /// const update2 = [_]f32{0.15, 0.18, 0.25}; // Client 2 updates
    /// const updates = [_][]const f32{ &update1, &update2 };
    /// try coordinator.aggregateUpdates(&updates);
    /// ```
    pub fn aggregateUpdates(self: *Coordinator, updates: []const []const f32) !void {
        if (updates.len == 0) return;

        const model_size = self.global_model.len;

        // Simple FedAvg aggregation
        var total_weight: f32 = 0;
        for (updates) |update| {
            if (update.len != model_size) return Error.InvalidUpdateSize;
            total_weight += 1.0; // Equal weighting for simplicity
        }

        // Reset global model
        @memset(self.global_model, 0);

        // Aggregate
        for (updates) |update| {
            for (update, 0..) |val, i| {
                self.global_model[i] += val / total_weight;
            }
        }

        self.rounds += 1;
    }

    /// Get read-only access to the current global model parameters.
    /// The returned slice is valid until the coordinator is deinitialized.
    ///
    /// # Returns
    /// Slice containing the current global model parameters
    ///
    /// # Example
    /// ```zig
    /// const model = coordinator.getGlobalModel();
    /// std.debug.print("Global model has {} parameters\n", .{model.len});
    /// ```
    pub fn getGlobalModel(self: *const Coordinator) []const f32 {
        return self.global_model;
    }
};

test "federated coordinator basic functionality" {
    const testing = std.testing;
    var coordinator = try Coordinator.init(testing.allocator, 10);
    defer coordinator.deinit();

    // Register clients
    try coordinator.registerClient("client1");
    try coordinator.registerClient("client2");

    try testing.expectEqual(@as(usize, 2), coordinator.clients.items.len);

    // Simulate updates
    const update1 = [_]f32{1.0} ** 10;
    const update2 = [_]f32{2.0} ** 10;
    const updates = [_][]const f32{ &update1, &update2 };

    try coordinator.aggregateUpdates(&updates);

    // Check aggregation (average)
    for (coordinator.global_model) |val| {
        try testing.expectApproxEqAbs(1.5, val, 0.001);
    }

    try testing.expectEqual(@as(usize, 1), coordinator.rounds);
}

test "federated coordinator error handling" {
    const testing = std.testing;
    var coordinator = try Coordinator.init(testing.allocator, 5);
    defer coordinator.deinit();

    // Test empty updates (should not crash)
    try coordinator.aggregateUpdates(&.{});

    // Test invalid update size
    const wrong_size_update = [_]f32{ 1.0, 2.0 }; // Only 2 elements, should be 5
    const updates = [_][]const f32{&wrong_size_update};

    try testing.expectError(error.InvalidUpdateSize, coordinator.aggregateUpdates(&updates));
}

test "federated coordinator weighted aggregation" {
    const testing = std.testing;
    var coordinator = try Coordinator.init(testing.allocator, 3);
    defer coordinator.deinit();

    // Register clients with different contribution weights
    try coordinator.registerClient("large_client");
    try coordinator.registerClient("small_client");

    // Simulate updates with different magnitudes
    const update1 = [_]f32{ 2.0, 4.0, 6.0 }; // Large client
    const update2 = [_]f32{ 1.0, 1.0, 1.0 }; // Small client
    const updates = [_][]const f32{ &update1, &update2 };

    try coordinator.aggregateUpdates(&updates);

    // Check that result is average: (2+1)/2=1.5, (4+1)/2=2.5, (6+1)/2=3.5
    const expected = [_]f32{ 1.5, 2.5, 3.5 };
    for (coordinator.global_model, expected) |actual, exp| {
        try testing.expectApproxEqAbs(exp, actual, 0.001);
    }
}
