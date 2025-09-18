//! Enterprise Features Demonstration
//!
//! This example showcases the comprehensive enterprise capabilities including:
//! - Advanced monitoring and observability
//! - Model registry and versioning
//! - Security features and audit logging
//! - Performance regression detection
//! - Production deployment patterns
//! - Scalability and reliability features

const std = @import("std");
const ai = @import("ai");
const monitoring = @import("monitoring");
const model_registry = @import("ai").model_registry;

/// Enterprise ML Platform
pub const EnterpriseMLPlatform = struct {
    allocator: std.mem.Allocator,
    model_registry: model_registry.ModelRegistry,
    performance_monitor: monitoring.PerformanceProfiler,
    health_checker: monitoring.HealthChecker,
    security_auditor: SecurityAuditor,

    pub fn init(allocator: std.mem.Allocator) !*EnterpriseMLPlatform {
        const self = try allocator.create(EnterpriseMLPlatform);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .model_registry = model_registry.ModelRegistry.init(allocator),
            .performance_monitor = try monitoring.PerformanceProfiler.init(allocator),
            .health_checker = try monitoring.HealthChecker.init(allocator),
            .security_auditor = SecurityAuditor.init(allocator),
        };

        return self;
    }

    pub fn deinit(self: *EnterpriseMLPlatform) void {
        self.model_registry.deinit();
        self.performance_monitor.deinit();
        self.health_checker.deinit();
        self.security_auditor.deinit();
        self.allocator.destroy(self);
    }

    /// Register a production model with comprehensive tracking
    pub fn registerProductionModel(
        self: *EnterpriseMLPlatform,
        model: *ai.NeuralNetwork,
        name: []const u8,
        version: []const u8,
        author: []const u8,
        description: []const u8,
    ) ![]const u8 {
        // Generate unique model ID
        const model_id = try self.generateModelId(name, version);

        // Create registry entry
        var entry = try model_registry.ModelEntry.init(self.allocator, model_id, name, version);
        errdefer entry.deinit(self.allocator);

        // Populate model metadata
        entry.architecture = try self.allocator.dupe(u8, "MLP");
        entry.author = try self.allocator.dupe(u8, author);
        entry.description = try self.allocator.dupe(u8, description);
        entry.num_parameters = model.getParameterCount();
        entry.model_size_bytes = model.getMemoryUsage();
        entry.input_shape = try self.allocator.dupe(usize, &[_]usize{model.getInputSize()});
        entry.output_shape = try self.allocator.dupe(usize, &[_]usize{model.getOutputSize()});

        // Add production tags
        try entry.tags.append(try self.allocator.dupe(u8, "production"));
        try entry.tags.append(try self.allocator.dupe(u8, "mlp"));
        try entry.categories.append(try self.allocator.dupe(u8, "classification"));

        // Set training metadata (would be populated from actual training)
        entry.training_config = .{
            .epochs = 100,
            .batch_size = 32,
            .learning_rate = 0.001,
            .optimizer = "Adam",
            .loss_function = "categorical_crossentropy",
            .dataset = "production_dataset_v1",
            .total_samples = 10000,
        };

        entry.training_metrics = .{
            .final_loss = 0.05,
            .final_accuracy = 0.95,
            .training_time_seconds = 3600,
            .best_epoch = 85,
            .convergence_epoch = 80,
        };

        // Register the model
        try self.model_registry.registerModel(&entry);

        // Log security event
        try self.security_auditor.logEvent(.{
            .event_type = .model_registration,
            .user = author,
            .resource = model_id,
            .action = "register",
            .success = true,
        });

        std.debug.print("Registered production model: {} v{} (ID: {})\n", .{ name, version, model_id });

        return model_id;
    }

    /// Perform comprehensive model evaluation
    pub fn evaluateModel(
        self: *EnterpriseMLPlatform,
        model_id: []const u8,
        test_data: []const []const f32,
        test_labels: []const []const f32,
    ) !model_registry.PerformanceMetrics {
        const model_entry = self.model_registry.getModel(model_id) orelse return error.ModelNotFound;

        // Start performance monitoring
        try self.performance_monitor.startSession("model_evaluation");

        const start_time = std.time.nanoTimestamp();

        // Evaluate model
        var total_loss: f32 = 0.0;
        var total_accuracy: f32 = 0.0;
        var total_precision: f32 = 0.0;
        var total_recall: f32 = 0.0;

        const output_size = model_entry.output_shape[0];

        for (test_data, test_labels) |input, label| {
            // Forward pass (simplified)
            const prediction = try self.allocator.alloc(f32, output_size);
            defer self.allocator.free(prediction);

            // Dummy prediction (would use actual model inference)
            for (prediction, 0..) |*pred, i| {
                pred.* = if (i == 0) 0.9 else 0.1; // Simplified
            }

            // Calculate metrics
            const loss = self.calculateLoss(prediction, label);
            const accuracy = self.calculateAccuracy(prediction, label);
            const precision = self.calculatePrecision(prediction, label);
            const recall = self.calculateRecall(prediction, label);

            total_loss += loss;
            total_accuracy += accuracy;
            total_precision += precision;
            total_recall += recall;
        }

        const end_time = std.time.nanoTimestamp();
        const latency_ms = @as(f32, @floatFromInt(end_time - start_time)) / 1_000_000.0;

        // Stop performance monitoring
        const perf_metrics = try self.performance_monitor.endSession();
        defer self.allocator.free(perf_metrics);

        const avg_loss = total_loss / @as(f32, @floatFromInt(test_data.len));
        const avg_accuracy = total_accuracy / @as(f32, @floatFromInt(test_data.len));
        const avg_precision = total_precision / @as(f32, @floatFromInt(test_data.len));
        const avg_recall = total_recall / @as(f32, @floatFromInt(test_data.len));
        const f1_score = 2.0 * avg_precision * avg_recall / (avg_precision + avg_recall);

        const metrics = model_registry.PerformanceMetrics{
            .timestamp = @as(u64, @intCast(std.time.nanoTimestamp())),
            .accuracy = avg_accuracy,
            .precision = avg_precision,
            .recall = avg_recall,
            .f1_score = f1_score,
            .latency_ms = latency_ms,
            .throughput_samples_per_sec = @as(f32, @floatFromInt(test_data.len)) / (latency_ms / 1000.0),
            .memory_usage_mb = @as(f32, @floatFromInt(perf_metrics[0].memory_usage_bytes)) / (1024 * 1024),
            .gpu_utilization = perf_metrics[0].gpu_utilization,
        };

        // Record metrics in registry
        try self.model_registry.recordMetrics(model_id, metrics);

        // Log performance event
        try self.security_auditor.logEvent(.{
            .event_type = .model_evaluation,
            .user = "system",
            .resource = model_id,
            .action = "evaluate",
            .success = true,
        });

        return metrics;
    }

    /// Deploy model to production with monitoring
    pub fn deployToProduction(self: *EnterpriseMLPlatform, model_id: []const u8) !void {
        // Health check before deployment
        const health_status = try self.health_checker.checkSystemHealth();
        if (health_status != .healthy) {
            std.debug.print("System health check failed, aborting deployment\n", .{});
            return error.SystemUnhealthy;
        }

        // Promote model
        try self.model_registry.promoteToProduction(model_id);

        // Log deployment event
        try self.security_auditor.logEvent(.{
            .event_type = .model_deployment,
            .user = "admin",
            .resource = model_id,
            .action = "deploy_production",
            .success = true,
        });

        std.debug.print("Successfully deployed model {} to production\n", .{model_id});
    }

    /// Comprehensive system monitoring
    pub fn runSystemMonitoring(self: *EnterpriseMLPlatform) !void {
        std.debug.print("=== System Monitoring Report ===\n", .{});

        // Performance metrics
        std.debug.print("Performance Metrics:\n", .{});
        const perf_stats = try self.performance_monitor.getSystemStats();
        std.debug.print("  CPU Usage: {d:.1}%\n", .{perf_stats.cpu_usage_percent});
        std.debug.print("  Memory Usage: {} MB\n", .{perf_stats.memory_usage_mb});
        std.debug.print("  GPU Usage: ", .{});
        if (perf_stats.gpu_usage_percent) |gpu| {
            std.debug.print("{d:.1}%\n", .{gpu});
        } else {
            std.debug.print("N/A\n", .{});
        }

        // Health check
        std.debug.print("\nHealth Status: {}\n", .{@tagName(try self.health_checker.checkSystemHealth())});

        // Model registry stats
        std.debug.print("\nModel Registry:\n", .{});
        var model_count: usize = 0;
        var it = self.model_registry.models.iterator();
        while (it.next()) |_| model_count += 1;
        std.debug.print("  Total Models: {}\n", .{model_count});

        var production_count: usize = 0;
        it = self.model_registry.models.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.*.deployment_status == .production) {
                production_count += 1;
            }
        }
        std.debug.print("  Production Models: {}\n", .{production_count});
    }

    fn generateModelId(self: *EnterpriseMLPlatform, name: []const u8, version: []const u8) ![]const u8 {
        const timestamp = @as(u64, @intCast(std.time.nanoTimestamp()));
        return std.fmt.allocPrint(self.allocator, "{s}-{s}-{d}", .{ name, version, timestamp });
    }

    fn calculateLoss(self: *EnterpriseMLPlatform, prediction: []const f32, target: []const f32) f32 {
        _ = self;
        var loss: f32 = 0.0;
        for (prediction, target) |pred, targ| {
            const diff = pred - targ;
            loss += diff * diff;
        }
        return loss / @as(f32, @floatFromInt(prediction.len));
    }

    fn calculateAccuracy(self: *EnterpriseMLPlatform, prediction: []const f32, target: []const f32) f32 {
        _ = self;
        // Simple accuracy calculation
        var pred_class: usize = 0;
        var true_class: usize = 0;
        var max_pred = prediction[0];
        var max_true = target[0];

        for (prediction, 0..) |pred, i| {
            if (pred > max_pred) {
                max_pred = pred;
                pred_class = i;
            }
            if (target[i] > max_true) {
                max_true = target[i];
                true_class = i;
            }
        }

        return if (pred_class == true_class) 1.0 else 0.0;
    }

    fn calculatePrecision(self: *EnterpriseMLPlatform, prediction: []const f32, target: []const f32) f32 {
        _ = self;
        _ = prediction;
        _ = target;
        // Simplified precision calculation
        return 0.85;
    }

    fn calculateRecall(self: *EnterpriseMLPlatform, prediction: []const f32, target: []const f32) f32 {
        _ = self;
        _ = prediction;
        _ = target;
        // Simplified recall calculation
        return 0.88;
    }
};

/// Security auditor for compliance and audit logging
pub const SecurityAuditor = struct {
    allocator: std.mem.Allocator,
    audit_log: std.ArrayList(AuditEvent),

    pub const AuditEvent = struct {
        timestamp: u64,
        event_type: EventType,
        user: []const u8,
        resource: []const u8,
        action: []const u8,
        success: bool,
        details: ?[]const u8,

        pub const EventType = enum {
            model_registration,
            model_evaluation,
            model_deployment,
            model_access,
            security_violation,
            system_access,
        };
    };

    pub fn init(allocator: std.mem.Allocator) SecurityAuditor {
        return SecurityAuditor{
            .allocator = allocator,
            .audit_log = std.ArrayList(AuditEvent).init(allocator),
        };
    }

    pub fn deinit(self: *SecurityAuditor) void {
        for (self.audit_log.items) |*event| {
            self.allocator.free(event.user);
            self.allocator.free(event.resource);
            self.allocator.free(event.action);
            if (event.details) |details| {
                self.allocator.free(details);
            }
        }
        self.audit_log.deinit();
    }

    pub fn logEvent(self: *SecurityAuditor, event: AuditEvent) !void {
        var logged_event = AuditEvent{
            .timestamp = event.timestamp,
            .event_type = event.event_type,
            .user = try self.allocator.dupe(u8, event.user),
            .resource = try self.allocator.dupe(u8, event.resource),
            .action = try self.allocator.dupe(u8, event.action),
            .success = event.success,
            .details = if (event.details) |d| try self.allocator.dupe(u8, d) else null,
        };

        try self.audit_log.append(logged_event);

        // Log to console for demonstration
        std.debug.print("[AUDIT] {} - User: {s}, Action: {s}, Resource: {s}, Success: {}\n", .{ @tagName(event.event_type), event.user, event.action, event.resource, event.success });
    }

    pub fn getAuditLog(self: *SecurityAuditor) []AuditEvent {
        return self.audit_log.items;
    }

    pub fn generateSecurityReport(self: *SecurityAuditor) !void {
        std.debug.print("=== Security Audit Report ===\n", .{});
        std.debug.print("Total Events: {}\n", .{self.audit_log.items.len});

        var event_counts = std.StringHashMap(usize).init(self.allocator);
        defer event_counts.deinit();

        for (self.audit_log.items) |event| {
            const event_name = @tagName(event.event_type);
            const count = event_counts.get(event_name) orelse 0;
            try event_counts.put(event_name, count + 1);
        }

        var it = event_counts.iterator();
        while (it.next()) |entry| {
            std.debug.print("  {s}: {}\n", .{ entry.key_ptr.*, entry.value_ptr.* });
        }

        std.debug.print("\nRecent Events:\n", .{});
        const recent_count = @min(5, self.audit_log.items.len);
        for (self.audit_log.items[self.audit_log.items.len - recent_count ..]) |event| {
            std.debug.print("  {} - {s} by {s}\n", .{ @tagName(event.event_type), event.action, event.user });
        }
    }
};

/// Main demonstration function
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator;

    std.debug.print("=== Enterprise ML Platform Demo ===\n", .{});

    // Initialize enterprise platform
    var platform = try EnterpriseMLPlatform.init(allocator);
    defer platform.deinit();

    // Create and register a production model
    std.debug.print("\n=== Model Registration ===\n", .{});

    var model = try ai.createMLP(allocator, &[_]usize{ 10, 64, 32, 2 }, &[_]ai.Activation{ .relu, .relu, .softmax });
    defer model.deinit();

    const model_id = try platform.registerProductionModel(
        model,
        "enterprise-classifier",
        "1.0.0",
        "ml-engineer@company.com",
        "Production classifier for enterprise use case",
    );
    defer allocator.free(model_id);

    // Model evaluation
    std.debug.print("\n=== Model Evaluation ===\n", .{});

    // Create dummy test data
    const num_test_samples = 100;
    var test_data = try allocator.alloc([]const f32, num_test_samples);
    defer {
        for (test_data) |sample| {
            allocator.free(sample);
        }
        allocator.free(test_data);
    }

    var test_labels = try allocator.alloc([]const f32, num_test_samples);
    defer {
        for (test_labels) |label| {
            allocator.free(label);
        }
        allocator.free(test_labels);
    }

    for (0..num_test_samples) |i| {
        const input = try allocator.alloc(f32, 10);
        const label = try allocator.alloc(f32, 2);

        // Generate dummy data
        for (0..10) |j| {
            input[j] = std.math.sin(@as(f32, @floatFromInt(i + j)) * 0.1);
        }
        label[0] = if (i % 2 == 0) 1.0 else 0.0;
        label[1] = if (i % 2 == 0) 0.0 else 1.0;

        test_data[i] = input;
        test_labels[i] = label;
    }

    const eval_metrics = try platform.evaluateModel(model_id, test_data, test_labels);
    std.debug.print("Evaluation Results:\n", .{});
    std.debug.print("  Accuracy: {d:.3}\n", .{eval_metrics.accuracy});
    std.debug.print("  Precision: {d:.3}\n", .{eval_metrics.precision});
    std.debug.print("  Recall: {d:.3}\n", .{eval_metrics.recall});
    std.debug.print("  F1 Score: {d:.3}\n", .{eval_metrics.f1_score});
    std.debug.print("  Latency: {d:.2} ms\n", .{eval_metrics.latency_ms});
    std.debug.print("  Throughput: {d:.1} samples/sec\n", .{eval_metrics.throughput_samples_per_sec});

    // Deploy to production
    std.debug.print("\n=== Production Deployment ===\n", .{});
    try platform.deployToProduction(model_id);

    // System monitoring
    std.debug.print("\n=== System Monitoring ===\n", .{});
    try platform.runSystemMonitoring();

    // Security audit report
    std.debug.print("\n=== Security Audit Report ===\n", .{});
    try platform.security_auditor.generateSecurityReport();

    // Model registry operations
    std.debug.print("\n=== Model Registry Operations ===\n", .{});

    var registry_cli = model_registry.RegistryCLI.init(&platform.model_registry);
    try registry_cli.listModels();

    std.debug.print("\nModel Details:\n", .{});
    try registry_cli.showModelDetails(model_id);

    std.debug.print("\n=== Demo Complete ===\n", .{});
}
