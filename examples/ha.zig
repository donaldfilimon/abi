//! High Availability Example
//!
//! Demonstrates the HA module for:
//! - Multi-region replication
//! - Automated backup orchestration
//! - Point-in-time recovery (PITR)
//! - Health monitoring and automatic failover

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== ABI High Availability Example ===\n\n", .{});

    // Initialize framework
    var framework = abi.initWithConfig(allocator, .{
        .database = .{},
    }) catch |err| {
        std.debug.print("Framework initialization failed: {}\n", .{err});
        return err;
    };
    defer framework.deinit();

    // === HA Manager Setup ===
    std.debug.print("--- HA Manager Setup ---\n", .{});

    var ha_manager = abi.ha.HaManager.init(allocator, .{
        .replication_factor = 3,
        .backup_interval_hours = 6,
        .enable_pitr = true,
        .pitr_retention_hours = 168, // 7 days
        .auto_failover = true,
        .health_check_interval_sec = 30,
        .on_event = logHaEvent,
    });
    defer ha_manager.deinit();

    std.debug.print("HA Manager configured:\n", .{});
    std.debug.print("  Replication factor: {d}\n", .{ha_manager.config.replication_factor});
    std.debug.print("  Backup interval: {d} hours\n", .{ha_manager.config.backup_interval_hours});
    std.debug.print("  PITR enabled: {}\n", .{ha_manager.config.enable_pitr});
    std.debug.print("  Auto-failover: {}\n", .{ha_manager.config.auto_failover});

    // === Start HA Services ===
    std.debug.print("\n--- Starting HA Services ---\n", .{});

    ha_manager.start() catch |err| {
        std.debug.print("Failed to start HA services: {}\n", .{err});
        return err;
    };

    const status = ha_manager.getStatus();
    std.debug.print("HA Status: {}\n", .{status});

    // === PITR Demo ===
    std.debug.print("\n--- Point-in-Time Recovery ---\n", .{});

    var pitr = abi.ha.PitrManager.init(allocator, .{
        .retention_hours = 24,
        .checkpoint_interval_sec = 60,
    });
    defer pitr.deinit();

    // Simulate capturing operations
    pitr.captureOperation(.insert, "user:1", "Alice", null) catch |err| {
        std.debug.print("Failed to capture operation: {}\n", .{err});
    };
    pitr.captureOperation(.update, "user:1", "Alice Smith", "Alice") catch |err| {
        std.debug.print("Failed to capture operation: {}\n", .{err});
    };
    pitr.captureOperation(.insert, "user:2", "Bob", null) catch |err| {
        std.debug.print("Failed to capture operation: {}\n", .{err});
    };

    // Create checkpoint
    const seq = pitr.createCheckpoint() catch |err| {
        std.debug.print("Failed to create checkpoint: {}\n", .{err});
        return err;
    };
    std.debug.print("Checkpoint created: sequence={d}\n", .{seq});

    // List recovery points
    const points = pitr.getRecoveryPoints();
    std.debug.print("Available recovery points: {d}\n", .{points.len});
    for (points) |point| {
        std.debug.print("  Seq {d}: {d} operations, {d} bytes\n", .{
            point.sequence,
            point.operation_count,
            point.size_bytes,
        });
    }

    // === Backup Demo ===
    std.debug.print("\n--- Backup Orchestration ---\n", .{});

    var backup = abi.ha.BackupOrchestrator.init(allocator, .{
        .interval_hours = 6,
    });
    defer backup.deinit();

    const backup_state = backup.getState();
    std.debug.print("Backup state: {t}\n", .{backup_state});

    // Trigger manual backup
    const backup_id = backup.triggerBackup() catch |err| {
        std.debug.print("Failed to trigger backup: {}\n", .{err});
        return err;
    };
    std.debug.print("Backup triggered: ID={d}\n", .{backup_id});

    // === Stop HA Services ===
    std.debug.print("\n--- Stopping HA Services ---\n", .{});
    ha_manager.stop();

    std.debug.print("\n=== HA Example Complete ===\n", .{});
}

fn logHaEvent(event: abi.ha.HaEvent) void {
    switch (event) {
        .replica_added => |info| {
            std.debug.print("[HA Event] Replica added: node={d} region={s}\n", .{ info.node_id, info.region });
        },
        .backup_completed => |info| {
            std.debug.print("[HA Event] Backup completed: id={d} size={d} bytes\n", .{ info.backup_id, info.size_bytes });
        },
        .failover_started => |info| {
            std.debug.print("[HA Event] Failover started: {d} -> {d}\n", .{ info.from_node, info.to_node });
        },
        .pitr_checkpoint => |info| {
            std.debug.print("[HA Event] PITR checkpoint: seq={d}\n", .{info.sequence});
        },
        else => {},
    }
}
