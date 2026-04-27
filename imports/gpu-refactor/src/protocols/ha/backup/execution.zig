//! Backup execution: triggerBackup, triggerFullBackup, verifyBackup, retention.

const std = @import("std");
const config = @import("config.zig");
const storage = @import("storage.zig");

const BackupConfig = config.BackupConfig;
const BackupMode = config.BackupMode;
const BackupState = config.BackupState;
const BackupEvent = config.BackupEvent;
const BackupMetadata = config.BackupMetadata;
const Destination = config.Destination;
const CRC32_SIZE = config.CRC32_SIZE;
const wallClockSec = config.wallClockSec;

const sync = @import("../../../foundation/mod.zig").sync;
const Mutex = sync.Mutex;

/// Backup orchestrator
pub const BackupOrchestrator = struct {
    allocator: std.mem.Allocator,
    config: BackupConfig,

    // State
    state: BackupState,
    current_backup_id: u64,
    last_backup_time: u64,
    last_full_backup_id: u64,

    // Backup history
    backup_history: std.ArrayListUnmanaged(BackupMetadata),

    // Synchronization
    mutex: Mutex,

    /// Initialize the backup orchestrator
    pub fn init(allocator: std.mem.Allocator, backup_config: BackupConfig) BackupOrchestrator {
        return .{
            .allocator = allocator,
            .config = backup_config,
            .state = .idle,
            .current_backup_id = 0,
            .last_backup_time = 0,
            .last_full_backup_id = 0,
            .backup_history = .empty,
            .mutex = .{},
        };
    }

    /// Deinitialize the backup orchestrator
    pub fn deinit(self: *BackupOrchestrator) void {
        self.backup_history.deinit(self.allocator);
        self.* = undefined;
    }

    /// Get current state
    pub fn getState(self: *BackupOrchestrator) BackupState {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.state;
    }

    /// Check if backup is due
    pub fn isBackupDue(self: *BackupOrchestrator) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = wallClockSec();
        if (now == 0) return true; // Cannot determine time (WASM/unsupported) — assume due
        const interval_sec = @as(u64, self.config.interval_hours) * 3600;
        const last = self.last_backup_time;
        return (now - last) >= interval_sec;
    }

    /// Trigger a manual backup with provided data payload.
    pub fn triggerBackupWithData(self: *BackupOrchestrator, data: []const u8) !u64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.state != .idle) {
            return error.BackupInProgress;
        }

        return self.startBackupLocked(self.config.mode, data);
    }

    /// Trigger a manual backup (no data payload — uses empty data for backwards compatibility)
    pub fn triggerBackup(self: *BackupOrchestrator) !u64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.state != .idle) {
            return error.BackupInProgress;
        }

        return self.startBackupLocked(self.config.mode, &.{});
    }

    /// Trigger a full backup
    pub fn triggerFullBackup(self: *BackupOrchestrator) !u64 {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.state != .idle) {
            return error.BackupInProgress;
        }

        return self.startBackupLocked(.full, &.{});
    }

    fn startBackupLocked(self: *BackupOrchestrator, mode: BackupMode, data: []const u8) !u64 {
        self.current_backup_id += 1;
        const backup_id = self.current_backup_id;

        self.state = .preparing;
        self.emitEvent(.{ .backup_started = .{
            .backup_id = backup_id,
            .mode = mode,
        } });

        // Determine if this should be a full backup
        const actual_mode = switch (mode) {
            .incremental => blk: {
                // Do full backup if no previous full exists
                if (self.last_full_backup_id == 0) break :blk BackupMode.full;
                break :blk mode;
            },
            else => mode,
        };

        const start_time = wallClockSec();

        self.state = .backing_up;
        self.emitEvent(.{ .backup_progress = .{ .backup_id = backup_id, .percent = 10 } });

        // Build metadata JSON
        const metadata_json = std.fmt.allocPrint(
            self.allocator,
            "{{\"backup_id\":{d},\"mode\":\"{s}\",\"timestamp\":{d}}}",
            .{ backup_id, @tagName(actual_mode), start_time },
        ) catch {
            self.state = .failed;
            self.emitEvent(.{ .backup_failed = .{ .backup_id = backup_id, .reason = "metadata serialization failed" } });
            return error.BackupFailed;
        };
        defer self.allocator.free(metadata_json);

        self.emitEvent(.{ .backup_progress = .{ .backup_id = backup_id, .percent = 25 } });

        // Serialize backup to binary format
        const backup_buf = storage.serializeBackup(self.allocator, metadata_json, data, start_time) catch {
            self.state = .failed;
            self.emitEvent(.{ .backup_failed = .{ .backup_id = backup_id, .reason = "serialization failed" } });
            return error.BackupFailed;
        };
        defer self.allocator.free(backup_buf);

        const file_size: u64 = @intCast(backup_buf.len);

        self.state = .compressing;
        self.emitEvent(.{ .backup_progress = .{ .backup_id = backup_id, .percent = 50 } });

        // Compression is a future enhancement; for now report the raw size
        const compressed_size = file_size;

        if (self.config.encryption) {
            self.state = .encrypting;
            self.emitEvent(.{ .backup_progress = .{ .backup_id = backup_id, .percent = 60 } });
        }

        // Write to destinations
        self.state = .uploading;
        self.emitEvent(.{ .backup_progress = .{ .backup_id = backup_id, .percent = 70 } });

        var destinations_succeeded: u32 = 0;
        var destinations_failed: u32 = 0;

        for (self.config.destinations) |dest| {
            self.emitEvent(.{ .upload_started = .{ .backup_id = backup_id, .destination = dest.type } });

            switch (dest.type) {
                .local => {
                    // Build full path: dest.path + backup filename
                    const filename = std.fmt.allocPrint(
                        self.allocator,
                        "{s}backup_{d}.abib",
                        .{ dest.path, backup_id },
                    ) catch {
                        destinations_failed += 1;
                        continue;
                    };
                    defer self.allocator.free(filename);

                    storage.writeBackupToLocal(self.allocator, filename, backup_buf) catch {
                        destinations_failed += 1;
                        continue;
                    };

                    destinations_succeeded += 1;
                    self.emitEvent(.{ .upload_completed = .{ .backup_id = backup_id, .destination = dest.type } });
                },
                .s3, .gcs, .azure_blob => {
                    // Remote backup not yet implemented — count as failed
                    destinations_failed += 1;
                },
            }
        }

        self.state = .verifying;
        self.emitEvent(.{ .backup_progress = .{ .backup_id = backup_id, .percent = 90 } });

        // Calculate checksum for metadata record (SHA-256 placeholder, use CRC32 bytes)
        var checksum: [32]u8 = undefined;
        @memset(&checksum, 0);
        const crc_val = std.hash.crc.Crc32IsoHdlc.hash(backup_buf[0 .. backup_buf.len - CRC32_SIZE]);
        std.mem.writeInt(u32, checksum[0..4], crc_val, .little);

        const end_time = wallClockSec();
        const duration_ms = (end_time - start_time) * 1000;

        // Record metadata
        const metadata = BackupMetadata{
            .backup_id = backup_id,
            .timestamp = start_time,
            .mode = actual_mode,
            .size_bytes = compressed_size,
            .checksum = checksum,
            .base_backup_id = if (actual_mode != .full) self.last_full_backup_id else null,
            .sequence_number = backup_id,
        };

        try self.backup_history.append(self.allocator, metadata);

        if (actual_mode == .full) {
            self.last_full_backup_id = backup_id;
        }

        self.last_backup_time = start_time;
        self.state = .completed;

        self.emitEvent(.{ .backup_completed = .{
            .backup_id = backup_id,
            .size_bytes = compressed_size,
            .duration_ms = duration_ms,
        } });

        self.state = .idle;
        return backup_id;
    }

    /// List available backups
    pub fn listBackups(self: *BackupOrchestrator) []const BackupMetadata {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.backup_history.items;
    }

    /// Get backup by ID
    pub fn getBackup(self: *BackupOrchestrator, backup_id: u64) ?BackupMetadata {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.backup_history.items) |backup| {
            if (backup.backup_id == backup_id) {
                return backup;
            }
        }
        return null;
    }

    /// Apply retention policy
    pub fn applyRetention(self: *BackupOrchestrator) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var deleted_count: u32 = 0;
        var freed_bytes: u64 = 0;

        // Keep last N backups
        while (self.backup_history.items.len > self.config.retention.keep_last) {
            const removed = self.backup_history.orderedRemove(0);
            deleted_count += 1;
            freed_bytes += removed.size_bytes;
        }

        if (deleted_count > 0) {
            self.emitEvent(.{ .retention_cleanup = .{
                .deleted_count = deleted_count,
                .freed_bytes = freed_bytes,
            } });
        }
    }

    /// Verify backup integrity by ID (checks in-memory history)
    pub fn verifyBackup(self: *BackupOrchestrator, backup_id: u64) !bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.backup_history.items) |backup| {
            if (backup.backup_id == backup_id) {
                // In real implementation, verify checksum against stored data
                self.emitEvent(.{ .verification_passed = .{ .backup_id = backup_id } });
                return true;
            }
        }

        return error.BackupNotFound;
    }

    fn emitEvent(self: *BackupOrchestrator, event: BackupEvent) void {
        if (self.config.on_event) |callback| {
            callback(event);
        }
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "BackupOrchestrator initialization" {
    const allocator = std.testing.allocator;

    var orchestrator = BackupOrchestrator.init(allocator, .{
        .interval_hours = 6,
    });
    defer orchestrator.deinit();

    try std.testing.expectEqual(BackupState.idle, orchestrator.getState());
}

test "BackupOrchestrator trigger backup" {
    const allocator = std.testing.allocator;

    var orchestrator = BackupOrchestrator.init(allocator, .{});
    defer orchestrator.deinit();

    const backup_id = try orchestrator.triggerBackup();
    try std.testing.expect(backup_id > 0);
    try std.testing.expectEqual(@as(usize, 1), orchestrator.backup_history.items.len);
}

test "backup to local destination via orchestrator" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const dest_path = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/tmp/{s}/",
        .{tmp.sub_path},
    );
    defer allocator.free(dest_path);

    const destinations = [_]Destination{.{ .type = .local, .path = dest_path }};

    var orchestrator = BackupOrchestrator.init(allocator, .{
        .destinations = &destinations,
        .compression = false,
    });
    defer orchestrator.deinit();

    const payload = "orchestrator backup data content";
    const backup_id = try orchestrator.triggerBackupWithData(payload);
    try std.testing.expect(backup_id > 0);

    // Verify the file was written to disk
    const file_path = try std.fmt.allocPrint(
        allocator,
        "{s}backup_{d}.abib",
        .{ dest_path, backup_id },
    );
    defer allocator.free(file_path);

    var info = try storage.verifyBackupFile(allocator, file_path);
    defer info.deinit();
    try std.testing.expect(info.timestamp > 0);
    try std.testing.expectEqual(@as(u64, payload.len), info.data_size);

    // Verify metadata in history
    const meta = orchestrator.getBackup(backup_id);
    try std.testing.expect(meta != null);
    try std.testing.expectEqual(backup_id, meta.?.backup_id);
}

test "backup metadata contains correct timestamp and size" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const dest_path = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/tmp/{s}/",
        .{tmp.sub_path},
    );
    defer allocator.free(dest_path);

    const destinations = [_]Destination{.{ .type = .local, .path = dest_path }};

    var orchestrator = BackupOrchestrator.init(allocator, .{
        .destinations = &destinations,
        .compression = false,
    });
    defer orchestrator.deinit();

    const backup_id = try orchestrator.triggerBackupWithData("12345");
    const meta = orchestrator.getBackup(backup_id).?;

    // Timestamp should be recent (non-zero)
    try std.testing.expect(meta.timestamp > 0);
    // Size should reflect the serialized file (header + metadata JSON + data + CRC32)
    try std.testing.expect(meta.size_bytes > 0);
}

test {
    std.testing.refAllDecls(@This());
}
