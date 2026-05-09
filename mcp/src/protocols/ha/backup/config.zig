//! Backup configuration types, scheduling parameters, and validation.

const std = @import("std");

/// Backup configuration
pub const BackupConfig = struct {
    /// Backup interval in hours
    interval_hours: u32 = 6,
    /// Backup mode
    mode: BackupMode = .incremental,
    /// Retention policy
    retention: RetentionPolicy = .{},
    /// Enable compression
    compression: bool = true,
    /// Compression level (1-9)
    compression_level: u8 = 6,
    /// Enable encryption
    encryption: bool = false,
    /// Backup destinations
    destinations: []const Destination = &.{.{ .type = .local, .path = "backups/" }},
    /// Callback for backup events
    on_event: ?*const fn (BackupEvent) void = null,
};

/// Backup mode
pub const BackupMode = enum {
    /// Full backup every time
    full,
    /// Incremental with periodic full
    incremental,
    /// Differential from last full
    differential,
};

/// Retention policy
pub const RetentionPolicy = struct {
    /// Keep last N backups
    keep_last: u32 = 10,
    /// Keep daily backups for N days
    keep_daily_days: u32 = 7,
    /// Keep weekly backups for N weeks
    keep_weekly_weeks: u32 = 4,
    /// Keep monthly backups for N months
    keep_monthly_months: u32 = 12,
};

/// Backup destination
pub const Destination = struct {
    type: DestinationType,
    path: []const u8,
    bucket: ?[]const u8 = null,
    region: ?[]const u8 = null,
    credentials: ?[]const u8 = null,
};

/// Destination types
pub const DestinationType = enum {
    local,
    s3,
    gcs,
    azure_blob,
};

/// Backup state
pub const BackupState = enum {
    idle,
    preparing,
    backing_up,
    compressing,
    encrypting,
    uploading,
    verifying,
    completed,
    failed,
};

/// Backup events
pub const BackupEvent = union(enum) {
    backup_started: struct { backup_id: u64, mode: BackupMode },
    backup_progress: struct { backup_id: u64, percent: u8 },
    backup_completed: struct { backup_id: u64, size_bytes: u64, duration_ms: u64 },
    backup_failed: struct { backup_id: u64, reason: []const u8 },
    upload_started: struct { backup_id: u64, destination: DestinationType },
    upload_completed: struct { backup_id: u64, destination: DestinationType },
    verification_passed: struct { backup_id: u64 },
    verification_failed: struct { backup_id: u64, reason: []const u8 },
    retention_cleanup: struct { deleted_count: u32, freed_bytes: u64 },
};

/// Backup result
pub const BackupResult = struct {
    backup_id: u64,
    timestamp: u64,
    mode: BackupMode,
    size_bytes: u64,
    size_compressed: u64,
    duration_ms: u64,
    checksum: [32]u8,
    destinations_succeeded: u32,
    destinations_failed: u32,
};

/// Backup metadata
pub const BackupMetadata = struct {
    backup_id: u64,
    timestamp: u64,
    mode: BackupMode,
    size_bytes: u64,
    checksum: [32]u8,
    base_backup_id: ?u64, // For incremental/differential
    sequence_number: u64,
};

/// Verified backup info returned by verifyBackupFile.
/// When returned from `verifyBackupFile`, the caller owns `_raw_buf` and must
/// call `deinit` to free it.  When returned from `deserializeBackup`, the
/// slices point into the caller-supplied buffer and `_raw_buf` is null.
pub const VerifiedBackupInfo = struct {
    timestamp: u64,
    metadata_json: []const u8,
    data_size: u64,
    format_version: u16,
    /// Backing allocation (set only by verifyBackupFile).
    _raw_buf: ?[]const u8 = null,
    _allocator: ?std.mem.Allocator = null,

    pub fn deinit(self: *VerifiedBackupInfo) void {
        if (self._raw_buf) |buf| {
            if (self._allocator) |alloc| {
                alloc.free(buf);
            }
        }
        self.* = undefined;
    }
};

/// Wall-clock timestamp in seconds (suitable for backup metadata that must
/// survive across process restarts, unlike the monotonic foundation.time).
pub fn wallClockSec() u64 {
    if (@hasDecl(std.posix, "system")) {
        var ts: std.posix.timespec = undefined;
        if (std.posix.errno(std.posix.system.clock_gettime(.REALTIME, &ts)) == .SUCCESS) {
            return @intCast(@max(ts.sec, 0));
        }
    }
    // Fallback for WASM or unsupported platforms
    return 0;
}

/// Backup file format magic bytes: "ABIB" (ABI Backup)
pub const BACKUP_MAGIC = [4]u8{ 'A', 'B', 'I', 'B' };

/// Current backup format version
pub const BACKUP_FORMAT_VERSION: u16 = 1;

/// Fixed header size: magic(4) + version(2) + timestamp(8) + metadata_size(4) + data_size(8) = 26 bytes
pub const BACKUP_HEADER_SIZE: usize = 26;

/// CRC32 checksum size (appended at end of file)
pub const CRC32_SIZE: usize = 4;

test {
    std.testing.refAllDecls(@This());
}
