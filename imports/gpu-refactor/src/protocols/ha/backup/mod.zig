//! Automated Backup Orchestrator
//!
//! Provides comprehensive backup management:
//! - Scheduled automatic backups
//! - Incremental and full backup modes
//! - Compression and encryption
//! - Multi-destination support (local, S3, GCS)
//! - Backup verification and integrity checks

// Re-export all public types from sub-modules
pub const BackupConfig = @import("config.zig").BackupConfig;
pub const BackupMode = @import("config.zig").BackupMode;
pub const RetentionPolicy = @import("config.zig").RetentionPolicy;
pub const Destination = @import("config.zig").Destination;
pub const DestinationType = @import("config.zig").DestinationType;
pub const BackupState = @import("config.zig").BackupState;
pub const BackupEvent = @import("config.zig").BackupEvent;
pub const BackupResult = @import("config.zig").BackupResult;
pub const BackupMetadata = @import("config.zig").BackupMetadata;
pub const VerifiedBackupInfo = @import("config.zig").VerifiedBackupInfo;

pub const serializeBackup = @import("storage.zig").serializeBackup;
pub const deserializeBackup = @import("storage.zig").deserializeBackup;
pub const verifyBackupFile = @import("storage.zig").verifyBackupFile;

pub const BackupOrchestrator = @import("execution.zig").BackupOrchestrator;

const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}
