//! Automated Backup Orchestrator
//!
//! Provides comprehensive backup management:
//! - Scheduled automatic backups
//! - Incremental and full backup modes
//! - Compression and encryption
//! - Multi-destination support (local, S3, GCS)
//! - Backup verification and integrity checks
//!
//! Implementation decomposed into backup/ subdirectory.

const mod = @import("backup/mod.zig");

pub const BackupConfig = mod.BackupConfig;
pub const BackupMode = mod.BackupMode;
pub const RetentionPolicy = mod.RetentionPolicy;
pub const Destination = mod.Destination;
pub const DestinationType = mod.DestinationType;
pub const BackupState = mod.BackupState;
pub const BackupEvent = mod.BackupEvent;
pub const BackupResult = mod.BackupResult;
pub const BackupMetadata = mod.BackupMetadata;
pub const VerifiedBackupInfo = mod.VerifiedBackupInfo;

pub const serializeBackup = mod.serializeBackup;
pub const deserializeBackup = mod.deserializeBackup;
pub const verifyBackupFile = mod.verifyBackupFile;

pub const BackupOrchestrator = mod.BackupOrchestrator;

const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}
