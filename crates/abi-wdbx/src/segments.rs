//! Active-epoch listing, reclamation and retain-latest compaction.
//!
//! The manifest is the source of truth. Compaction atomically publishes a
//! reduced active set before deleting retired files, so a crash may leave
//! harmless orphans but never a manifest that points at a deleted checkpoint.

use crate::{FormatError, Manifest, StorePaths};
use std::path::Path;

/// Segment-maintenance failure.
#[derive(Debug)]
pub enum SegmentError {
    /// Retaining zero checkpoints would remove the WAL recovery baseline.
    InvalidCompactionPolicy,
    /// Manifest parsing or publication failed.
    Format(FormatError),
    /// A retired segment could not be deleted.
    Io {
        /// Path that failed.
        path: std::path::PathBuf,
        /// OS error text.
        message: String,
    },
}

impl std::fmt::Display for SegmentError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCompactionPolicy => {
                formatter.write_str("compaction must retain at least one checkpoint")
            }
            Self::Format(error) => write!(formatter, "{error}"),
            Self::Io { path, message } => {
                write!(formatter, "I/O failed for {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for SegmentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Format(error) => Some(error),
            Self::InvalidCompactionPolicy | Self::Io { .. } => None,
        }
    }
}

impl From<FormatError> for SegmentError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

/// Segment result alias.
pub type Result<T> = std::result::Result<T, SegmentError>;

/// Outcome of retaining the newest checkpoint epochs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionResult {
    /// Active epochs before compaction.
    pub before: usize,
    /// Active epochs after compaction.
    pub after: usize,
    /// Manifest entries retired.
    pub deleted: usize,
    /// Requested retention count.
    pub keep_latest: usize,
    /// Highest active epoch, when any exist.
    pub latest_epoch: Option<u64>,
    /// Lowest epoch retained by this compaction.
    pub watermark_epoch: Option<u64>,
}

/// Active epochs in ascending order.
pub fn active_epochs(paths: &StorePaths) -> Result<Vec<u64>> {
    Ok(paths.read_manifest()?.active)
}

/// Highest active epoch, or `None` for an empty store.
pub fn latest_epoch(paths: &StorePaths) -> Result<Option<u64>> {
    Ok(paths.read_manifest()?.active.last().copied())
}

/// Retire every manifest-listed epoch below `keep_from_epoch`.
///
/// Returns the number of manifest entries retired. Missing retired files count
/// as reclaimed because they are already absent.
pub fn reclaim(paths: &StorePaths, keep_from_epoch: u64) -> Result<usize> {
    let mut manifest = paths.read_manifest()?;
    let retired = manifest
        .active
        .iter()
        .copied()
        .filter(|&epoch| epoch < keep_from_epoch)
        .collect::<Vec<_>>();
    if retired.is_empty() {
        return Ok(0);
    }

    manifest.active.retain(|&epoch| epoch >= keep_from_epoch);
    write_manifest(paths, &manifest)?;

    for epoch in &retired {
        remove_if_present(&paths.segment(*epoch))?;
    }
    Ok(retired.len())
}

/// Retain the newest `keep_latest` active checkpoint epochs.
pub fn compact_retain_latest(paths: &StorePaths, keep_latest: usize) -> Result<CompactionResult> {
    if keep_latest == 0 {
        return Err(SegmentError::InvalidCompactionPolicy);
    }

    let active = active_epochs(paths)?;
    let before = active.len();
    let Some(&latest) = active.last() else {
        return Ok(CompactionResult {
            before: 0,
            after: 0,
            deleted: 0,
            keep_latest,
            latest_epoch: None,
            watermark_epoch: None,
        });
    };

    if before <= keep_latest {
        return Ok(CompactionResult {
            before,
            after: before,
            deleted: 0,
            keep_latest,
            latest_epoch: Some(latest),
            watermark_epoch: active.first().copied(),
        });
    }

    let watermark = active[before - keep_latest];
    let deleted = reclaim(paths, watermark)?;
    Ok(CompactionResult {
        before,
        after: before - deleted,
        deleted,
        keep_latest,
        latest_epoch: Some(latest),
        watermark_epoch: Some(watermark),
    })
}

/// Remove all manifest-listed checkpoints and the manifest.
pub fn reset(paths: &StorePaths) -> Result<()> {
    let manifest = paths.read_manifest()?;
    for epoch in manifest.active {
        remove_if_present(&paths.segment(epoch))?;
    }
    remove_if_present(&paths.manifest())
}

fn write_manifest(paths: &StorePaths, manifest: &Manifest) -> Result<()> {
    let path = paths.manifest();
    abi_foundation::io::write_file_atomic(&path, manifest.render()).map_err(|error| {
        SegmentError::Format(FormatError::Io {
            path,
            message: error.to_string(),
        })
    })
}

fn remove_if_present(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(SegmentError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Snapshot, persistence};

    struct Fixture {
        dir: std::path::PathBuf,
        paths: StorePaths,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let dir = abi_foundation::temp_path::temp_file_path(name, "store");
            std::fs::create_dir_all(&dir).expect("fixture directory");
            Self {
                paths: StorePaths {
                    dir: dir.clone(),
                    base: "segments".to_string(),
                },
                dir,
            }
        }

        fn flush_epochs(&self, count: usize) {
            let mut snapshot = Snapshot::new();
            for index in 0..count {
                snapshot.kv.insert("epoch".to_string(), index.to_string());
                snapshot.recount();
                assert_eq!(
                    persistence::flush(&self.paths, &snapshot).expect("flush"),
                    u64::try_from(index).expect("test epoch fits")
                );
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn active_and_latest_epochs_follow_the_manifest() {
        let fixture = Fixture::new("abi_segments_list");
        fixture.flush_epochs(3);
        assert_eq!(active_epochs(&fixture.paths).expect("active"), [0, 1, 2]);
        assert_eq!(latest_epoch(&fixture.paths).expect("latest"), Some(2));
    }

    #[test]
    fn compaction_retains_the_newest_checkpoints_and_next_epoch() {
        let fixture = Fixture::new("abi_segments_compact");
        fixture.flush_epochs(4);
        let result = compact_retain_latest(&fixture.paths, 2).expect("compact");
        assert_eq!(
            result,
            CompactionResult {
                before: 4,
                after: 2,
                deleted: 2,
                keep_latest: 2,
                latest_epoch: Some(3),
                watermark_epoch: Some(2),
            }
        );
        let manifest = fixture.paths.read_manifest().expect("manifest");
        assert_eq!(manifest.active, [2, 3]);
        assert_eq!(manifest.next_epoch, 4);
        assert!(!fixture.paths.segment(0).exists());
        assert!(!fixture.paths.segment(1).exists());
        assert!(fixture.paths.segment(2).is_file());
        assert!(fixture.paths.segment(3).is_file());
    }

    #[test]
    fn missing_retired_file_is_already_reclaimed() {
        let fixture = Fixture::new("abi_segments_missing");
        fixture.flush_epochs(3);
        std::fs::remove_file(fixture.paths.segment(0)).expect("remove early");
        let result = compact_retain_latest(&fixture.paths, 1).expect("compact");
        assert_eq!(result.deleted, 2);
        assert_eq!(active_epochs(&fixture.paths).expect("active"), [2]);
    }

    #[test]
    fn single_and_empty_stores_are_no_ops() {
        let empty = Fixture::new("abi_segments_empty");
        let result = compact_retain_latest(&empty.paths, 1).expect("empty compact");
        assert_eq!(result.before, 0);
        assert_eq!(result.after, 0);

        let single = Fixture::new("abi_segments_single");
        single.flush_epochs(1);
        let result = compact_retain_latest(&single.paths, 1).expect("single compact");
        assert_eq!(result.before, 1);
        assert_eq!(result.after, 1);
        assert_eq!(result.deleted, 0);
        assert!(single.paths.segment(0).is_file());
    }

    #[test]
    fn retaining_zero_is_rejected_without_mutation() {
        let fixture = Fixture::new("abi_segments_zero");
        fixture.flush_epochs(2);
        assert!(matches!(
            compact_retain_latest(&fixture.paths, 0),
            Err(SegmentError::InvalidCompactionPolicy)
        ));
        assert_eq!(active_epochs(&fixture.paths).expect("active"), [0, 1]);
    }

    #[test]
    fn reset_removes_only_manifest_listed_checkpoints() {
        let fixture = Fixture::new("abi_segments_reset");
        fixture.flush_epochs(2);
        let orphan = fixture.paths.segment(99);
        std::fs::write(&orphan, "unlisted").expect("orphan");
        reset(&fixture.paths).expect("reset");
        assert!(!fixture.paths.manifest().exists());
        assert!(!fixture.paths.segment(0).exists());
        assert!(!fixture.paths.segment(1).exists());
        assert!(
            orphan.is_file(),
            "unlisted files are outside manifest scope"
        );
    }
}
