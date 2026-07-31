//! Manifest and store paths for the WDBX on-disk format.

use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::hash::{FormatError, Result};
use crate::segment::MIRROR_EPOCH_HEADER;

/// Magic first line of the manifest.
pub const MANIFEST_HEADER: &str = "# ABI-WDBX-SEGMENTS v1";

/// The segment manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// The epoch the next new segment will take.
    pub next_epoch: u64,
    /// Live epochs, ascending and deduplicated.
    ///
    /// A segment file whose epoch is absent is garbage awaiting collection and
    /// must not be read.
    pub active: Vec<u64>,
}

impl Manifest {
    /// An empty manifest.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            next_epoch: 0,
            active: Vec::new(),
        }
    }

    /// Parse manifest text.
    pub fn parse(content: &str) -> Result<Self> {
        let mut lines = content.lines();
        let header = lines.next().unwrap_or_default();
        if header != MANIFEST_HEADER {
            return Err(FormatError::InvalidManifest {
                reason: format!("expected {MANIFEST_HEADER:?}, found {header:?}"),
            });
        }

        let mut next_epoch = None;
        let mut active = BTreeSet::new();

        for line in lines {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                return Err(FormatError::InvalidManifest {
                    reason: format!("line {line:?} is not key=value"),
                });
            };
            match key.trim() {
                "next_epoch" => {
                    next_epoch = Some(value.trim().parse::<u64>().map_err(|_| {
                        FormatError::InvalidManifest {
                            reason: format!("next_epoch {value:?} is not a number"),
                        }
                    })?);
                }
                "active" => {
                    for token in value.split(',') {
                        let token = token.trim();
                        if token.is_empty() {
                            continue;
                        }
                        active.insert(token.parse::<u64>().map_err(|_| {
                            FormatError::InvalidManifest {
                                reason: format!("active epoch {token:?} is not a number"),
                            }
                        })?);
                    }
                }
                // Forward compatibility: a newer writer may add keys, and
                // refusing to open the store over one would be worse than
                // ignoring it.
                _ => {}
            }
        }

        let active: Vec<u64> = active.into_iter().collect();
        Ok(Self {
            next_epoch: next_epoch.unwrap_or_else(|| active.last().map_or(0, |last| last + 1)),
            active,
        })
    }

    /// Render manifest text.
    #[must_use]
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(out, "{MANIFEST_HEADER}");
        let _ = writeln!(out, "next_epoch={}", self.next_epoch);
        let joined: Vec<String> = self.active.iter().map(u64::to_string).collect();
        let _ = writeln!(out, "active={}", joined.join(","));
        out
    }
}

/// Paths within a WDBX store directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorePaths {
    /// The store directory.
    pub dir: PathBuf,
    /// Base name of the store files, `wdbx` by default.
    pub base: String,
}

impl StorePaths {
    /// Paths for `dir` using the default `wdbx` base name.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            base: "wdbx".to_string(),
        }
    }

    /// The binary index file.
    #[must_use]
    pub fn index(&self) -> PathBuf {
        self.dir.join(&self.base)
    }

    /// The manifest file.
    #[must_use]
    pub fn manifest(&self) -> PathBuf {
        self.dir.join(format!("{}.manifest", self.base))
    }

    /// Epoch provenance for the monolithic compatibility mirror.
    #[must_use]
    pub fn mirror_epoch(&self) -> PathBuf {
        self.dir.join(format!("{}.mirror-epoch", self.base))
    }

    /// The segment file for `epoch`.
    #[must_use]
    pub fn segment(&self, epoch: u64) -> PathBuf {
        self.dir.join(format!("{}.seg.{epoch}.jsonl", self.base))
    }

    /// Read the manifest, or an empty one if the store does not exist yet.
    ///
    /// A missing manifest is "no store yet", not an error — that is what a first
    /// run looks like.
    pub fn read_manifest(&self) -> Result<Manifest> {
        let path = self.manifest();
        match std::fs::read_to_string(&path) {
            Ok(content) => Manifest::parse(&content),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Manifest::empty()),
            Err(e) => Err(FormatError::Io {
                path,
                message: e.to_string(),
            }),
        }
    }

    /// Read compatibility-mirror epoch provenance.
    ///
    /// A missing sidecar means epoch 0, which preserves readability of every
    /// Zig-written monolithic snapshot.
    pub fn read_mirror_epoch(&self) -> Result<u64> {
        let path = self.mirror_epoch();
        let content = match std::fs::read_to_string(&path) {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(error) => {
                return Err(FormatError::Io {
                    path,
                    message: error.to_string(),
                });
            }
        };
        let mut lines = content.lines();
        if lines.next() != Some(MIRROR_EPOCH_HEADER) {
            return Err(FormatError::InvalidManifest {
                reason: format!(
                    "mirror epoch sidecar {} has an invalid header",
                    path.display()
                ),
            });
        }
        let epoch = lines
            .next()
            .and_then(|line| line.strip_prefix("epoch="))
            .and_then(|value| value.parse::<u64>().ok())
            .ok_or_else(|| FormatError::InvalidManifest {
                reason: format!(
                    "mirror epoch sidecar {} has an invalid epoch",
                    path.display()
                ),
            })?;
        if lines.any(|line| !line.trim().is_empty()) {
            return Err(FormatError::InvalidManifest {
                reason: format!(
                    "mirror epoch sidecar {} has unexpected content",
                    path.display()
                ),
            });
        }
        Ok(epoch)
    }
}
