//! Where model weights are allowed to live.
//!
//! Weights must never land inside a source repository — they are large, opaque,
//! and licensed separately from the code. The rule is enforced in two
//! independent pieces so the guard is not a no-op on machines other than the
//! one that built the crate:
//!
//! - [`StorageRoot::resolve`] produces a root that is external *by
//!   construction*: an operator override, or `$HOME/.abi/models`.
//! - [`StorageRoot::reject_inside`] is a pure predicate that takes the
//!   repository root **from the caller** and refuses a storage root beneath it.
//!
//! Containment is decided lexically, after normalizing `.` and `..` segments,
//! because the storage root frequently does not exist yet and so cannot be
//! canonicalized.

use crate::error::ModelError;
use crate::manifest::{Artifact, ModelManifest};
use std::path::{Component, Path, PathBuf};

/// Environment variable overriding the model storage root.
///
/// Read through `abi_foundation::env::get`, so the framework's test overlay and
/// its "empty means unset" rule both apply. See the crate-level note about why
/// this constant is declared here rather than in `abi-foundation`.
pub const STORAGE_ROOT_VAR: &str = "ABI_MODELS_DIR";

/// Directory beneath `$HOME` used when no override is set.
const DEFAULT_SUFFIX: &str = ".abi/models";

/// A validated root directory for downloaded weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageRoot {
    /// Normalized absolute-or-relative root path.
    path: PathBuf,
}

impl StorageRoot {
    /// Resolve the storage root from the environment.
    ///
    /// Precedence: [`STORAGE_ROOT_VAR`], then `$HOME/.abi/models`. Neither
    /// candidate is inside a source checkout, which is what makes the default
    /// safe without consulting any repository path.
    pub fn resolve() -> Result<Self, ModelError> {
        if let Some(override_path) = abi_foundation::env::get(STORAGE_ROOT_VAR) {
            return Ok(Self::at(override_path));
        }
        let Some(home) = abi_foundation::env::get("HOME") else {
            return Err(ModelError::StorageRootUnresolvable {
                var: STORAGE_ROOT_VAR,
            });
        };
        Ok(Self::at(Path::new(&home).join(DEFAULT_SUFFIX)))
    }

    /// Use an explicit root, normalizing it.
    #[must_use]
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self {
            path: normalize(&path.into()),
        }
    }

    /// The root directory.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Refuse this root if it lies inside `repository`.
    ///
    /// The caller supplies the repository root — typically its own source
    /// directory — so the check is meaningful wherever the binary runs, not
    /// only on the build host.
    pub fn reject_inside(&self, repository: &Path) -> Result<(), ModelError> {
        let repository = normalize(repository);
        if is_inside(&self.path, &repository) {
            return Err(ModelError::StorageInsideRepository {
                path: self.path.clone(),
                repository,
            });
        }
        Ok(())
    }

    /// Directory holding one model at one revision.
    ///
    /// Revision-scoped, so two revisions of the same repository never share a
    /// directory and a stale file can never be mistaken for a current one.
    #[must_use]
    pub fn model_dir(&self, manifest: &ModelManifest) -> PathBuf {
        self.path
            .join(slug(&manifest.repository))
            .join(manifest.revision.as_str())
    }

    /// Final on-disk path of one artifact.
    #[must_use]
    pub fn artifact_path(&self, manifest: &ModelManifest, artifact: &Artifact) -> PathBuf {
        self.model_dir(manifest).join(&artifact.path)
    }
}

/// Flatten a `org/name` repository into a single path segment.
fn slug(repository: &str) -> String {
    repository.replace(['/', '\\'], "__")
}

/// Resolve `.` and `..` textually, without touching the filesystem.
#[must_use]
pub fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(out.components().next_back(), Some(Component::Normal(_))) {
                    out.pop();
                } else {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    out
}

/// True when `candidate` is `root` or lies beneath it, comparing whole path
/// components so `/tmp/abi-models` is not treated as inside `/tmp/abi`.
#[must_use]
pub fn is_inside(candidate: &Path, root: &Path) -> bool {
    let candidate = normalize(candidate);
    let root = normalize(root);
    let mut theirs = root.components();
    let mut ours = candidate.components();
    loop {
        match (theirs.next(), ours.next()) {
            (None, _) => return true,
            (Some(_), None) => return false,
            (Some(a), Some(b)) if a != b => return false,
            (Some(_), Some(_)) => {}
        }
    }
}
