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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ModelError;
    use crate::fixtures::{REVISION, Scratch, manifest};

    /// The abi repository root, derived from this crate’s source location.
    ///
    /// Supplied by the *test* rather than baked into library logic, so the
    /// guard in `reject_inside` stays meaningful on hosts other than this one.
    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .canonicalize()
            .expect("crate lives inside the repository")
    }

    /// The sibling abbey checkout, whether or not it exists here.
    fn sibling_abbey() -> PathBuf {
        normalize(&repo_root().join("..").join("abbey"))
    }

    #[test]
    fn the_default_root_is_under_home_and_outside_both_repositories() {
        let _guard = abi_foundation::env::lock_for_test();
        let scratch = Scratch::new("storage_home");
        abi_foundation::env::clear_override(STORAGE_ROOT_VAR);
        abi_foundation::env::set_override("HOME", &scratch.path().display().to_string());

        let root = StorageRoot::resolve().expect("resolves from HOME");
        assert_eq!(root.path(), scratch.path().join(".abi/models"));
        root.reject_inside(&repo_root())
            .expect("default root must be outside the abi repository");
        root.reject_inside(&sibling_abbey())
            .expect("default root must be outside the abbey repository");

        abi_foundation::env::clear_override("HOME");
    }

    #[test]
    fn the_real_default_root_on_this_host_is_outside_both_repositories() {
        // No overrides: this exercises the value an operator would actually get.
        let _guard = abi_foundation::env::lock_for_test();
        abi_foundation::env::clear_override(STORAGE_ROOT_VAR);
        abi_foundation::env::clear_override("HOME");

        let root = StorageRoot::resolve().expect("HOME is set in the test environment");
        root.reject_inside(&repo_root())
            .expect("weights must never resolve inside the abi repository");
        root.reject_inside(&sibling_abbey())
            .expect("weights must never resolve inside the abbey repository");
    }

    #[test]
    fn an_override_is_honoured() {
        let _guard = abi_foundation::env::lock_for_test();
        abi_foundation::env::set_override(STORAGE_ROOT_VAR, "/var/lib/abi/models");
        let root = StorageRoot::resolve().expect("resolves from the override");
        assert_eq!(root.path(), Path::new("/var/lib/abi/models"));
        abi_foundation::env::clear_override(STORAGE_ROOT_VAR);
    }

    #[test]
    fn a_root_inside_the_repository_is_a_hard_error() {
        let repo = repo_root();
        for inside in [
            repo.join("crates/abi-models/weights"),
            repo.join("target/models"),
            repo.clone(),
            repo.join("crates/../models"),
        ] {
            let root = StorageRoot::at(&inside);
            let error = root
                .reject_inside(&repo)
                .expect_err("a storage root inside the repository must be refused");
            assert!(
                matches!(error, ModelError::StorageInsideRepository { .. }),
                "{} gave {error:?}",
                inside.display()
            );
        }
    }

    #[test]
    fn a_misconfigured_override_pointing_into_the_repository_is_caught() {
        let _guard = abi_foundation::env::lock_for_test();
        let inside = repo_root().join("crates/abi-models/downloaded");
        abi_foundation::env::set_override(STORAGE_ROOT_VAR, &inside.display().to_string());

        let root = StorageRoot::resolve().expect("an override always resolves");
        let error = root
            .reject_inside(&repo_root())
            .expect_err("the guard must catch an operator pointing storage into the repo");
        assert!(
            matches!(error, ModelError::StorageInsideRepository { .. }),
            "{error:?}"
        );

        abi_foundation::env::clear_override(STORAGE_ROOT_VAR);
    }

    #[test]
    fn without_home_or_an_override_resolution_fails() {
        let _guard = abi_foundation::env::lock_for_test();
        abi_foundation::env::set_override(STORAGE_ROOT_VAR, "");
        abi_foundation::env::set_override("HOME", "");

        let error = StorageRoot::resolve().expect_err("nothing to resolve from");
        assert!(
            matches!(error, ModelError::StorageRootUnresolvable { .. }),
            "{error:?}"
        );

        abi_foundation::env::clear_override(STORAGE_ROOT_VAR);
        abi_foundation::env::clear_override("HOME");
    }

    #[test]
    fn a_sibling_sharing_a_name_prefix_is_not_inside() {
        // Textual prefix matching would wrongly call this containment.
        assert!(!is_inside(
            Path::new("/srv/abi-models"),
            Path::new("/srv/abi")
        ));
        assert!(is_inside(
            Path::new("/srv/abi/models"),
            Path::new("/srv/abi")
        ));
        assert!(is_inside(Path::new("/srv/abi"), Path::new("/srv/abi")));
        assert!(!is_inside(Path::new("/srv"), Path::new("/srv/abi")));
    }

    #[test]
    fn normalization_resolves_relative_segments() {
        assert_eq!(normalize(Path::new("/a/./b/../c")), PathBuf::from("/a/c"));
        assert_eq!(normalize(Path::new("a/b/../../c")), PathBuf::from("c"));
        assert_eq!(normalize(Path::new("../a")), PathBuf::from("../a"));
        assert_eq!(normalize(Path::new("")), PathBuf::from("."));
    }

    #[test]
    fn model_directories_are_revision_scoped_and_flatten_the_repository() {
        let root = StorageRoot::at("/var/lib/abi/models");
        let manifest = manifest();
        let dir = root.model_dir(&manifest);
        assert_eq!(
            dir,
            Path::new("/var/lib/abi/models/example-org__example-2b").join(REVISION)
        );
        let artifact = manifest.weights().next().expect("weights");
        assert_eq!(
            root.artifact_path(&manifest, artifact),
            dir.join("model.safetensors")
        );
    }
}
