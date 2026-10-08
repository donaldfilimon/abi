use super::*;
use crate::training_registry::AuthorityPolicy;
use abi_wdbx::durable::DurableStore;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
struct Fixture {
    root: PathBuf,
    store: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let parent = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "abi-artifact-test-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let mut parent_builder = fs::DirBuilder::new();
        #[cfg(unix)]
        parent_builder.mode(0o700);
        parent_builder.create(&parent).unwrap();
        let root = parent.join("artifacts");
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        builder.mode(0o700);
        builder.create(&root).unwrap();
        let store = parent.join("registry");
        builder.create(&store).unwrap();
        Self { root, store }
    }
    fn open(&self) -> TrainingRegistry {
        TrainingRegistry::new(
            DurableStore::open_directory(&self.store).unwrap(),
            AuthorityPolicy {
                principals: BTreeMap::new(),
                evidence_registrars: BTreeSet::new(),
                human_approvers: BTreeSet::new(),
                permission_operators: BTreeSet::new(),
                revocation_operators: BTreeSet::new(),
                qualification_reviewers: BTreeSet::new(),
            },
        )
        .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(self.root.parent().unwrap()).unwrap();
    }
}
fn reference(bytes: &[u8]) -> ArtifactRef {
    let sha256 = sha256_hex(bytes);
    ArtifactRef {
        version: 1,
        relative_path: format!("{sha256}.artifact"),
        sha256,
        byte_length: bytes.len() as u64,
    }
}
#[test]
fn invalid_artifact_path_refuses() {
    let fixture = Fixture::new();
    let mut artifact = reference(b"artifact");
    artifact.relative_path = "../escape".into();
    assert!(verify_artifact(&fixture.root, &artifact).is_err());
}
#[cfg(unix)]
#[test]
fn missing_corrupt_and_digest_mismatch_refuse() {
    let fixture = Fixture::new();
    let mut registry = fixture.open();
    let artifact = reference(b"artifact");
    assert!(verify_artifact(&fixture.root, &artifact).is_err());
    assert!(publish_artifact(&mut registry, &fixture.root, &artifact.sha256, b"wrong").is_err());
    assert!(!fixture.root.join(&artifact.relative_path).exists());
    let published =
        publish_artifact(&mut registry, &fixture.root, &artifact.sha256, b"artifact").unwrap();
    assert_eq!(published, artifact);
    verify_artifact(&fixture.root, &artifact).unwrap();
    fs::write(fixture.root.join(&artifact.relative_path), b"corrupt!").unwrap();
    assert!(verify_artifact(&fixture.root, &artifact).is_err());
    assert!(publish_artifact(&mut registry, &fixture.root, &artifact.sha256, b"artifact").is_err());
}
#[cfg(unix)]
#[test]
fn symlink_root_destination_and_public_modes_refuse_without_bytes() {
    use std::os::unix::fs::symlink;
    let fixture = Fixture::new();
    let mut registry = fixture.open();
    let artifact = reference(b"artifact");
    let alias = fixture.root.with_file_name("alias");
    symlink(&fixture.root, &alias).unwrap();
    assert!(publish_artifact(&mut registry, &alias, &artifact.sha256, b"artifact").is_err());
    let outside = fixture.root.with_file_name("outside");
    fs::write(&outside, b"preserve").unwrap();
    symlink(&outside, fixture.root.join(&artifact.relative_path)).unwrap();
    assert!(publish_artifact(&mut registry, &fixture.root, &artifact.sha256, b"artifact").is_err());
    assert_eq!(fs::read(outside).unwrap(), b"preserve");
    fs::remove_file(fixture.root.join(&artifact.relative_path)).unwrap();
    fs::set_permissions(&fixture.root, fs::Permissions::from_mode(0o777)).unwrap();
    assert!(publish_artifact(&mut registry, &fixture.root, &artifact.sha256, b"artifact").is_err());
    assert_eq!(fs::read_dir(&fixture.root).unwrap().count(), 0);
}
#[cfg(unix)]
#[test]
fn interrupted_staging_never_registers_and_competing_writer_refuses() {
    let fixture = Fixture::new();
    let mut registry = fixture.open();
    let artifact = reference(b"artifact");
    assert!(DurableStore::open_directory(&fixture.store).is_err());
    assert!(
        publish(
            &mut registry,
            &fixture.root,
            &artifact.sha256,
            b"artifact",
            Publication::Interrupted
        )
        .is_err()
    );
    assert_eq!(fs::read_dir(&fixture.root).unwrap().count(), 0);
    drop(registry);
    let store = DurableStore::open_directory(&fixture.store).unwrap();
    assert!(store.get("training:v1:state").is_none());
}
#[cfg(unix)]
#[test]
fn file_permissions_and_hardlinks_refuse() {
    let fixture = Fixture::new();
    let mut registry = fixture.open();
    let artifact = reference(b"artifact");
    publish_artifact(&mut registry, &fixture.root, &artifact.sha256, b"artifact").unwrap();
    let path = fixture.root.join(&artifact.relative_path);
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(verify_artifact(&fixture.root, &artifact).is_err());
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    fs::hard_link(&path, fixture.root.join("hardlink")).unwrap();
    assert!(verify_artifact(&fixture.root, &artifact).is_err());
}

#[cfg(unix)]
#[test]
fn kernel_probe_owner_and_exact_mode_must_match() {
    assert!(validate_owner(501, 502, 0o100_600).is_err());
    assert!(validate_owner(501, 501, 0o100_640).is_err());
    assert!(validate_owner(501, 501, 0o104_600).is_err());
    validate_owner(501, 501, 0o100_600).unwrap();
}
