//! Multi-process crash and compaction stress for the lock-free v2 writer path.

use abi_wdbx::v2::V2AuditBlock;
use abi_wdbx::{RecordId, V2Mutation, V2Store};
use sha2::{Digest as _, Sha256};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use uuid::Uuid;

const CHILDREN: usize = 50;
const ROOT_ENV: &str = "ABI_V2_PROCESS_STRESS_ROOT";
const INDEX_ENV: &str = "ABI_V2_PROCESS_STRESS_INDEX";

fn hash_for(index: usize) -> String {
    format!(
        "{:x}",
        Sha256::digest(format!("audit-child-{index}").as_bytes())
    )
}

fn wait_for(path: &Path, deadline: Instant) {
    while !path.exists() {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {}",
            path.display()
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn child_process_entry() {
    let Some(root) = std::env::var_os(ROOT_ENV).map(PathBuf::from) else {
        return;
    };
    let index = std::env::var(INDEX_ENV)
        .expect("child index")
        .parse::<usize>()
        .expect("numeric child index");
    let bounded_index = u16::try_from(index).expect("stress index is bounded");
    let isolated_root = root.join("isolated").join(index.to_string());
    let shared_root = root.join("shared");
    let vector_id = RecordId::V2(Uuid::new_v4());
    let audit_hash = hash_for(index);
    let mut isolated = V2Store::open(&isolated_root).expect("open isolated writer");
    let transaction = isolated
        .commit_export(vec![
            V2Mutation::PutKv {
                key: "overlap".into(),
                value: format!("writer-{index}"),
            },
            V2Mutation::PutVector {
                id: vector_id,
                values: vec![f32::from(bounded_index), 1.0, -1.0],
            },
            V2Mutation::PutAudit {
                block: V2AuditBlock {
                    hash: audit_hash,
                    parents: Vec::new(),
                    timestamp_ms: i64::from(bounded_index),
                    sequence: u64::from(bounded_index),
                    profile: "process-stress".into(),
                    query_id: vector_id,
                    response_id: vector_id,
                    metadata: format!("child={index}"),
                },
            },
        ])
        .expect("build isolated committed transaction");
    let mut shared = V2Store::open(&shared_root).expect("open shared target");
    std::fs::write(root.join("ready").join(index.to_string()), b"ready\n")
        .expect("publish readiness");
    wait_for(&root.join("go"), Instant::now() + Duration::from_secs(30));
    shared
        .import_committed(&transaction)
        .expect("import exact committed transaction");
    std::fs::write(
        root.join("committed").join(index.to_string()),
        format!("{}\n", transaction.transaction_id()),
    )
    .expect("publish committed marker");
}

fn spawn_child(binary: &Path, root: &Path, index: usize) -> Child {
    Command::new(binary)
        .args([
            "--exact",
            "child_process_entry",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ROOT_ENV, root)
        .env(INDEX_ENV, index.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn stress child")
}

#[test]
fn fifty_processes_recover_reported_commits_conflicts_and_audit_dag() {
    if std::env::var_os(ROOT_ENV).is_some() {
        return;
    }
    let root = std::env::temp_dir().join(format!("abi-v2-process-stress-{}", Uuid::new_v4()));
    let shared_root = root.join("shared");
    std::fs::create_dir_all(root.join("ready")).unwrap();
    std::fs::create_dir_all(root.join("committed")).unwrap();
    V2Store::open(&shared_root).expect("initialize shared v2 store");

    let binary = std::env::current_exe().expect("test binary");
    let mut children = (0..CHILDREN)
        .map(|index| spawn_child(&binary, &root, index))
        .collect::<Vec<_>>();
    let ready_deadline = Instant::now() + Duration::from_secs(30);
    for index in 0..CHILDREN {
        wait_for(&root.join("ready").join(index.to_string()), ready_deadline);
    }

    let mut compactor = V2Store::open(&shared_root).expect("open compactor");
    std::fs::write(root.join("go"), b"go\n").unwrap();
    for child in children.iter_mut().step_by(10) {
        let _ = child.kill();
    }
    for _ in 0..3 {
        compactor.compact().expect("lock-free immutable compaction");
        std::thread::yield_now();
    }
    for child in &mut children {
        let _ = child.wait();
    }

    let reported = std::fs::read_dir(root.join("committed")).unwrap().count();
    assert!(
        (CHILDREN - 5..=CHILDREN).contains(&reported),
        "unexpected reported commit count: {reported}"
    );
    let recovered = V2Store::open(&shared_root).expect("recover shared v2 store");
    let snapshot = recovered.snapshot();
    let recovered_transactions = snapshot.committed_transactions();
    assert!(recovered_transactions >= reported);
    assert!(recovered_transactions <= CHILDREN);
    assert_eq!(snapshot.vector_count(), recovered_transactions);
    assert_eq!(snapshot.audit_blocks().count(), recovered_transactions);
    snapshot
        .verify_audit_dag()
        .expect("valid concurrent audit DAG");

    let overlap = snapshot.get("overlap").expect("overlapping key survives");
    assert_eq!(overlap.conflicts.len() + 1, recovered_transactions);
    let visible = std::iter::once(overlap.preferred.value.as_str())
        .chain(
            overlap
                .conflicts
                .iter()
                .map(|version| version.value.as_str()),
        )
        .collect::<std::collections::BTreeSet<_>>();
    for entry in std::fs::read_dir(root.join("committed")).unwrap() {
        let index = entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .parse::<usize>()
            .unwrap();
        assert!(visible.contains(format!("writer-{index}").as_str()));
    }
    std::fs::remove_dir_all(root).unwrap();
}
