use super::*;
use crate::{RecordedOutcome, SignedDeliveryPolicy, StreamFrame, TerminalPacket};
use ed25519_dalek::SigningKey;
use std::{
    io::{BufRead as _, Write as _},
    process::{Child, Command, ExitStatus, Stdio},
    sync::mpsc,
    time::{Duration, Instant},
};

#[derive(Serialize, Deserialize)]
struct Config {
    path: std::path::PathBuf,
    page: DeliveryPage,
    checkpoint: String,
}
struct KillChild(Child);
impl Drop for KillChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        if let Err(error) = reap_before(&mut self.0, Instant::now() + Duration::from_secs(5)) {
            eprintln!("child cleanup failed: {error}");
        }
    }
}

fn reap_before(child: &mut Child, deadline: Instant) -> std::io::Result<ExitStatus> {
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if Instant::now() >= deadline {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "child reap deadline exceeded",
            ));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn assert_force_killed(status: ExitStatus) {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        assert_eq!(status.signal(), Some(nix::libc::SIGKILL));
    }
    #[cfg(windows)]
    assert_eq!(status.code(), Some(1)); // Child::kill uses TerminateProcess with exit code 1.
}

fn pause_checkpoint(point: &str) {
    println!("CHECKPOINT:{point}");
    std::io::stdout().flush().unwrap();
    let mut line = String::new();
    assert_eq!(
        std::io::stdin().read_line(&mut line).unwrap(),
        0,
        "parent must kill child"
    );
    panic!("parent pipe closed before kill");
}
fn before(point: &str) {
    if point == "before_commit" {
        pause_checkpoint(point);
    }
}
fn after(point: &str) {
    if point == "after_commit" {
        pause_checkpoint(point);
    }
}

#[test]
fn import_kill_child() {
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).unwrap() == 0 {
        return;
    }
    let config: Config = serde_json::from_str(&line).unwrap();
    let mut store = VerifiedDeliveryStore::open(&config.path).unwrap();
    store.import_checkpoint = Some(if config.checkpoint == "before_commit" {
        before
    } else {
        after
    });
    store
        .import(&config.page, &RelayKey::from_seed(&[8; 32]).unwrap())
        .unwrap();
    panic!("import returned before parent killed child");
}

fn kill_at(path: &Path, page: &DeliveryPage, checkpoint: &str) {
    let mut child = KillChild(
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "delivery_consumer::interruption_tests::import_kill_child",
                "--nocapture",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let mut input = child.0.stdin.take().unwrap();
    writeln!(
        input,
        "{}",
        serde_json::to_string(&Config {
            path: path.into(),
            page: page.clone(),
            checkpoint: checkpoint.into(),
        })
        .unwrap()
    )
    .unwrap();
    input.flush().unwrap();
    let stdout = child.0.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in std::io::BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if let Some(point) = line.strip_prefix("CHECKPOINT:") {
                let _ = tx.send(point.to_owned());
            }
        }
        let _ = done_tx.send(());
    });
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(20)).unwrap(),
        checkpoint
    );
    assert!(
        child.0.try_wait().unwrap().is_none(),
        "child exited at checkpoint"
    );
    child.0.kill().unwrap();
    let status = reap_before(&mut child.0, Instant::now() + Duration::from_secs(5))
        .expect("force-killed child must reap before deadline");
    assert_force_killed(status);
    drop(input);
    done_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("stdout reader completion deadline exceeded");
    drop(reader);
}

fn completed(page: &DeliveryPage, signing: &SigningKey) -> DeliveryPage {
    #[derive(Serialize)]
    struct Canonical<'a> {
        job_id: uuid::Uuid,
        request_digest: &'a str,
        status: abi_worker::JobStatus,
        chunks: &'a [abi_worker::ResultChunk],
        completed_at_unix_ms: i64,
    }
    let job_id = uuid::Uuid::parse_str(&page.receipt.binding.job_id).unwrap();
    let digest = "a".repeat(64);
    let chunks = vec![abi_worker::ResultChunk {
        sequence: 0,
        bytes: b"private-verified-prefix complete".to_vec(),
    }];
    let canonical = serde_json::to_string(&Canonical {
        job_id,
        request_digest: &digest,
        status: abi_worker::JobStatus::Succeeded,
        chunks: &chunks,
        completed_at_unix_ms: 100_000,
    })
    .unwrap();
    let result = abi_worker::JobResult {
        job_id,
        request_digest: abi_worker::ResultDigest::parse(digest).unwrap(),
        status: abi_worker::JobStatus::Succeeded,
        chunks,
        result_digest: abi_worker::ResultDigest::parse({
            use sha2::Digest as _;
            use std::fmt::Write as _;
            let mut hex = String::new();
            for byte in sha2::Sha256::digest(canonical.as_bytes()) {
                write!(&mut hex, "{byte:02x}").unwrap();
            }
            hex
        })
        .unwrap(),
        completed_at_unix_ms: 100_000,
    };
    assert!(result.verify_digest());
    let mut out = page.clone();
    let terminal = OutboxPacket::Terminal {
        packet: TerminalPacket::seal(
            &page.receipt.binding,
            &serde_json::to_vec(&result).unwrap(),
            signing,
        )
        .unwrap(),
    };
    let mut manifest = out.manifest.take().unwrap();
    manifest.outcome = RecordedOutcome::Completed;
    manifest.completed_at = Some(100);
    manifest.terminal_digest = Some(terminal.id(&page.receipt.binding).unwrap().digest);
    out.manifest = Some(manifest.sign(signing).unwrap());
    out.packets = vec![terminal];
    out
}

fn completed_oracle(
    path: &Path,
    prefix: &DeliveryPage,
    terminal: &DeliveryPage,
    key: &RelayKey,
    signing: &SigningKey,
) -> (Vec<u8>, Vec<u8>) {
    let mut reference = VerifiedDeliveryStore::create(
        path,
        &prefix.receipt.binding,
        signing.verifying_key().to_bytes(),
        signing.verifying_key().to_bytes(),
    )
    .unwrap();
    reference.import(prefix, key).unwrap();
    let ack = reference.import(terminal, key).unwrap();
    assert_eq!(ack.contiguous, 1);
    assert_eq!(
        ack.manifest_digest,
        terminal.manifest.as_ref().unwrap().digest().unwrap()
    );
    assert_eq!(
        ack.terminal_digest,
        terminal.manifest.as_ref().unwrap().terminal_digest
    );
    assert!(ack.terminal_digest.is_some());
    assert_eq!(
        reference.output(key).unwrap(),
        b"private-verified-prefix complete"
    );
    (
        encode(&reference.read().unwrap()).unwrap(),
        encode(&ack).unwrap(),
    )
}

fn assert_completed_oracle(
    store: &VerifiedDeliveryStore,
    key: &RelayKey,
    oracle: &(Vec<u8>, Vec<u8>),
) {
    assert_eq!(encode(&store.read().unwrap()).unwrap(), oracle.0);
    assert_eq!(encode(&store.ack().unwrap().unwrap()).unwrap(), oracle.1);
    assert_eq!(
        store.output(key).unwrap(),
        b"private-verified-prefix complete"
    );
}

#[test]
fn parent_kill_import_transaction_preserves_complete_old_or_new_view() {
    for checkpoint in ["before_commit", "after_commit"] {
        let (dir, page, key, signing) = tests::fixture();
        let path = dir.path().join("kill-consumer.sqlite");
        let mut store = VerifiedDeliveryStore::create(
            &path,
            &page.receipt.binding,
            signing.verifying_key().to_bytes(),
            signing.verifying_key().to_bytes(),
        )
        .unwrap();
        let previous = store.import(&page, &key).unwrap();
        let previous_body = encode(&store.read().unwrap()).unwrap();
        let terminal = completed(&page, &signing);
        let oracle = completed_oracle(
            &dir.path().join("reference.sqlite"),
            &page,
            &terminal,
            &key,
            &signing,
        );
        drop(store);
        kill_at(&path, &terminal, checkpoint);
        let mut store = VerifiedDeliveryStore::open(&path).unwrap();
        if checkpoint == "before_commit" {
            assert_eq!(encode(&store.read().unwrap()).unwrap(), previous_body);
            assert_eq!(
                encode(&store.ack().unwrap().unwrap()).unwrap(),
                encode(&previous).unwrap()
            );
            assert_eq!(store.output(&key).unwrap(), b"private-verified-prefix");
        } else {
            assert_completed_oracle(&store, &key, &oracle);
        }
        assert_eq!(
            encode(&store.import(&terminal, &key).unwrap()).unwrap(),
            oracle.1
        );
        assert_completed_oracle(&store, &key, &oracle);
        for _ in 0..3 {
            assert_eq!(
                encode(&store.import(&terminal, &key).unwrap()).unwrap(),
                oracle.1
            );
            assert_completed_oracle(&store, &key, &oracle);
        }
    }
}

#[test]
fn signed_conflicting_duplicate_and_foreign_binding_preserve_journal() {
    let (dir, page, key, signing) = tests::fixture();
    let mut store = VerifiedDeliveryStore::create(
        &dir.path().join("conflicts.sqlite"),
        &page.receipt.binding,
        signing.verifying_key().to_bytes(),
        signing.verifying_key().to_bytes(),
    )
    .unwrap();
    store.import(&page, &key).unwrap();
    let body = encode(&store.read().unwrap()).unwrap();
    let mut conflict = page.clone();
    conflict.packets = vec![OutboxPacket::Delta {
        frame: StreamFrame::seal(
            &page.receipt.binding,
            0,
            b"validly signed conflicting duplicate",
            &signing,
        )
        .unwrap(),
    }];
    let mut manifest = conflict.manifest.take().unwrap();
    manifest.frame_digests = vec![
        conflict.packets[0]
            .id(&page.receipt.binding)
            .unwrap()
            .digest,
    ];
    conflict.manifest = Some(manifest.sign(&signing).unwrap());
    assert!(store.import(&conflict, &key).is_err());
    for owner in [false, true] {
        let mut foreign = page.clone();
        if owner {
            foreign.receipt.binding.account_id = "foreign-owner".into();
        } else {
            foreign.receipt.binding.job_id = uuid::Uuid::new_v4().to_string();
        }
        foreign.packets = vec![OutboxPacket::Delta {
            frame: StreamFrame::seal(
                &foreign.receipt.binding,
                0,
                b"private-verified-prefix",
                &signing,
            )
            .unwrap(),
        }];
        let mut manifest = foreign.manifest.take().unwrap();
        manifest.binding = foreign.receipt.binding.clone();
        manifest.frame_digests = vec![
            foreign.packets[0]
                .id(&foreign.receipt.binding)
                .unwrap()
                .digest,
        ];
        foreign.manifest = Some(manifest.sign(&signing).unwrap());
        foreign.policy.policy.binding_digest =
            crate::authority::hash(&crate::enrollment::json(&foreign.receipt.binding).unwrap());
        foreign.policy =
            SignedDeliveryPolicy::sign(foreign.policy.policy.clone(), &signing).unwrap();
        let mut matching = VerifiedDeliveryStore::create(
            &dir.path().join(format!("foreign-{owner}.sqlite")),
            &foreign.receipt.binding,
            signing.verifying_key().to_bytes(),
            signing.verifying_key().to_bytes(),
        )
        .unwrap();
        assert_eq!(matching.import(&foreign, &key).unwrap().contiguous, 1);
        assert_eq!(matching.output(&key).unwrap(), b"private-verified-prefix");
        assert!(store.import(&foreign, &key).is_err());
        assert_eq!(encode(&store.read().unwrap()).unwrap(), body);
    }
    assert_eq!(encode(&store.read().unwrap()).unwrap(), body);
}
