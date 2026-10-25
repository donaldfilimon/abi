use super::*;
use crate::StreamFrame;
fn start(f: &mut Fixture, s: Submission) -> String {
    let id = s.binding.job_id.clone();
    f.relay.submit(&mut f.authority, &f.alice, s, 101).unwrap();
    for action in [NodeAction::Poll, NodeAction::Inspect { job_id: id.clone() }] {
        let msg = NodeMessage::sign(&f.node_a, &f.key_a, action, 102).unwrap();
        f.relay.node(&mut f.authority, &msg, 102).unwrap();
    }
    id
}
fn frame(f: &mut Fixture, id: &str, frame: StreamFrame) -> Result<NodeReply> {
    let msg = NodeMessage::sign(
        &f.node_a,
        &f.key_a,
        NodeAction::Stream {
            job_id: id.into(),
            frame,
        },
        103,
    )
    .unwrap();
    f.relay.node(&mut f.authority, &msg, 103)
}
#[test]
fn streaming_frames_are_ordered_private_idempotent_and_do_not_resume_after_restart() {
    let mut f = fixture();
    let s = submission(&f, &f.alice, &f.node_a);
    let b = s.binding.clone();
    let id = start(&mut f, s);
    let one = StreamFrame::seal(&b, 1, b"next-private-delta", &f.key_a).unwrap();
    assert!(frame(&mut f, &id, one.clone()).is_err());
    let zero = StreamFrame::seal(&b, 0, b"first-private-delta", &f.key_a).unwrap();
    frame(&mut f, &id, zero.clone()).unwrap();
    frame(&mut f, &id, zero.clone()).unwrap();
    let changed = StreamFrame::seal(&b, 0, b"changed-delta", &f.key_a).unwrap();
    assert!(frame(&mut f, &id, changed).is_err());
    frame(&mut f, &id, one.clone()).unwrap();
    let view = f.relay.view(&mut f.authority, &f.alice, &id, 103).unwrap();
    assert_eq!(view.frames.len(), 2);
    assert_eq!(
        view.frames[0]
            .open(&b, &f.node_a.claim.public_key, &f.reply)
            .unwrap(),
        b"first-private-delta"
    );
    assert!(
        view.frames[0]
            .open(&b, &f.node_a.claim.public_key, &f.reply_b)
            .is_err()
    );
    let mut bad = zero.clone();
    bad.sequence = 1;
    assert!(bad.open(&b, &f.node_a.claim.public_key, &f.reply).is_err());
    assert!(f.relay.view(&mut f.authority, &f.bob, &id, 103).is_err());
    let database = std::fs::read(f.dir.path().join("authority.sqlite")).unwrap();
    assert!(!database.windows(19).any(|w| w == b"first-private-delta"));
    assert!(
        !database
            .windows(16)
            .any(|w| w == &zero.payload.ciphertext[..16])
    );
    assert_eq!(
        f.relay
            .cancel(&mut f.authority, &f.alice, &id, 103)
            .unwrap()
            .state,
        "cancel_requested"
    );
    drop(f.relay);
    f.relay = Relay::open(&mut f.authority, RelayLimits::default(), 104).unwrap();
    let view = f.relay.view(&mut f.authority, &f.alice, &id, 104).unwrap();
    assert_eq!(view.receipt.state, "unresolved");
    assert!(view.frames.is_empty());
    let two = StreamFrame::seal(&b, 2, b"late", &f.key_a).unwrap();
    assert!(frame(&mut f, &id, two).is_err());
}
#[test]
fn dispatched_cancellation_requires_worker_acknowledgement_and_retains_capacity() {
    let mut f = fixture();
    f.relay.limits.active_jobs = 1;
    let s = submission(&f, &f.alice, &f.node_a);
    let b = s.binding.clone();
    let id = start(&mut f, s);
    assert!(f.relay.cancel(&mut f.authority, &f.bob, &id, 103).is_err());
    assert_eq!(
        f.relay
            .cancel(&mut f.authority, &f.alice, &id, 103)
            .unwrap()
            .state,
        "cancel_requested"
    );
    let delta = StreamFrame::seal(&b, 0, b"after-stop-request", &f.key_a).unwrap();
    assert!(frame(&mut f, &id, delta).is_err());
    let other = submission(&f, &f.alice, &f.node_a);
    f.relay
        .submit(&mut f.authority, &f.alice, other, 103)
        .unwrap();
    let poll = NodeMessage::sign(&f.node_a, &f.key_a, NodeAction::Poll, 103).unwrap();
    assert!(
        f.relay
            .node(&mut f.authority, &poll, 103)
            .unwrap()
            .dispatch
            .is_none()
    );
    let wrong = NodeMessage::sign(
        &f.node_b,
        &f.key_b,
        NodeAction::Stopped { job_id: id.clone() },
        103,
    )
    .unwrap();
    assert!(f.relay.node(&mut f.authority, &wrong, 103).is_err());
    for _ in 0..2 {
        let ack = NodeMessage::sign(
            &f.node_a,
            &f.key_a,
            NodeAction::Stopped { job_id: id.clone() },
            103,
        )
        .unwrap();
        assert_eq!(
            f.relay
                .node(&mut f.authority, &ack, 103)
                .unwrap()
                .receipt
                .unwrap()
                .state,
            "cancelled"
        );
    }
    let poll = NodeMessage::sign(&f.node_a, &f.key_a, NodeAction::Poll, 104).unwrap();
    assert!(
        f.relay
            .node(&mut f.authority, &poll, 104)
            .unwrap()
            .dispatch
            .is_some()
    );
}
#[test]
fn stream_frame_and_memory_limits_refuse_before_persisting_commitments() {
    let mut f = fixture();
    let s = submission(&f, &f.alice, &f.node_a);
    let b = s.binding.clone();
    let id = start(&mut f, s);
    // Minimum global memory budget, already charged by other buffers, rejects a valid frame.
    f.relay.limits.memory_bytes = 1;
    let zero = StreamFrame::seal(&b, 0, b"private", &f.key_a).unwrap();
    assert!(matches!(frame(&mut f, &id, zero.clone()), Err(Error::Full)));
    f.relay.limits.memory_bytes = 16 * 1024 * 1024;
    frame(&mut f, &id, zero).unwrap();
    for sequence in 1..16 {
        let next = StreamFrame::seal(&b, sequence, b"x", &f.key_a).unwrap();
        frame(&mut f, &id, next).unwrap();
    }
    assert!(StreamFrame::seal(&b, 16, b"extra", &f.key_a).is_err());
    assert_eq!(
        f.relay
            .view(&mut f.authority, &f.alice, &id, 103)
            .unwrap()
            .frames
            .len(),
        16
    );
}
