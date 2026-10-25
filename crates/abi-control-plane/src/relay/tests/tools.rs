use super::*;
use crate::{DeliveryPolicy, DurableSubmission};

fn tool_node(f: &mut Fixture, alice: bool) -> NodeCertificate {
    let (account, key, enc) = if alice {
        (&f.alice, &f.key_a, &f.enc_a)
    } else {
        (&f.bob, &f.key_b, &f.enc_b)
    };
    let ticket = f
        .authority
        .begin_pairing(
            &PairRequest {
                public_key: key.verifying_key().to_bytes(),
                encryption_public_key: Some(enc.public_key()),
                capabilities: vec!["tool.execute".into()],
            },
            100,
        )
        .unwrap();
    let claim = f
        .authority
        .claim_pairing(account, &ticket.user_code, 100)
        .unwrap();
    f.authority
        .confirm_pairing(
            &ticket.enrollment_id,
            &ticket.poll_secret,
            &key.sign(&claim.signing_bytes().unwrap()).to_bytes(),
            100,
        )
        .unwrap()
}
fn tool_submission(f: &Fixture, alice: bool, node: &NodeCertificate) -> Submission {
    let mut s = submission(f, if alice { &f.alice } else { &f.bob }, node);
    s.binding.capability = "tool.execute".into();
    s.payload = seal(
        &node.claim.encryption_public_key.unwrap(),
        &s.binding.aad("request").unwrap(),
        b"opaque-governed-tool-canary",
    )
    .unwrap();
    s
}
fn admit(f: &mut Fixture, alice: bool, s: Submission) -> Result<JobReceipt> {
    let delivery = DeliveryPolicy::for_submission(&s, 86500).unwrap();
    f.relay.submit_durable(
        &mut f.authority,
        if alice { &f.alice } else { &f.bob },
        DurableSubmission {
            submission: s,
            delivery,
        },
        101,
    )
}
#[test]
fn tool_jobs_require_durable_delivery_and_exact_enrolled_capability() {
    let mut f = fixture();
    let forged = tool_submission(&f, true, &f.node_a);
    assert!(admit(&mut f, true, forged).is_err());
    let node = tool_node(&mut f, true);
    let s = tool_submission(&f, true, &node);
    assert!(
        f.relay
            .submit(&mut f.authority, &f.alice, s.clone(), 101)
            .is_err()
    );
    let model = submission(&f, &f.alice, &node);
    assert!(admit(&mut f, true, model).is_err());
    let mut wrong_tenant = s.clone();
    wrong_tenant.binding.tenant_id = f.bob.tenant_id.clone();
    assert!(admit(&mut f, true, wrong_tenant).is_err());
    admit(&mut f, true, s).unwrap();
}
#[test]
fn two_tenants_receive_only_their_encrypted_tool_dispatch_and_bound_permit() {
    let mut f = fixture();
    let a = tool_node(&mut f, true);
    let b = tool_node(&mut f, false);
    let sa = tool_submission(&f, true, &a);
    let sb = tool_submission(&f, false, &b);
    admit(&mut f, true, sa.clone()).unwrap();
    admit(&mut f, false, sb.clone()).unwrap();
    for (node, key, enc, other, s) in [
        (&a, &f.key_a, &f.enc_a, &f.enc_b, &sa),
        (&b, &f.key_b, &f.enc_b, &f.enc_a, &sb),
    ] {
        let request = NodeMessage::sign(node, key, NodeAction::Poll, 102).unwrap();
        let dispatch = f
            .relay
            .node(&mut f.authority, &request, 102)
            .unwrap()
            .dispatch
            .unwrap();
        dispatch.verify(&f.authority.public_key()).unwrap();
        assert_eq!(json(&dispatch.submission).unwrap(), json(s).unwrap());
        assert_eq!(
            enc.open(&s.binding.aad("request").unwrap(), &s.payload)
                .unwrap(),
            b"opaque-governed-tool-canary"
        );
        assert!(
            other
                .open(&s.binding.aad("request").unwrap(), &s.payload)
                .is_err()
        );
        let inspect = NodeMessage::sign(
            node,
            key,
            NodeAction::Inspect {
                job_id: s.binding.job_id.clone(),
            },
            103,
        )
        .unwrap();
        let permit = f
            .relay
            .node(&mut f.authority, &inspect, 103)
            .unwrap()
            .permit
            .unwrap();
        assert_eq!(permit.binding, s.binding);
        assert_eq!(permit.binding.capability, "tool.execute");
    }
    let bytes = std::fs::read(f.dir.path().join("authority.sqlite")).unwrap();
    assert!(
        !bytes
            .windows(b"opaque-governed-tool-canary".len())
            .any(|w| w == b"opaque-governed-tool-canary")
    );
}
#[test]
fn revoked_tool_node_cannot_inspect_an_offered_job() {
    let mut f = fixture();
    let node = tool_node(&mut f, true);
    let s = tool_submission(&f, true, &node);
    admit(&mut f, true, s.clone()).unwrap();
    let poll = NodeMessage::sign(&node, &f.key_a, NodeAction::Poll, 102).unwrap();
    assert!(
        f.relay
            .node(&mut f.authority, &poll, 102)
            .unwrap()
            .dispatch
            .is_some()
    );
    f.authority.revoke(&f.alice, &node.node_id, 103).unwrap();
    let inspect = NodeMessage::sign(
        &node,
        &f.key_a,
        NodeAction::Inspect {
            job_id: s.binding.job_id,
        },
        104,
    )
    .unwrap();
    assert!(f.relay.node(&mut f.authority, &inspect, 104).is_err());
}
#[test]
fn tool_inspection_refreshes_short_permit_after_human_delay_without_requeue() {
    let mut f = fixture();
    let node = tool_node(&mut f, true);
    let s = tool_submission(&f, true, &node);
    let id = s.binding.job_id.clone();
    admit(&mut f, true, s).unwrap();
    let poll = NodeMessage::sign(&node, &f.key_a, NodeAction::Poll, 102).unwrap();
    f.relay.node(&mut f.authority, &poll, 102).unwrap();
    let inspect = |f: &mut Fixture, time| {
        let request = NodeMessage::sign(
            &node,
            &f.key_a,
            NodeAction::Inspect { job_id: id.clone() },
            time,
        )
        .unwrap();
        f.relay.node(&mut f.authority, &request, time)
    };
    let first = inspect(&mut f, 103).unwrap().permit.unwrap();
    assert_eq!(first.expires_at, 108);
    let fresh = inspect(&mut f, 120).unwrap().permit.unwrap();
    assert_eq!(fresh.binding, first.binding);
    assert_eq!(fresh.submission_digest, first.submission_digest);
    assert_eq!(fresh.issued_at, 120);
    fresh.verify(&f.authority.public_key(), 120).unwrap();
    let poll = NodeMessage::sign(&node, &f.key_a, NodeAction::Poll, 121).unwrap();
    assert!(
        f.relay
            .node(&mut f.authority, &poll, 121)
            .unwrap()
            .dispatch
            .is_none()
    );
    f.relay
        .cancel(&mut f.authority, &f.alice, &id, 122)
        .unwrap();
    assert!(inspect(&mut f, 123).unwrap().permit.is_none());
    f.authority.revoke(&f.alice, &node.node_id, 124).unwrap();
    assert!(inspect(&mut f, 125).is_err());
}
