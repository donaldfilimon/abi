use super::*;
fn submission() -> Submission {
    Submission {
        binding: JobBinding {
            protocol: "abi.relay-job.v1".into(),
            audience: "https://abi.example".into(),
            job_id: "00000000-0000-4000-8000-000000000001".into(),
            account_id: "account".into(),
            tenant_id: "tenant".into(),
            node_id: "node".into(),
            capability: "model.infer".into(),
            expires_at: 200,
            submitted_at: 100,
            max_result_bytes: 4096,
            reply_key: [1; 32],
        },
        payload: SealedPayload {
            encapsulated_key: vec![2; 32],
            ciphertext: vec![3; 16],
        },
    }
}
#[test]
fn version_one_aad_and_dispatch_signing_bytes_remain_frozen() {
    let s = submission();
    let aad = s.binding.aad("request").unwrap();
    assert_eq!(
        hash(std::str::from_utf8(&aad).unwrap()),
        "fu-7TujhlERuhA9NMvHME9jZvbFIlbpBW1DZ8re27Ag"
    );
    let d = crate::Dispatch {
        submission: s,
        signature: vec![],
    };
    assert_eq!(
        hash(std::str::from_utf8(&d.bytes().unwrap()).unwrap()),
        "3D9XRZmvD0WnyH_rB5Q6a3Xxbit8jtyP0CMYhI0ufSE"
    );
}
#[test]
fn replay_grant_round_trip_cannot_become_execution_permit_and_expires_exclusively() {
    let s = submission();
    let key = SigningKey::from_bytes(&[4; 32]);
    let mut grant = ReplayGrant {
        binding: s.binding.clone(),
        policy: DeliveryPolicy::for_submission(&s, 86500).unwrap(),
        manifest_digest: "manifest".into(),
        epoch: "one".into(),
        packets: vec![],
        cancelled: false,
        issued_at: 210,
        expires_at: 240,
        signature: vec![],
    };
    grant.signature = key.sign(&grant.bytes().unwrap()).to_bytes().to_vec();
    let encoded = serde_json::to_vec(&grant).unwrap();
    let recovered: ReplayGrant = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(serde_json::to_vec(&recovered).unwrap(), encoded);
    recovered
        .verify(&key.verifying_key().to_bytes(), 239)
        .unwrap();
    assert!(
        recovered
            .verify(&key.verifying_key().to_bytes(), 209)
            .is_err()
    );
    assert!(
        recovered
            .verify(&key.verifying_key().to_bytes(), 240)
            .is_err()
    );
    assert!(serde_json::from_slice::<crate::DispatchPermit>(&encoded).is_err());
    let mut changed = recovered;
    changed.epoch = "two".into();
    assert!(
        changed
            .verify(&key.verifying_key().to_bytes(), 211)
            .is_err()
    );
}
#[test]
fn policy_retention_is_immutable_and_bounded_by_seven_days() {
    let s = submission();
    for until in [159, 100 + 7 * 86400 + 1] {
        assert!(
            DeliveryPolicy::for_submission(&s, until)
                .unwrap()
                .validate(&s)
                .is_err()
        );
    }
    for until in [160, 86500, 100 + 7 * 86400] {
        DeliveryPolicy::for_submission(&s, until)
            .unwrap()
            .validate(&s)
            .unwrap();
    }
    let mut policy = DeliveryPolicy::for_submission(&s, 86500).unwrap();
    policy.submission_digest.push('x');
    assert!(policy.validate(&s).is_err());
}
