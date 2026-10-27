//! Dispatch rechecks node credential validity after the authority callback.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

#[test]
fn node_certificate_expiring_during_authority_recheck_never_dispatches() {
    let mut f = fixture();
    assert_eq!(f.node_a.expires_at, 3700); // enrolled at 100, exclusive expiry
    let s = submission_at(&f, &f.alice, &f.node_a, 3695, 3800);
    let id = s.binding.job_id.clone();
    f.relay.submit(&mut f.authority, &f.alice, s, 3696).unwrap();
    let dispatch = poll(&mut f, true, 3697);
    let permit = permit(&mut f, true, &id, 3698);
    assert!(permit.expires_at > f.node_a.expires_at); // permit outlives the certificate
    let mut runtime = crate::RelayRuntime::new(
        f.node_a.clone(),
        f.key_a.clone(),
        RelayKey::from_seed(&[101; 32]).unwrap(),
        f.authority.public_key(),
        "local-echo-v1",
        abi_worker::MemoryControlStore::new(),
        3698,
    )
    .unwrap();
    let provider = CountProvider(AtomicUsize::new(0));
    // Clock reads: pre-check, pre-admission, admitted_at, started_at (authority time), dispatch_at, completed_at.
    let mut ticks = [3698, 3698, 3698, 3700, 3700, 3700].into_iter();
    assert!(
        runtime
            .execute(&dispatch, &permit, &provider, || ticks
                .next()
                .unwrap_or(3700))
            .is_err()
    );
    assert_eq!(provider.0.load(SeqCst), 0);
    // The durable admission stays unresolved; an exact retry never dispatches.
    assert!(matches!(
        runtime.execute(&dispatch, &permit, &provider, || 3699),
        Ok(None)
    ));
    assert_eq!(provider.0.load(SeqCst), 0);
}

#[test]
fn node_certificate_expiring_between_authority_callback_and_dispatch_never_dispatches() {
    let mut f = fixture();
    let s = submission_at(&f, &f.alice, &f.node_a, 3695, 3800);
    let id = s.binding.job_id.clone();
    f.relay.submit(&mut f.authority, &f.alice, s, 3696).unwrap();
    let dispatch = poll(&mut f, true, 3697);
    let permit = permit(&mut f, true, &id, 3698);
    let mut runtime = crate::RelayRuntime::new(
        f.node_a.clone(),
        f.key_a.clone(),
        RelayKey::from_seed(&[101; 32]).unwrap(),
        f.authority.public_key(),
        "local-echo-v1",
        abi_worker::MemoryControlStore::new(),
        3698,
    )
    .unwrap();
    let provider = CountProvider(AtomicUsize::new(0));
    // The certificate is still valid at the authority callback (started_at = 3699)
    // and expires at the worker's dispatch-time read (dispatch_at = 3700).
    let mut ticks = [3698, 3698, 3698, 3699, 3700, 3700].into_iter();
    assert!(
        runtime
            .execute(&dispatch, &permit, &provider, || ticks
                .next()
                .unwrap_or(3700))
            .is_err()
    );
    assert_eq!(provider.0.load(SeqCst), 0);
    assert!(matches!(
        runtime.execute(&dispatch, &permit, &provider, || 3699),
        Ok(None)
    ));
    assert_eq!(provider.0.load(SeqCst), 0);
}
