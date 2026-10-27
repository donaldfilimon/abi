//! Governed tool effects recheck the local node certificate at each authority refresh.
//!
//! The certificate is valid when the approved call starts and expires (exclusive
//! `expires_at`) at a refresh before the adapter, while the refreshed permit and the
//! local approval stay valid. Only the node certificate can refuse the effect.
use super::*;
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU64, Ordering};

type Outcome = crate::Result<crate::GovernedToolOutcome>;

struct StepClock(AtomicU64);
impl Clock for StepClock {
    fn now_ms(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

struct Run {
    outcome: Outcome,
    effects: usize,
    /// For each refresh in the approved call: had the adapter effect happened yet?
    after_effect: Vec<bool>,
}

/// Approve a tool job submitted two seconds before certificate expiry `E`, then run
/// the approved call. When `expire_at` is set, the trusted clock moves from
/// `E - 0.4 s` to exactly `E` inside that refresh (1-based). The refreshed permit is
/// issued at `E - 1` and stays valid until `E + 4`.
async fn approved(expire_at: Option<usize>) -> Run {
    let f = fixture().await;
    let mut c = case_timed(&f, None, |cert| cert.expires_at - 2);
    let expiry = u64::try_from(c.cert.expires_at).unwrap() * 1000;
    let ms = expiry - 400;
    let clock = StepClock(AtomicU64::new(ms));
    let (sources, registration, mut facts) = governance(&c.cert, ms);
    let mut approvals = Local::new(f.dir.path().join("local-approval.json"));
    let effect_path = f.dir.path().join("cert-effect.txt");
    let mut executor = FileAdapter {
        path: effect_path.clone(),
        effects: 0,
        invocations: 0,
    };
    let effects = vec!["Create cert-effect.txt in the pinned scratch directory".into()];
    let invoke =
        |c: &mut Case,
         approvals: &mut Local,
         executor: &mut FileAdapter,
         facts: &mut Facts,
         refresh: &mut dyn FnMut() -> crate::Result<crate::DispatchPermit>| {
            c.runtime.execute_governed(
                &c.dispatch,
                &c.policy,
                &mut crate::GovernedToolContext {
                    user_id: "local-user",
                    resource: "scratch/cert-effect.txt",
                    expected_effects: &effects,
                    sources: &sources,
                    registration: registration.clone(),
                    facts,
                    approvals,
                    executor,
                    clock: &clock,
                    refresh,
                },
                &CancellationToken::new(),
            )
        };
    let issued = c.cert.expires_at - 1;
    let mut pending_refresh = refresh_at(&f, &c, issued);
    let pending = invoke(
        &mut c,
        &mut approvals,
        &mut executor,
        &mut facts,
        &mut pending_refresh,
    )
    .unwrap();
    assert!(matches!(pending, crate::GovernedToolOutcome::Pending(_)));
    let preview = approvals.preview();
    approvals
        .decide_local_approval(
            &preview.request.call_id,
            preview.digest(),
            "human-decision",
            true,
            ms,
        )
        .unwrap();
    let calls = Cell::new(0usize);
    let after_effect = RefCell::new(Vec::new());
    let mut inner = refresh_at(&f, &c, issued);
    let mut counted = || {
        calls.set(calls.get() + 1);
        after_effect.borrow_mut().push(effect_path.exists());
        if Some(calls.get()) == expire_at {
            clock.0.store(expiry, Ordering::SeqCst);
        }
        inner()
    };
    let outcome = invoke(
        &mut c,
        &mut approvals,
        &mut executor,
        &mut facts,
        &mut counted,
    );
    Run {
        outcome,
        effects: executor.effects,
        after_effect: after_effect.into_inner(),
    }
}

#[tokio::test]
async fn node_certificate_expiring_after_approval_never_reaches_tool_adapter() {
    // Calibration: with the certificate valid throughout, the approved effect runs once.
    let calibration = approved(None).await;
    assert!(
        matches!(calibration.outcome, Ok(crate::GovernedToolOutcome::Recorded(ref r)) if r.status == abi_worker::JobStatus::Succeeded),
        "{:?}",
        calibration.outcome
    );
    assert_eq!(calibration.effects, 1);
    let last_pre_adapter = calibration
        .after_effect
        .iter()
        .rposition(|after| !after)
        .expect("a refresh precedes the adapter")
        + 1;
    // Expiry at the first refresh after approval (before durable admission): refused.
    let early = approved(Some(1)).await;
    assert_eq!(early.effects, 0, "{:?}", early.outcome);
    assert!(early.outcome.is_err(), "{:?}", early.outcome);
    // Expiry at the last refresh before the adapter (after admission): no effect, no success.
    let late = approved(Some(last_pre_adapter)).await;
    assert_eq!(late.effects, 0, "{:?}", late.outcome);
    assert!(
        match &late.outcome {
            Err(_) | Ok(crate::GovernedToolOutcome::Unresolved) => true,
            Ok(crate::GovernedToolOutcome::Recorded(r)) =>
                r.status != abi_worker::JobStatus::Succeeded,
            Ok(crate::GovernedToolOutcome::Pending(_)) => false,
        },
        "{:?}",
        late.outcome
    );
}
