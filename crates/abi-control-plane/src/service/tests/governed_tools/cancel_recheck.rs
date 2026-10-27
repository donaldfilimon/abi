//! Cancellation raised inside the last pre-adapter authority refresh never reaches
//! the adapter; a cancellation in the post-effect refresh stays unresolved.
//!
//! Stage order (`abi-agent-host` `governed.rs`): the final refresh before
//! `dispatch.execute` is the `facts.observe` in the second `live_authority_valid`
//! (after stage 12); the next refresh is the post-effect observation. The test
//! locates both at runtime from whether the effect file exists yet, not from a
//! hard-coded call number.
use super::*;
use std::cell::{Cell, RefCell};

type Outcome = crate::Result<crate::GovernedToolOutcome>;

struct Run {
    outcome: Outcome,
    effects: usize,
    /// For each refresh in the approved call: had the adapter effect happened yet?
    after_effect: Vec<bool>,
    retry: Outcome,
}

/// Reach an approved preview, then run the approved call, cancelling inside
/// refresh number `cancel_at` (1-based), then retry once with a fresh token.
async fn approved(cancel_at: Option<usize>) -> Run {
    let f = fixture().await;
    let mut c = case(&f);
    let ms = u64::try_from(c.dispatch.submission.binding.submitted_at).unwrap() * 1000;
    let clock = FixedClock::new(ms);
    let (sources, registration, mut facts) = governance(&c.cert, ms);
    let mut approvals = Local::new(f.dir.path().join("local-approval.json"));
    let effect_path = f.dir.path().join("cancel-effect.txt");
    let mut executor = FileAdapter {
        path: effect_path.clone(),
        effects: 0,
        invocations: 0,
    };
    let effects = vec!["Create cancel-effect.txt in the pinned scratch directory".into()];
    let invoke = |c: &mut Case,
                  approvals: &mut Local,
                  executor: &mut FileAdapter,
                  facts: &mut Facts,
                  refresh: &mut dyn FnMut() -> crate::Result<crate::DispatchPermit>,
                  cancel: &CancellationToken| {
        c.runtime.execute_governed(
            &c.dispatch,
            &c.policy,
            &mut crate::GovernedToolContext {
                user_id: "local-user",
                resource: "scratch/cancel-effect.txt",
                expected_effects: &effects,
                sources: &sources,
                registration: registration.clone(),
                facts,
                approvals,
                executor,
                clock: &clock,
                refresh,
            },
            cancel,
        )
    };
    let mut pending_refresh = refresh(&f, &c);
    let pending = invoke(
        &mut c,
        &mut approvals,
        &mut executor,
        &mut facts,
        &mut pending_refresh,
        &CancellationToken::new(),
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
    let token = CancellationToken::new();
    let calls = Cell::new(0usize);
    let after_effect = RefCell::new(Vec::new());
    let mut inner = refresh(&f, &c);
    let mut counted = || {
        calls.set(calls.get() + 1);
        after_effect.borrow_mut().push(effect_path.exists());
        let permit = inner();
        if Some(calls.get()) == cancel_at {
            token.cancel();
        }
        permit
    };
    let outcome = invoke(
        &mut c,
        &mut approvals,
        &mut executor,
        &mut facts,
        &mut counted,
        &token,
    );
    let mut retry_refresh = refresh(&f, &c);
    let retry = invoke(
        &mut c,
        &mut approvals,
        &mut executor,
        &mut facts,
        &mut retry_refresh,
        &CancellationToken::new(),
    );
    Run {
        outcome,
        effects: executor.effects,
        after_effect: after_effect.into_inner(),
        retry,
    }
}

fn succeeded(outcome: &Outcome) -> bool {
    matches!(outcome, Ok(crate::GovernedToolOutcome::Recorded(r)) if r.status == abi_worker::JobStatus::Succeeded)
}

#[tokio::test]
async fn governed_cancellation_during_final_authority_refresh_never_reaches_adapter() {
    // Calibration: locate the last pre-adapter refresh and the post-effect refresh.
    let calibration = approved(None).await;
    assert!(succeeded(&calibration.outcome), "{:?}", calibration.outcome);
    assert_eq!(calibration.effects, 1);
    let pre = calibration
        .after_effect
        .iter()
        .rposition(|after| !after)
        .expect("a refresh precedes the adapter")
        + 1;
    let post = calibration
        .after_effect
        .iter()
        .position(|after| *after)
        .expect("a refresh follows the adapter effect")
        + 1;
    assert_eq!(post, pre + 1, "{:?}", calibration.after_effect);

    // Cancel inside the last pre-adapter refresh: no effect, no success, ever.
    let cancelled = approved(Some(pre)).await;
    assert_eq!(cancelled.effects, 0, "{:?}", cancelled.outcome);
    assert!(
        match &cancelled.outcome {
            Err(_) | Ok(crate::GovernedToolOutcome::Unresolved) => true,
            Ok(crate::GovernedToolOutcome::Recorded(r)) =>
                r.status != abi_worker::JobStatus::Succeeded,
            Ok(crate::GovernedToolOutcome::Pending(_)) => false,
        },
        "{:?}",
        cancelled.outcome
    );
    assert!(!succeeded(&cancelled.retry), "{:?}", cancelled.retry);

    // Cancel inside the post-effect refresh: the effect already happened once by
    // design, so the outcome is unresolved and is never repeated or reported success.
    let late = approved(Some(post)).await;
    assert_eq!(late.effects, 1);
    assert!(
        matches!(late.outcome, Ok(crate::GovernedToolOutcome::Unresolved)),
        "{:?}",
        late.outcome
    );
    assert!(!succeeded(&late.retry), "{:?}", late.retry);
}
