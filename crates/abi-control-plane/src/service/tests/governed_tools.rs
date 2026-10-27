//! Full encrypted dispatch to a local approval owner and scratch filesystem adapter.
use super::*;
use abi_agent_host::{local_approval::*, *};
use abi_agent_runtime::{CancellationToken, ToolCall};
use abi_capability::{lifecycle::*, source::*, *};
use abi_worker::{FileControlStore, FileControlStoreLimits};
use std::{collections::BTreeSet, path::PathBuf};
mod cancel_recheck;
mod support;
use serde_json::json;
use support::Local;
fn source() -> PackageSource {
    serde_json::from_value(json!({
        "id":"local.observe", "version":"1.0.0",
        "input_schema":{"type":"object","additionalProperties":false},
        "output_schema":{"type":"boolean"},"error_schema":{"type":"null"},
        "streaming":"None","cancellation":"BeforeEffect", "required_platform_permissions":[],
        "resource_scope":{"organization_id":"org","deployment_id":"deploy","tenant_id":"tenant","platform":"local","guild_ref":null,"resource_digest":Digest::from_bytes([1;32]),"subject_digest":Digest::from_bytes([2;32])},
        "side_effect":"LocalState","reversibility":"ReversibleWithLoss","risk_class":"Low",
        "data_classes":["Operational"],"residency":"LocalOnly","retention":"Session",
        "preconditions":["fresh"],"invariants":["scoped"],"deadline_ms":1000,"rate_class":"bounded",
        "budgets":{"max_input_bytes":512,"max_output_bytes":512,"max_events":4,"max_calls":1},
        "confirmation_policy":"A0None","delegation_policy":"Forbidden","expiry_policy_ms":1000,"revocation_policy":"RecheckBeforeEffect","approval_policy":"A0None",
        "idempotency":"RequiresKey","compensation":null,"rollback":"None","postconditions":["observed"],
        "adapter_bindings":[{"adapter_id":"recording","api_version":"1"}],
        "receipt_requirements":{"redacted":true,"require_postcondition_evidence":true},
        "fixtures":{"deterministic":Digest::from_bytes([3;32]),"adversarial":Digest::from_bytes([4;32]),"failure":Digest::from_bytes([5;32])},
        "promotion_thresholds":{"minimum_passed":1,"maximum_failed":0},
        "expiration_criteria":{"expires_at_ms":10000,"disable_on_schema_drift":true,"disable_on_missing_evidence":true}
    })).unwrap()
}
struct Facts(GovernedFacts);
impl GovernedFactProvider for Facts {
    fn observe(
        &mut self,
        _: &abi_capability::Request,
    ) -> std::result::Result<GovernedFacts, LifecycleError> {
        Ok(self.0.clone())
    }
    fn postconditions(
        &mut self,
        request: &abi_capability::Request,
        required: &BTreeSet<String>,
    ) -> std::result::Result<Vec<PostconditionObservation>, LifecycleError> {
        Ok(required
            .iter()
            .map(|predicate| PostconditionObservation {
                predicate: predicate.clone(),
                request_digest: request.request_digest,
                observed_at_ms: self.0.observed_at_ms,
                expires_at_ms: self.0.expires_at_ms,
                evidence_digest: Some(Digest::from_bytes([7; 32])),
                source: ObservationSource::RecordingFixture,
                status: Postconditions::Verified,
            })
            .collect())
    }
}
struct FileAdapter {
    path: PathBuf,
    effects: usize,
    invocations: usize,
}
impl GovernedLiveExecutor for FileAdapter {
    fn binding(&self) -> AdapterBinding {
        AdapterBinding {
            adapter_id: "recording".into(),
            api_version: "1".into(),
        }
    }
    fn execute(
        &mut self,
        _: &ToolCall,
        _: &abi_capability::Request,
        _: &CompiledSource,
        _: &PreparedExecution,
        _: ToolExecutionContext<'_>,
    ) -> std::result::Result<ToolOutput, ToolExecutionError> {
        use std::io::Write as _;
        self.invocations += 1;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.path)
            .map_err(|_| ToolExecutionError::new("create_failed"))?;
        file.write_all(b"approved-scratch-effect")
            .map_err(|_| ToolExecutionError::new("write_failed"))?;
        file.sync_all()
            .map_err(|_| ToolExecutionError::new("sync_failed"))?;
        self.effects += 1;
        Ok(ToolOutput::ok("true"))
    }
}
fn governance(
    cert: &crate::NodeCertificate,
    ms: u64,
) -> (SourceRegistry, GovernedRegistration, Facts) {
    let mut src = source();
    src.resource_scope
        .tenant_id
        .clone_from(&cert.claim.tenant_id);
    src.approval_policy = ApprovalLevel::A1Actor;
    src.confirmation_policy = ApprovalLevel::A1Actor;
    src.expiration_criteria.expires_at_ms = ms + 600_000;
    let sources = SourceRegistry::compile([src]).unwrap();
    let compiled = sources.get("local.observe", "1.0.0").unwrap();
    let principal = Principal::new(&cert.claim.account_id, PrincipalKind::HumanSubject).unwrap();
    let request = abi_capability::Request::new(
        "attempt",
        "call",
        principal.clone(),
        compiled.source().resource_scope.clone(),
        "local.observe",
        "1.0.0",
        compiled.package().digest,
        Digest::default(),
        "p",
        "g",
        "s",
        0,
    )
    .unwrap();
    let grant = Grant::active(
        "grant",
        Principal::new("owner", PrincipalKind::OrganizationOwner).unwrap(),
        principal,
        request.scope.clone(),
        compiled.package(),
        ms - 1,
        ms + 600_000,
        ApprovalLevel::A1Actor,
        0,
        "p",
        "g",
        "s",
    )
    .unwrap();
    let facts = Facts(GovernedFacts {
        platform: PlatformFacts::new(BTreeSet::new(), 0, "p", "g", "s"),
        grants: vec![grant],
        used_decision_ids: BTreeSet::new(),
        predicates: BTreeSet::from(["fresh".into(), "scoped".into()]),
        captured_pre_state: None,
        observed_at_ms: ms - 1,
        expires_at_ms: ms + 600_000,
        permitted_data_classes: BTreeSet::from([DataClass::Operational]),
        residency: ResidencyPolicy::LocalOnly,
        retention: RetentionClass::Session,
        safety_allows: true,
        rate_allows: true,
    });
    (
        sources,
        GovernedRegistration {
            tool_name: "observe".into(),
            request,
        },
        facts,
    )
}
struct Case {
    runtime: crate::DurableRelayRuntime<FileControlStore>,
    dispatch: crate::Dispatch,
    policy: crate::SignedDeliveryPolicy,
    cert: crate::NodeCertificate,
    key: SigningKey,
    account: crate::Account,
}
fn enroll_tool(
    f: &Fixture,
    account: Option<crate::Account>,
) -> (
    crate::Account,
    crate::NodeCertificate,
    SigningKey,
    crate::RelayKey,
) {
    let seconds = now().unwrap();
    let mut authority = f.authority.lock().unwrap();
    let account = account.unwrap_or_else(|| {
        authority
            .account(
                &crate::types::VerifiedIdentity {
                    issuer: "https://governed.example".into(),
                    subject: "local-user".into(),
                },
                seconds,
            )
            .unwrap()
    });
    let key = SigningKey::from_bytes(&[119; 32]);
    let encryption = crate::RelayKey::from_seed(&[120; 32]).unwrap();
    let ticket = authority
        .begin_pairing(
            &crate::PairRequest {
                public_key: key.verifying_key().to_bytes(),
                encryption_public_key: Some(encryption.public_key()),
                capabilities: vec!["tool.execute".into()],
            },
            seconds,
        )
        .unwrap();
    let claim = authority
        .claim_pairing(&account, &ticket.user_code, seconds)
        .unwrap();
    let cert = authority
        .confirm_pairing(
            &ticket.enrollment_id,
            &ticket.poll_secret,
            &key.sign(&claim.signing_bytes().unwrap()).to_bytes(),
            seconds,
        )
        .unwrap();
    (account, cert, key, encryption)
}
fn case(f: &Fixture) -> Case {
    case_for(f, None)
}
fn case_for(f: &Fixture, owner: Option<crate::Account>) -> Case {
    let (account, cert, key, encryption) = enroll_tool(f, owner);
    let seconds = now().unwrap();
    let mut authority = f.authority.lock().unwrap();
    let binding = crate::JobBinding {
        protocol: "abi.relay-job.v1".into(),
        audience: cert.audience.clone(),
        job_id: uuid::Uuid::new_v4().to_string(),
        account_id: account.account_id.clone(),
        tenant_id: account.tenant_id.clone(),
        node_id: cert.node_id.clone(),
        capability: "tool.execute".into(),
        submitted_at: seconds,
        expires_at: seconds + 300,
        max_result_bytes: 16384,
        reply_key: crate::RelayKey::generate().unwrap().public_key(),
    };
    let job = crate::ToolJob {
        protocol: "abi.governed-tool.v1".into(),
        tool: "observe".into(),
        revision: "scratch-v1".into(),
        parameters: serde_json::json!({}),
    };
    let payload = crate::seal(
        &encryption.public_key(),
        &binding.aad("request").unwrap(),
        &serde_json::to_vec(&job).unwrap(),
    )
    .unwrap();
    let submission = crate::Submission { binding, payload };
    let delivery = crate::DeliveryPolicy::for_submission(&submission, seconds + 86400).unwrap();
    let mut relay = f.relay.lock().unwrap();
    relay
        .submit_durable(
            &mut authority,
            &account,
            crate::DurableSubmission {
                submission,
                delivery,
            },
            seconds,
        )
        .unwrap();
    let poll = crate::NodeMessage::sign(&cert, &key, crate::NodeAction::Poll, seconds).unwrap();
    let reply = relay.node(&mut authority, &poll, seconds).unwrap();
    let runtime = crate::RelayRuntime::new_tools(
        cert.clone(),
        key.clone(),
        encryption,
        authority.public_key(),
        "scratch-v1",
        FileControlStore::create(
            &f.dir.path().join("tool-worker.log"),
            FileControlStoreLimits::default(),
        )
        .unwrap(),
        seconds,
    )
    .unwrap();
    let runtime = crate::DurableRelayRuntime::new(
        runtime,
        crate::FileRelayOutbox::create(
            &f.dir.path().join("tool-outbox.sqlite"),
            crate::OutboxLimits::default(),
        )
        .unwrap(),
    )
    .unwrap();
    Case {
        runtime,
        dispatch: reply.dispatch.unwrap(),
        policy: reply.delivery.unwrap(),
        cert,
        key,
        account,
    }
}
fn refresh(f: &Fixture, c: &Case) -> impl FnMut() -> crate::Result<crate::DispatchPermit> + use<> {
    let authority = f.authority.clone();
    let relay = f.relay.clone();
    let cert = c.cert.clone();
    let key = c.key.clone();
    let id = c.dispatch.submission.binding.job_id.clone();
    let time = c.dispatch.submission.binding.submitted_at;
    move || {
        let request = crate::NodeMessage::sign(
            &cert,
            &key,
            crate::NodeAction::Inspect { job_id: id.clone() },
            time,
        )?;
        relay
            .lock()
            .unwrap()
            .node(&mut authority.lock().unwrap(), &request, time)?
            .permit
            .ok_or(crate::Error::Denied)
    }
}
#[tokio::test]
async fn governed_tool_pending_approval_then_creates_scratch_file_once() {
    let f = fixture().await;
    let mut c = case(&f);
    let ms = u64::try_from(c.dispatch.submission.binding.submitted_at).unwrap() * 1000;
    let clock = FixedClock::new(ms);
    let (sources, registration, mut facts) = governance(&c.cert, ms);
    let mut approvals = Local::new(f.dir.path().join("local-approval.json"));
    let mut executor = FileAdapter {
        path: f.dir.path().join("approved-effect.txt"),
        effects: 0,
        invocations: 0,
    };
    let mut fresh = refresh(&f, &c);
    let effects = vec!["Create approved-effect.txt in the pinned scratch directory".into()];
    let mut invoke =
        |c: &mut Case, approvals: &mut Local, executor: &mut FileAdapter, facts: &mut Facts| {
            c.runtime.execute_governed(
                &c.dispatch,
                &c.policy,
                &mut crate::GovernedToolContext {
                    user_id: "local-user",
                    resource: "scratch/approved-effect.txt",
                    expected_effects: &effects,
                    sources: &sources,
                    registration: registration.clone(),
                    facts,
                    approvals,
                    executor,
                    clock: &clock,
                    refresh: &mut fresh,
                },
                &CancellationToken::new(),
            )
        };
    let first = invoke(&mut c, &mut approvals, &mut executor, &mut facts).unwrap();
    assert!(matches!(first, crate::GovernedToolOutcome::Pending(_)));
    assert!(!executor.path.exists());
    assert_eq!(executor.effects, 0);
    assert_eq!(executor.invocations, 0);
    assert!(
        c.runtime
            .reconcile_local(now().unwrap())
            .unwrap()
            .is_empty()
    );
    let entry = c
        .runtime
        .outbox
        .entry(&c.dispatch.submission.binding.job_id)
        .unwrap()
        .unwrap();
    assert!(
        entry.manifest.is_none(),
        "pending local approval must survive reconciliation"
    );
    let preview = approvals.preview();
    assert_eq!(preview.account_id, c.account.account_id);
    approvals
        .decide_local_approval(
            &preview.request.call_id,
            preview.digest(),
            "human-decision",
            true,
            ms,
        )
        .unwrap();
    let approved = invoke(&mut c, &mut approvals, &mut executor, &mut facts).unwrap();
    assert!(
        matches!(approved, crate::GovernedToolOutcome::Recorded(ref r) if r.status == abi_worker::JobStatus::Succeeded),
        "{approved:?}"
    );
    assert_eq!(
        std::fs::read(&executor.path).unwrap(),
        b"approved-scratch-effect"
    );
    assert_eq!(executor.effects, 1);
    assert_eq!(executor.invocations, 1);
    let duplicate = invoke(&mut c, &mut approvals, &mut executor, &mut facts).unwrap();
    assert!(matches!(duplicate, crate::GovernedToolOutcome::Recorded(_)));
    assert_eq!(executor.effects, 1);
    assert_eq!(executor.invocations, 1);
    let entry = c
        .runtime
        .outbox
        .entry(&c.dispatch.submission.binding.job_id)
        .unwrap()
        .unwrap();
    assert_eq!(
        entry.manifest.as_ref().unwrap().outcome,
        crate::RecordedOutcome::Completed
    );
    assert_completed_reopen(&f, c, &mut executor, &mut facts, &mut invoke);
}
fn assert_completed_reopen(
    f: &Fixture,
    mut c: Case,
    executor: &mut FileAdapter,
    facts: &mut Facts,
    invoke: &mut impl FnMut(
        &mut Case,
        &mut Local,
        &mut FileAdapter,
        &mut Facts,
    ) -> crate::Result<crate::GovernedToolOutcome>,
) {
    let entry = c
        .runtime
        .outbox
        .entry(&c.dispatch.submission.binding.job_id)
        .unwrap()
        .unwrap();
    let saved_invocations = executor.invocations;
    let saved_packets = serde_json::to_vec(&entry.packets).unwrap();
    let saved_manifest = entry.manifest;
    let saved_effect = std::fs::metadata(&executor.path)
        .unwrap()
        .modified()
        .unwrap();
    drop(c.runtime);
    c.runtime = crate::DurableRelayRuntime::new(
        crate::RelayRuntime::new_tools(
            c.cert.clone(),
            c.key.clone(),
            crate::RelayKey::from_seed(&[120; 32]).unwrap(),
            f.authority.lock().unwrap().public_key(),
            "scratch-v1",
            FileControlStore::open(
                &f.dir.path().join("tool-worker.log"),
                FileControlStoreLimits::default(),
            )
            .unwrap(),
            now().unwrap(),
        )
        .unwrap(),
        crate::FileRelayOutbox::open(
            &f.dir.path().join("tool-outbox.sqlite"),
            crate::OutboxLimits::default(),
        )
        .unwrap(),
    )
    .unwrap();
    // Recovery owns a recorded outcome, not permission to request another approval.
    let unavailable_approval_path = f.dir.path().join("unavailable-approval.json");
    let mut unavailable_approvals = Local::new(unavailable_approval_path.clone());
    facts.0.safety_allows = false;
    let replay = invoke(&mut c, &mut unavailable_approvals, executor, facts).unwrap();
    assert!(matches!(replay, crate::GovernedToolOutcome::Recorded(_)));
    assert_eq!(executor.invocations, saved_invocations);
    assert_eq!(executor.effects, 1);
    assert!(!unavailable_approval_path.exists());
    assert_eq!(
        std::fs::read(&executor.path).unwrap(),
        b"approved-scratch-effect"
    );
    assert_eq!(
        std::fs::metadata(&executor.path)
            .unwrap()
            .modified()
            .unwrap(),
        saved_effect
    );
    let replayed = c
        .runtime
        .outbox
        .entry(&c.dispatch.submission.binding.job_id)
        .unwrap()
        .unwrap();
    assert_eq!(
        serde_json::to_vec(&replayed.packets).unwrap(),
        saved_packets
    );
    assert_eq!(
        serde_json::to_vec(&replayed.manifest).unwrap(),
        serde_json::to_vec(&saved_manifest).unwrap()
    );
}
#[tokio::test]
async fn governed_tool_tenant_mutation_and_fresh_revocation_refuse_effects() {
    for mode in [
        "tenant",
        "changed-authority",
        "revoked",
        "cancelled",
        "abandoned",
    ] {
        let f = fixture().await;
        let mut c = case(&f);
        let ms = u64::try_from(c.dispatch.submission.binding.submitted_at).unwrap() * 1000;
        let clock = FixedClock::new(ms);
        let (sources, mut registration, mut facts) = governance(&c.cert, ms);
        let mut approvals = Local::new(f.dir.path().join("local-approval.json"));
        let mut executor = FileAdapter {
            path: f.dir.path().join("refused-effect.txt"),
            effects: 0,
            invocations: 0,
        };
        let mut fresh = refresh(&f, &c);
        let effects = vec!["Create only the approved scratch file".into()];
        if mode != "tenant" {
            let outcome = c
                .runtime
                .execute_governed(
                    &c.dispatch,
                    &c.policy,
                    &mut crate::GovernedToolContext {
                        user_id: "local-user",
                        resource: "scratch/refused-effect.txt",
                        expected_effects: &effects,
                        sources: &sources,
                        registration: registration.clone(),
                        facts: &mut facts,
                        approvals: &mut approvals,
                        executor: &mut executor,
                        clock: &clock,
                        refresh: &mut fresh,
                    },
                    &CancellationToken::new(),
                )
                .unwrap_or_else(|e| panic!("initial pending {mode}: {e:?}"));
            assert!(matches!(outcome, crate::GovernedToolOutcome::Pending(_)));
            let p = approvals.preview();
            approvals
                .decide_local_approval(&p.request.call_id, p.digest(), "human-decision", true, ms)
                .unwrap();
        }
        invalidate(mode, &f, &mut c, &mut registration, &mut facts);
        if matches!(mode, "cancelled" | "abandoned") {
            c.runtime.reconcile_local(now().unwrap()).unwrap();
            let entry = c
                .runtime
                .outbox
                .entry(&c.dispatch.submission.binding.job_id)
                .unwrap()
                .unwrap();
            assert!(
                entry.manifest.is_some(),
                "sticky stop must not be treated as awaiting approval"
            );
        }
        let result = c.runtime.execute_governed(
            &c.dispatch,
            &c.policy,
            &mut crate::GovernedToolContext {
                user_id: "local-user",
                resource: "scratch/refused-effect.txt",
                expected_effects: &effects,
                sources: &sources,
                registration,
                facts: &mut facts,
                approvals: &mut approvals,
                executor: &mut executor,
                clock: &clock,
                refresh: &mut fresh,
            },
            &CancellationToken::new(),
        );
        assert!(
            result.is_err() || matches!(result, Ok(crate::GovernedToolOutcome::Unresolved)),
            "{mode}: {result:?}"
        );
        assert_eq!(executor.effects, 0, "{mode}");
        assert!(!executor.path.exists(), "{mode}");
    }
}

fn invalidate(
    mode: &str,
    f: &Fixture,
    c: &mut Case,
    registration: &mut GovernedRegistration,
    facts: &mut Facts,
) {
    match mode {
        "tenant" => registration.request.scope.tenant_id = "foreign-tenant".into(),
        "changed-authority" => facts.0.platform.revocation_epoch += 1,
        "revoked" => f
            .authority
            .lock()
            .unwrap()
            .revoke(&c.account, &c.cert.node_id, now().unwrap())
            .unwrap(),
        "cancelled" => c
            .runtime
            .outbox
            .observe_cancel(&c.dispatch.submission.binding.job_id)
            .unwrap(),
        "abandoned" => c
            .runtime
            .outbox
            .mark_unresolved(&c.dispatch.submission.binding.job_id, now().unwrap())
            .unwrap(),
        _ => unreachable!(),
    }
}
#[tokio::test]
async fn browser_tool_submission_requires_durable_endpoint() {
    let f = fixture().await;
    let alice = browser(&f, "tool-browser").await;
    let account = super::json(
        request(
            &f.app,
            "GET",
            "/v1/account",
            &alice.cookie,
            None,
            None,
            None,
        )
        .await,
    )
    .await;
    let account: crate::Account = serde_json::from_value(account["account"].clone()).unwrap();
    let c = case_for(&f, Some(account));
    let submission = serde_json::to_value(&c.dispatch.submission).unwrap();
    let legacy = request(
        &f.app,
        "POST",
        "/v1/jobs",
        &alice.cookie,
        Some("http://127.0.0.1:3000"),
        Some(&alice.csrf),
        Some(submission),
    )
    .await;
    assert_eq!(legacy.status(), StatusCode::UNAUTHORIZED);
    let durable = crate::DurableSubmission {
        submission: c.dispatch.submission.clone(),
        delivery: c.policy.policy.clone(),
    };
    let response = request(
        &f.app,
        "POST",
        "/v2/jobs",
        &alice.cookie,
        Some("http://127.0.0.1:3000"),
        Some(&alice.csrf),
        Some(serde_json::to_value(&durable).unwrap()),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
}
