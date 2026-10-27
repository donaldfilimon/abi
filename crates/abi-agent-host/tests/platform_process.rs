//! Real scratch-directory capability binding and source validation.
#![cfg(unix)]
#[path = "platform_process/bindings.rs"]
mod bindings;
use abi_agent_host::platform_process::{ProcessLimits, ProcessPlan};
use abi_capability::{
    Digest, FixedClock,
    source::{PackageSource, SourceRegistry},
};
use serde_json::json;
use std::{path::PathBuf, sync::Arc};

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        use std::os::unix::fs::PermissionsExt as _;
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let name = format!(
            "abi-process-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        );
        let path = std::env::temp_dir().join(name);
        std::fs::create_dir(&path).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        Self(path.canonicalize().unwrap())
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn source(resource: Digest) -> PackageSource {
    serde_json::from_value(json!({
        "id":"local.process.cwd", "version":"1.0.0",
        "input_schema":{"type":"object","additionalProperties":false},
        "output_schema":{"type":"object","properties":{"cwd":{"type":"string"},"exit_code":{"const":0}},"required":["cwd","exit_code"],"additionalProperties":false},
        "error_schema":{"type":"null"}, "streaming":"None","cancellation":"Cooperative",
        "required_platform_permissions":["process.pinned_exec_access","process.pinned_cwd_search"],
        "resource_scope":{"organization_id":"org","deployment_id":"deploy","tenant_id":"tenant","platform":"local","guild_ref":null,"resource_digest":resource,"subject_digest":Digest::from_bytes([2;32])},
        "side_effect":"PlatformRead","reversibility":"Irreversible","risk_class":"Low",
        "data_classes":["Operational"],"residency":"LocalOnly","retention":"Session",
        "preconditions":["process.ready"],"invariants":["process.scoped"],"deadline_ms":1000,"rate_class":"bounded",
        "budgets":{"max_input_bytes":512,"max_output_bytes":8192,"max_events":4,"max_calls":1},
        "confirmation_policy":"A1Actor","delegation_policy":"Forbidden","expiry_policy_ms":1000,"revocation_policy":"RecheckBeforeEffect","approval_policy":"A1Actor",
        "idempotency":"NaturallyIdempotent","compensation":null,"rollback":"None",
        "postconditions":["process.exited_zero","process.cwd_matches"],
        "adapter_bindings":[{"adapter_id":"abi.pinned-process","api_version":"1"}],
        "receipt_requirements":{"redacted":true,"require_postcondition_evidence":true},
        "fixtures":{"deterministic":Digest::from_bytes([3;32]),"adversarial":Digest::from_bytes([4;32]),"failure":Digest::from_bytes([5;32])},
        "promotion_thresholds":{"minimum_passed":1,"maximum_failed":0},
        "expiration_criteria":{"expires_at_ms":10000,"disable_on_schema_drift":true,"disable_on_missing_evidence":true}
    })).unwrap()
}

#[test]
fn pinned_directory_binds_only_to_exact_closed_source() {
    let scratch = Scratch::new();
    let plan = ProcessPlan::working_directory(&scratch.0, ProcessLimits::default()).unwrap();
    assert_ne!(plan.resource_digest(), Digest::default());
    let registry = SourceRegistry::compile([source(plan.resource_digest())]).unwrap();
    assert!(
        plan.bind(
            registry.get("local.process.cwd", "1.0.0").unwrap(),
            Arc::new(FixedClock::new(100))
        )
        .is_ok()
    );
}

#[test]
fn permissive_schemas_and_missing_owned_prerequisites_refuse_binding() {
    let scratch = Scratch::new();
    for mutation in [
        "input",
        "output",
        "precondition",
        "invariant",
        "permission",
        "postcondition",
        "approval",
        "resource",
        "budget",
        "adapter",
    ] {
        let plan = ProcessPlan::working_directory(&scratch.0, ProcessLimits::default()).unwrap();
        let mut src = source(plan.resource_digest());
        match mutation {
            "input" => src.input_schema = json!({"type":"object"}),
            "output" => src.output_schema = json!({"type":"object"}),
            "precondition" => src.preconditions.clear(),
            "invariant" => src.invariants.clear(),
            "permission" => src.required_platform_permissions.clear(),
            "postcondition" => {
                src.postconditions.insert("extra".into());
            }
            "approval" => src.approval_policy = abi_capability::ApprovalLevel::A0None,
            "resource" => src.resource_scope.resource_digest = Digest::default(),
            "budget" => src.budgets.max_output_bytes = 1,
            "adapter" => src.adapter_bindings[0].adapter_id = "other".into(),
            _ => unreachable!(),
        }
        let registry = SourceRegistry::compile([src]).unwrap();
        assert!(
            plan.bind(
                registry.get("local.process.cwd", "1.0.0").unwrap(),
                Arc::new(FixedClock::new(100))
            )
            .is_err(),
            "{mutation}"
        );
    }
}

#[test]
fn symlink_directory_and_invalid_limits_refuse_before_execution() {
    let scratch = Scratch::new();
    let link = scratch.0.join("alias");
    std::os::unix::fs::symlink(&scratch.0, &link).unwrap();
    assert!(ProcessPlan::working_directory(&link, ProcessLimits::default()).is_err());
    for limits in [
        ProcessLimits {
            timeout_ms: 0,
            ..ProcessLimits::default()
        },
        ProcessLimits {
            timeout_ms: 30001,
            ..ProcessLimits::default()
        },
        ProcessLimits {
            max_output_bytes: 0,
            ..ProcessLimits::default()
        },
        ProcessLimits {
            max_output_bytes: 65537,
            ..ProcessLimits::default()
        },
        ProcessLimits {
            cleanup_grace_ms: 0,
            ..ProcessLimits::default()
        },
        ProcessLimits {
            cleanup_grace_ms: 1001,
            ..ProcessLimits::default()
        },
    ] {
        assert!(ProcessPlan::working_directory(&scratch.0, limits).is_err());
    }
}

use abi_agent_host::{
    GovernedFactProvider, GovernedFacts, GovernedLiveExecutor, ObservationSource,
    PostconditionObservation, Postconditions, ToolExecutionContext,
};
use abi_agent_runtime::{CancellationToken, ToolCall};
use abi_capability::{
    ApprovalLevel, Clock, PlatformFacts, Principal, PrincipalKind,
    lifecycle::{AuthorizationBinding, LifecycleError, PreparedExecution},
    source::{CompiledSource, DataClass, ResidencyPolicy, RetentionClass},
};
use sha2::{Digest as _, Sha256};
use std::{
    collections::BTreeSet,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};
fn hash(value: &impl serde::Serialize) -> Digest {
    Digest::from_bytes(Sha256::digest(serde_json::to_vec(value).unwrap()).into())
}
struct TestClock(AtomicU64);
impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}
struct Upstream {
    expires: u64,
}
impl GovernedFactProvider for Upstream {
    fn observe(&mut self, _: &abi_capability::Request) -> Result<GovernedFacts, LifecycleError> {
        Ok(GovernedFacts {
            platform: PlatformFacts::new(
                BTreeSet::from(["upstream", "process.pinned_exec_access"]),
                0,
                "p",
                "g",
                "s",
            ),
            grants: vec![],
            used_decision_ids: BTreeSet::from(["used".into()]),
            predicates: BTreeSet::from(["upstream".into(), "process.ready".into()]),
            captured_pre_state: Some(Digest::from_bytes([9; 32])),
            observed_at_ms: 90,
            expires_at_ms: self.expires,
            permitted_data_classes: BTreeSet::from([DataClass::Operational]),
            residency: ResidencyPolicy::LocalOnly,
            retention: RetentionClass::Session,
            safety_allows: true,
            rate_allows: true,
        })
    }
    fn postconditions(
        &mut self,
        _: &abi_capability::Request,
        _: &BTreeSet<String>,
    ) -> Result<Vec<PostconditionObservation>, LifecycleError> {
        panic!("upstream cannot fabricate process postconditions")
    }
}
fn execution(compiled: &CompiledSource) -> (abi_capability::Request, PreparedExecution, ToolCall) {
    let principal = Principal::new("actor", PrincipalKind::Service).unwrap();
    let call = ToolCall::new("call", "process_cwd", "{}");
    let src = compiled.source();
    let id = hash(&(
        "abi.host.execution.v2",
        "attempt",
        &principal,
        &src.resource_scope,
        &src.id,
        &src.version,
        &call.id,
        &call.name,
    ));
    let request = abi_capability::Request::new(
        "attempt",
        id.to_string().trim_start_matches("sha256:"),
        principal,
        src.resource_scope.clone(),
        &src.id,
        &src.version,
        compiled.package().digest,
        hash(&json!({})),
        "p",
        "g",
        "s",
        0,
    )
    .unwrap();
    let binding = AuthorizationBinding {
        request_digest: request.request_digest,
        principal_digest: hash(&request.principal),
        scope_digest: hash(&request.scope),
        package_digest: request.package_digest,
        grant_digest: Digest::from_bytes([6; 32]),
        policy_digest: hash(&("abi.policy.v1", "p", "g", "s", 0_u64)),
        schema_digest: compiled.schema_digest(),
        precondition_digest: Digest::from_bytes([7; 32]),
        compensation_origin: None,
        required_approval: ApprovalLevel::A1Actor,
    };
    (request, PreparedExecution::after_commit(id, binding), call)
}
fn required() -> BTreeSet<String> {
    BTreeSet::from(["process.exited_zero".into(), "process.cwd_matches".into()])
}

#[test]
fn real_pwd_has_request_bound_single_use_os_postconditions() {
    let scratch = Scratch::new();
    let plan = ProcessPlan::working_directory(&scratch.0, ProcessLimits::default()).unwrap();
    let registry = SourceRegistry::compile([source(plan.resource_digest())]).unwrap();
    let compiled = registry.get("local.process.cwd", "1.0.0").unwrap();
    let clock = Arc::new(TestClock(AtomicU64::new(100)));
    let (mut executor, handle) = plan.bind(compiled, clock).unwrap();
    let mut facts = handle.decorate(Upstream { expires: 500 });
    let (request, prepared, call) = execution(compiled);
    let observed = facts.observe(&request).unwrap();
    assert_eq!((observed.observed_at_ms, observed.expires_at_ms), (90, 500));
    assert!(observed.platform.permissions.contains("upstream"));
    assert!(
        observed
            .platform
            .permissions
            .contains("process.pinned_cwd_search")
    );
    assert!(observed.predicates.contains("upstream"));
    assert_eq!(observed.used_decision_ids, BTreeSet::from(["used".into()]));
    let cancel = CancellationToken::new();
    let output = executor
        .execute(
            &call,
            &request,
            compiled,
            &prepared,
            ToolExecutionContext {
                cancellation: &cancel,
                deadline: Instant::now() + Duration::from_secs(2),
            },
        )
        .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(output.payload()).unwrap(),
        json!({"cwd":scratch.0.to_str().unwrap(),"exit_code":0})
    );
    let evidence = facts.postconditions(&request, &required()).unwrap();
    assert_eq!(evidence.len(), 2);
    assert!(
        evidence
            .iter()
            .all(|o| o.source == ObservationSource::TrustedLocalAdapter
                && o.status == Postconditions::Verified
                && o.request_digest == request.request_digest
                && o.observed_at_ms == 100
                && o.expires_at_ms == 1100
                && o.evidence_digest.is_some())
    );
    assert!(facts.postconditions(&request, &required()).is_err());
}

#[test]
fn namespaced_source_labels_do_not_widen_frozen_constructor() {
    let scratch = Scratch::new();
    let plan = ProcessPlan::working_directory(&scratch.0, ProcessLimits::default()).unwrap();
    let registry = SourceRegistry::compile([source(plan.resource_digest())]).unwrap();
    let package = registry
        .get("local.process.cwd", "1.0.0")
        .unwrap()
        .package();
    assert!(
        package
            .required_permissions
            .contains("process.pinned_exec_access")
    );
    assert!(package.postconditions.contains("process.cwd_matches"));
    assert!(
        abi_capability::CapabilityPackage::new(
            "local.process.cwd",
            "1.0.0",
            package.digest,
            abi_capability::SideEffect::PlatformRead,
            abi_capability::Reversibility::Irreversible,
            abi_capability::RiskClass::Low,
            ApprovalLevel::A1Actor,
            ["process.pinned_exec_access"],
            ["process.cwd_matches"]
        )
        .is_err()
    );
}

#[test]
fn receipt_expiry_wrong_request_wrong_predicates_and_refused_execute_consume_evidence() {
    for refusal in ["expiry", "rollback", "request", "predicates", "execute"] {
        let scratch = Scratch::new();
        let plan = ProcessPlan::working_directory(&scratch.0, ProcessLimits::default()).unwrap();
        let registry = SourceRegistry::compile([source(plan.resource_digest())]).unwrap();
        let compiled = registry.get("local.process.cwd", "1.0.0").unwrap();
        let clock = Arc::new(TestClock(AtomicU64::new(100)));
        let (mut executor, handle) = plan.bind(compiled, clock.clone()).unwrap();
        let mut facts = handle.decorate(Upstream { expires: 500 });
        let (request, prepared, call) = execution(compiled);
        let cancel = CancellationToken::new();
        executor
            .execute(
                &call,
                &request,
                compiled,
                &prepared,
                ToolExecutionContext {
                    cancellation: &cancel,
                    deadline: Instant::now() + Duration::from_secs(2),
                },
            )
            .unwrap();
        let mut changed_request = request.clone();
        let mut predicates = required();
        match refusal {
            "expiry" => clock.0.store(1100, Ordering::SeqCst),
            "rollback" => clock.0.store(99, Ordering::SeqCst),
            "request" => changed_request.request_digest = Digest::default(),
            "predicates" => {
                predicates.insert("fake".into());
            }
            "execute" => {
                cancel.cancel();
                assert!(
                    executor
                        .execute(
                            &call,
                            &request,
                            compiled,
                            &prepared,
                            ToolExecutionContext {
                                cancellation: &cancel,
                                deadline: Instant::now() + Duration::from_secs(2),
                            }
                        )
                        .is_err()
                );
            }
            _ => unreachable!(),
        }
        assert!(
            facts.postconditions(&changed_request, &predicates).is_err(),
            "{refusal}"
        );
        clock.0.store(100, Ordering::SeqCst);
        assert!(
            facts.postconditions(&request, &required()).is_err(),
            "receipt must be consumed: {refusal}"
        );
    }
}

#[test]
fn changed_directory_and_stale_upstream_cannot_supply_owned_facts() {
    for expired in [true, false] {
        let scratch = Scratch::new();
        let original = scratch.0.join("original");
        std::fs::create_dir(&original).unwrap();
        let plan = ProcessPlan::working_directory(&original, ProcessLimits::default()).unwrap();
        let registry = SourceRegistry::compile([source(plan.resource_digest())]).unwrap();
        let compiled = registry.get("local.process.cwd", "1.0.0").unwrap();
        let (_, handle) = plan
            .bind(compiled, Arc::new(TestClock(AtomicU64::new(100))))
            .unwrap();
        let mut facts = handle.decorate(Upstream {
            expires: if expired { 100 } else { 500 },
        });
        let (request, _, _) = execution(compiled);
        if !expired {
            assert!(facts.observe(&request).is_ok());
            std::fs::rename(&original, scratch.0.join("moved")).unwrap();
            std::fs::create_dir(&original).unwrap();
        }
        assert!(facts.observe(&request).is_err());
    }
}

#[test]
fn non_utf8_directories_and_escaped_output_budgets_are_refused() {
    use std::os::unix::ffi::OsStringExt as _;
    let scratch = Scratch::new();
    let invalid = scratch.0.join(std::ffi::OsString::from_vec(vec![0xff]));
    if let Err(err) = std::fs::create_dir(&invalid) {
        // APFS refuses ill-formed UTF-8 names before the adapter sees a directory.
        // Linux filesystems may accept them; both must refuse the operation.
        assert_eq!(err.raw_os_error(), Some(libc::EILSEQ));
    }
    assert!(ProcessPlan::working_directory(&invalid, ProcessLimits::default()).is_err());
    let escaped = scratch.0.join("quotes-\"\n\\");
    std::fs::create_dir(&escaped).unwrap();
    let plan = ProcessPlan::working_directory(&escaped, ProcessLimits::default()).unwrap();
    let mut src = source(plan.resource_digest());
    // Raw cwd fits, but the JSON-escaped result and metadata do not fit this ceiling.
    src.budgets.max_output_bytes = u32::try_from(escaped.to_str().unwrap().len() + 1).unwrap();
    let registry = SourceRegistry::compile([src]).unwrap();
    assert!(
        plan.bind(
            registry.get("local.process.cwd", "1.0.0").unwrap(),
            Arc::new(FixedClock::new(100))
        )
        .is_err()
    );
}

#[test]
fn actual_directory_search_permission_is_required_and_restorable() {
    use std::os::unix::fs::PermissionsExt as _;
    let scratch = Scratch::new();
    std::fs::set_permissions(&scratch.0, std::fs::Permissions::from_mode(0o600)).unwrap();
    // Kernel privileges may bypass directory search restrictions. The normal
    // unprivileged test user must refuse before any process can be spawned.
    #[allow(unsafe_code)]
    let privileged = unsafe { libc::geteuid() } == 0;
    if !privileged {
        assert!(ProcessPlan::working_directory(&scratch.0, ProcessLimits::default()).is_err());
    }
    std::fs::set_permissions(&scratch.0, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(ProcessPlan::working_directory(&scratch.0, ProcessLimits::default()).is_ok());
}
