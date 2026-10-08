//! Private training registry. The caller authenticates principals before invoking
//! privileged APIs; this library enforces configured authority, not authentication.
//! Evidence registration is a trusted attestation boundary: the registrar must
//! verify weights and evaluation outside this API. Digests alone never qualify
//! a record. Completed-run and artifact verification belong to the next layer.
//! Training/teacher callers use `training_records`, which refuses any held-out
//! snapshot instead of filtering records under an unchanged committed root.
//! Corpus privacy requires a real owned 0700 Unix root. Backend file modes are
//! protected by that directory; symlinks/hard links and escaping store names refuse.
//! This trusts the same user against concurrent path/mode replacement. A supplied
//! store has already recovered before this constructor; callers must open it safely.
//! A supplied `DurableStore` owns WDBX's advisory writer lock for the whole session.
use abi_ai::training_contracts::{
    ApprovalKind, ApprovedRecord, ContractError, SnapshotManifest, SnapshotPolicy, Split,
    build_snapshot, canonical_record_bytes, sha256_hex,
};
use abi_wdbx::durable::DurableStore;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const STATE_KEY: &str = "training:v1:state";
/// Registry failures are explicit and never authorize a degraded path.
#[derive(Debug)]
pub enum RegistryError {
    /// Invalid contract or stored schema.
    Invalid(String),
    /// Authenticated caller lacks configured authority.
    Unauthorized,
    /// Missing registered evidence or record.
    Missing(String),
    /// Immutable identity already binds different content.
    Conflict(String),
    /// Snapshot contains an irreversibly revoked record.
    Revoked(String),
    /// Eligibility receipt predates a registry mutation.
    StaleEligibility,
    /// A durable write had an uncertain outcome; drop and reopen the session.
    PoisonedSession,
    /// Durable storage failed.
    Storage(String),
}
impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for RegistryError {}
impl From<ContractError> for RegistryError {
    fn from(e: ContractError) -> Self {
        Self::Invalid(e.to_string())
    }
}
impl From<serde_json::Error> for RegistryError {
    fn from(e: serde_json::Error) -> Self {
        Self::Invalid(e.to_string())
    }
}
/// Explicit authority policy. Aliases map directly to canonical principals;
/// unknown principals refuse. There is no default allow-all policy.
#[derive(Debug, Clone)]
pub struct AuthorityPolicy {
    /// Every accepted spelling, including canonical self mappings.
    pub principals: BTreeMap<String, String>,
    /// Canonical principals allowed to register evidence.
    pub evidence_registrars: BTreeSet<String>,
    /// Canonical human decision authorities.
    pub human_approvers: BTreeSet<String>,
    /// Canonical permission authorities.
    pub permission_operators: BTreeSet<String>,
    /// Canonical irreversible revocation authorities.
    pub revocation_operators: BTreeSet<String>,
    /// Canonical independent qualification reviewers.
    pub qualification_reviewers: BTreeSet<String>,
}
macro_rules! data {
    ($name:ident {$($field:ident: $ty:ty),* $(,)?}) => {
        #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        #[doc = concat!(stringify!($name), " registry binding.")]
        pub struct $name {$(#[doc = concat!(stringify!($field), " binding.")] pub $field: $ty),*}
    };
}
data!(ArtifactIdentity {
    base_digest: String,
    adapter_digest: String,
    tokenizer_digest: String,
    generation_configuration_digest: String
});
/// Qualification purposes cannot be interchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualificationPurpose {
    /// Authorizes synthetic record generation.
    Teacher,
    /// Authorizes approval evaluation.
    Evaluator,
}
data!(QualificationEvidence {
    digest: String,
    artifact: ArtifactIdentity,
    artifact_principal: String,
    reviewer_principal: String,
    purpose: QualificationPurpose,
    passed: bool,
    verified_weight_artifact: bool
});
data!(PermissionDecision {
    id: String,
    operator_identity: String,
    record_digest: String,
    approved_use: String
});
data!(ApprovalDecision {
    id: String,
    approver_identity: String,
    record_digest: String,
    approved_use: String,
    evidence: ApprovalKind
});
data!(RevocationDecision {
    operator_identity: String,
    decision_id: String,
    reason: String
});
data!(Eligibility {
    snapshot_id: String,
    registry_revision: u64
});
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    version: u32,
    revision: u64,
    records: BTreeMap<String, String>,
    permissions: BTreeMap<String, PermissionDecision>,
    approvals: BTreeMap<String, ApprovalDecision>,
    qualifications: BTreeMap<String, QualificationEvidence>,
    snapshots: BTreeMap<String, SnapshotManifest>,
    revocations: BTreeMap<String, RevocationDecision>,
    #[serde(default)]
    runs: runs::Runs,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    version: u32,
    record: ApprovedRecord,
}
impl Default for State {
    fn default() -> Self {
        Self {
            version: 1,
            revision: 0,
            records: BTreeMap::new(),
            permissions: BTreeMap::new(),
            approvals: BTreeMap::new(),
            qualifications: BTreeMap::new(),
            snapshots: BTreeMap::new(),
            revocations: BTreeMap::new(),
            runs: runs::Runs::default(),
        }
    }
}
/// One owned writer session. All visible state transitions publish through one
/// WAL put. Immutable record blobs may remain orphaned after a failed publication.
pub struct TrainingRegistry {
    store: DurableStore,
    policy: AuthorityPolicy,
    state: State,
    poisoned: bool,
    #[cfg(test)]
    fail_after_write: bool,
    #[cfg(test)]
    wal_budget: u64,
}
impl TrainingRegistry {
    /// Opens an already locked store with explicit authority configuration.
    /// Refuses unsafe store paths before registry reads/writes. Unix ownership is
    /// mandatory; no caller directory or backend file permissions are changed.
    pub fn new(store: DurableStore, policy: AuthorityPolicy) -> Result<Self, RegistryError> {
        privacy::validate_store(store.paths())?;
        for (alias, principal) in &policy.principals {
            nonblank(alias)?;
            nonblank(principal)?;
            if policy.principals.get(principal) != Some(principal) {
                return Err(RegistryError::Invalid(
                    "aliases must resolve directly to canonical self mappings".into(),
                ));
            }
        }
        for group in [
            &policy.evidence_registrars,
            &policy.human_approvers,
            &policy.permission_operators,
            &policy.revocation_operators,
            &policy.qualification_reviewers,
        ] {
            for principal in group {
                if policy.principals.get(principal) != Some(principal) {
                    return Err(RegistryError::Unauthorized);
                }
            }
        }
        let state: State = store
            .get(STATE_KEY)
            .map(serde_json::from_str)
            .transpose()?
            .unwrap_or_default();
        if state.version != 1 {
            return Err(RegistryError::Invalid("unsupported registry schema".into()));
        }
        let registry = Self {
            store,
            policy,
            state,
            poisoned: false,
            #[cfg(test)]
            fail_after_write: false,
            #[cfg(test)]
            wal_budget: abi_wdbx::wal::MAX_WAL_BYTES,
        };
        for id in registry.state.records.keys() {
            registry.record(id)?;
        }
        for (id, manifest) in &registry.state.snapshots {
            if id != &manifest.root_digest {
                return Err(RegistryError::Invalid("snapshot key mismatch".into()));
            }
            registry.validate_manifest(manifest)?;
        }
        validate_split_isolation(registry.state.snapshots.values())?;
        registry.validate_runs()?;
        Ok(registry)
    }
    fn principal<'a>(&'a self, identity: &str) -> Result<&'a str, RegistryError> {
        self.policy
            .principals
            .get(identity)
            .map(String::as_str)
            .ok_or(RegistryError::Unauthorized)
    }
    fn authorize(&self, identity: &str, allowed: &BTreeSet<String>) -> Result<(), RegistryError> {
        if allowed.contains(self.principal(identity)?) {
            Ok(())
        } else {
            Err(RegistryError::Unauthorized)
        }
    }
    fn healthy(&self) -> Result<(), RegistryError> {
        if self.poisoned {
            Err(RegistryError::PoisonedSession)
        } else {
            Ok(())
        }
    }
    fn write(&mut self, key: &str, value: &str) -> Result<(), RegistryError> {
        self.healthy()?;
        let limit = abi_wdbx::wal::MAX_WAL_BYTES;
        #[cfg(test)]
        let limit = limit.min(self.wal_budget);
        // Six bytes per input byte bounds JSON's longest escape (\u0000).
        // 64 bytes covers the KV JSON envelope, CRC, space, and newline;
        // another 64 reserves the longest u64 epoch header after checkpoint.
        let frame_bytes = frame_size_bound(key.len(), value.len())?;
        if frame_bytes.checked_add(64).is_none_or(|size| size > limit) {
            return Err(RegistryError::Invalid(
                "WAL frame exceeds recoverable budget".into(),
            ));
        }
        let wal = abi_wdbx::wal::wal_path(self.store.paths());
        let wal_bytes = std::fs::metadata(&wal)
            .map_err(|e| RegistryError::Storage(e.to_string()))?
            .len();
        if wal_bytes
            .checked_add(frame_bytes)
            .is_none_or(|size| size > limit)
        {
            self.checkpoint()?;
        }
        let result = self
            .store
            .put(key, value)
            .map_err(|e| RegistryError::Storage(e.to_string()));
        #[cfg(test)]
        let result = if result.is_ok() && self.fail_after_write {
            self.fail_after_write = false;
            Err(RegistryError::Storage(
                "injected error after WAL publication".into(),
            ))
        } else {
            result
        };
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    fn publish(&mut self, mut next: State) -> Result<(), RegistryError> {
        next.revision = self
            .state
            .revision
            .checked_add(1)
            .ok_or_else(|| RegistryError::Invalid("revision exhausted".into()))?;
        self.write(STATE_KEY, &serde_json::to_string(&next)?)?;
        self.state = next;
        Ok(())
    }
    /// Registers immutable permission evidence under privileged caller authority.
    pub fn register_permission(
        &mut self,
        caller: &str,
        decision: PermissionDecision,
    ) -> Result<(), RegistryError> {
        self.healthy()?;
        self.authorize(caller, &self.policy.evidence_registrars)?;
        self.authorize(
            &decision.operator_identity,
            &self.policy.permission_operators,
        )?;
        nonblank(&decision.id)?;
        digest(&decision.record_digest)?;
        training_use(&decision.approved_use)?;
        let mut next = self.state.clone();
        if insert(&mut next.permissions, decision.id.clone(), decision)? {
            self.publish(next)?;
        }
        Ok(())
    }
    /// Registers an exact artifact qualification, never a metadata placeholder.
    pub fn register_qualification(
        &mut self,
        caller: &str,
        evidence: QualificationEvidence,
    ) -> Result<(), RegistryError> {
        self.healthy()?;
        self.authorize(caller, &self.policy.evidence_registrars)?;
        self.authorize(
            &evidence.reviewer_principal,
            &self.policy.qualification_reviewers,
        )?;
        if self.principal(&evidence.artifact_principal)?
            == self.principal(&evidence.reviewer_principal)?
        {
            return Err(RegistryError::Unauthorized);
        }
        digest(&evidence.digest)?;
        for d in [
            &evidence.artifact.base_digest,
            &evidence.artifact.adapter_digest,
            &evidence.artifact.tokenizer_digest,
            &evidence.artifact.generation_configuration_digest,
        ] {
            digest(d)?;
        }
        if !evidence.passed || !evidence.verified_weight_artifact {
            return Err(RegistryError::Invalid(
                "qualification needs passed evaluation and verified weights".into(),
            ));
        }
        let mut next = self.state.clone();
        if insert(&mut next.qualifications, evidence.digest.clone(), evidence)? {
            self.publish(next)?;
        }
        Ok(())
    }
    /// Registers subject-bound human or qualified evaluator approval evidence.
    pub fn register_approval(
        &mut self,
        caller: &str,
        decision: ApprovalDecision,
    ) -> Result<(), RegistryError> {
        self.healthy()?;
        self.authorize(caller, &self.policy.evidence_registrars)?;
        nonblank(&decision.id)?;
        digest(&decision.record_digest)?;
        training_use(&decision.approved_use)?;
        self.approval_authority(&decision.approver_identity, &decision.evidence)?;
        let mut next = self.state.clone();
        if insert(&mut next.approvals, decision.id.clone(), decision)? {
            self.publish(next)?;
        }
        Ok(())
    }
    fn qualification(
        &self,
        id: &str,
        purpose: QualificationPurpose,
    ) -> Result<&QualificationEvidence, RegistryError> {
        let q = self
            .state
            .qualifications
            .get(id)
            .ok_or_else(|| RegistryError::Missing(id.into()))?;
        if q.purpose != purpose || !q.passed || !q.verified_weight_artifact {
            return Err(RegistryError::Invalid(
                "qualification purpose or outcome".into(),
            ));
        }
        self.authorize(&q.reviewer_principal, &self.policy.qualification_reviewers)?;
        if self.principal(&q.artifact_principal)? == self.principal(&q.reviewer_principal)? {
            return Err(RegistryError::Unauthorized);
        }
        Ok(q)
    }
    fn approval_authority(
        &self,
        approver: &str,
        evidence: &ApprovalKind,
    ) -> Result<(), RegistryError> {
        match evidence {
            ApprovalKind::Human {} => self.authorize(approver, &self.policy.human_approvers),
            ApprovalKind::QualifiedEvaluator {
                artifact_digest,
                qualification_digest,
            } => {
                let q =
                    self.qualification(qualification_digest, QualificationPurpose::Evaluator)?;
                if artifact_digest != &q.artifact.adapter_digest
                    || self.principal(approver)? != self.principal(&q.artifact_principal)?
                {
                    return Err(RegistryError::Unauthorized);
                }
                Ok(())
            }
        }
    }
    fn resolve(&self, record: &ApprovedRecord) -> Result<(), RegistryError> {
        let subject = sha256_hex(&canonical_record_bytes(record)?);
        let p = self
            .state
            .permissions
            .get(
                &record
                    .permission
                    .as_ref()
                    .ok_or(ContractError::MissingPermission)?
                    .approved_use_decision_id,
            )
            .ok_or_else(|| RegistryError::Missing("permission decision".into()))?;
        self.authorize(&p.operator_identity, &self.policy.permission_operators)?;
        let a = record
            .approval
            .as_ref()
            .ok_or(ContractError::MissingApproval)?;
        let decision = self
            .state
            .approvals
            .get(&a.decision_id)
            .ok_or_else(|| RegistryError::Missing("approval decision".into()))?;
        if p.record_digest != subject
            || decision.record_digest != subject
            || self.principal(&a.approver_identity)?
                != self.principal(&decision.approver_identity)?
            || a.evidence != decision.evidence
        {
            return Err(RegistryError::Invalid(
                "decision subject or approver mismatch".into(),
            ));
        }
        training_use(&p.approved_use)?;
        training_use(&decision.approved_use)?;
        self.approval_authority(&a.approver_identity, &a.evidence)?;
        if let Some(t) = &record.teacher {
            let q = self.qualification(&t.qualification_digest, QualificationPurpose::Teacher)?;
            let artifact = ArtifactIdentity {
                base_digest: t.base_digest.clone(),
                adapter_digest: t.adapter_digest.clone(),
                tokenizer_digest: t.tokenizer_digest.clone(),
                generation_configuration_digest: t.generation_configuration_digest.clone(),
            };
            if q.artifact != artifact
                || self.principal(&q.artifact_principal)? == self.principal(&a.approver_identity)?
            {
                return Err(RegistryError::Unauthorized);
            }
            if let ApprovalKind::QualifiedEvaluator {
                qualification_digest,
                ..
            } = &a.evidence
            {
                let evaluator =
                    self.qualification(qualification_digest, QualificationPurpose::Evaluator)?;
                if evaluator.artifact == q.artifact
                    || evaluator.artifact.adapter_digest == q.artifact.adapter_digest
                {
                    return Err(RegistryError::Unauthorized);
                }
            }
        }
        Ok(())
    }
    fn record(&self, id: &str) -> Result<ApprovedRecord, RegistryError> {
        let expected = self
            .state
            .records
            .get(id)
            .ok_or_else(|| RegistryError::Missing(id.into()))?;
        let blob = self
            .store
            .get(&format!("training:v1:record:{expected}"))
            .ok_or_else(|| RegistryError::Missing(id.into()))?;
        let envelope: Envelope = serde_json::from_str(blob)?;
        if envelope.version != 1
            || envelope.record.id != id
            || sha256_hex(&canonical_record_bytes(&envelope.record)?) != *expected
        {
            return Err(RegistryError::Invalid("record schema or digest".into()));
        }
        Ok(envelope.record)
    }
    /// Admits only evidence-resolved immutable content. Re-admission cannot undo revocation.
    pub fn admit(&mut self, record: ApprovedRecord) -> Result<(), RegistryError> {
        self.healthy()?;
        self.resolve(&record)?;
        let bytes = canonical_record_bytes(&record)?;
        let digest = sha256_hex(&bytes);
        if let Some(old) = self.state.records.get(&record.id) {
            return if old == &digest {
                Ok(())
            } else {
                Err(RegistryError::Conflict(record.id))
            };
        }
        let key = format!("training:v1:record:{digest}");
        let value = String::from_utf8(bytes).map_err(|e| RegistryError::Invalid(e.to_string()))?;
        if let Some(old) = self.store.get(&key) {
            if old != value {
                return Err(RegistryError::Conflict(key));
            }
        } else {
            self.write(&key, &value)?;
        }
        let mut next = self.state.clone();
        next.records.insert(record.id, digest);
        self.publish(next)
    }
    /// Recomputes the manifest against admitted records and rejects all mismatches.
    /// Snapshot identity is its root digest.
    pub fn register_snapshot(&mut self, manifest: SnapshotManifest) -> Result<(), RegistryError> {
        self.healthy()?;
        self.validate_manifest(&manifest)?;
        for entry in &manifest.records {
            self.not_revoked(&entry.id)?;
            self.resolve(&self.record(&entry.id)?)?;
        }
        validate_split_isolation(
            self.state
                .snapshots
                .values()
                .chain(std::iter::once(&manifest)),
        )?;
        let mut next = self.state.clone();
        if insert(&mut next.snapshots, manifest.root_digest.clone(), manifest)? {
            self.publish(next)?;
        }
        Ok(())
    }
    fn validate_manifest(&self, manifest: &SnapshotManifest) -> Result<(), RegistryError> {
        let mut records = Vec::new();
        let mut families = BTreeMap::new();
        for entry in &manifest.records {
            let record = self.record(&entry.id)?;
            if let Some(old) = families.insert(entry.family_id.clone(), entry.split)
                && old != entry.split
            {
                return Err(RegistryError::Invalid("family split conflict".into()));
            }
            records.push(record);
        }
        let expected = build_snapshot(
            &records,
            &SnapshotPolicy {
                family_splits: families,
                permitted_specification_digests: manifest.specification_digests.clone(),
            },
        )?;
        if manifest != &expected || manifest.records.is_empty() {
            return Err(RegistryError::Invalid(
                "snapshot mismatch or empty snapshot".into(),
            ));
        }
        Ok(())
    }
    fn not_revoked(&self, id: &str) -> Result<(), RegistryError> {
        if self.state.revocations.contains_key(id) {
            Err(RegistryError::Revoked(id.into()))
        } else {
            Ok(())
        }
    }
    /// Irreversible v1 revocation; changing its decision also refuses.
    pub fn revoke(
        &mut self,
        record_id: &str,
        decision: RevocationDecision,
    ) -> Result<(), RegistryError> {
        self.healthy()?;
        self.authorize(
            &decision.operator_identity,
            &self.policy.revocation_operators,
        )?;
        nonblank(&decision.decision_id)?;
        nonblank(&decision.reason)?;
        self.record(record_id)?;
        let mut next = self.state.clone();
        if insert(&mut next.revocations, record_id.into(), decision)? {
            self.publish(next)?;
        }
        Ok(())
    }
    /// Resolves current record evidence and revocations for a registered snapshot.
    pub fn eligibility(&self, snapshot_id: &str) -> Result<Eligibility, RegistryError> {
        self.healthy()?;
        let snapshot = self
            .state
            .snapshots
            .get(snapshot_id)
            .ok_or_else(|| RegistryError::Missing(snapshot_id.into()))?;
        for entry in &snapshot.records {
            self.not_revoked(&entry.id)?;
            self.resolve(&self.record(&entry.id)?)?;
        }
        Ok(Eligibility {
            snapshot_id: snapshot_id.into(),
            registry_revision: self.state.revision,
        })
    }
    /// Resume/publication must check this in the same owned writer session.
    pub fn validate_eligibility(&self, receipt: &Eligibility) -> Result<(), RegistryError> {
        self.healthy()?;
        if self.eligibility(&receipt.snapshot_id)? != *receipt {
            return Err(RegistryError::StaleEligibility);
        }
        Ok(())
    }
    /// Returns a currently eligible manifest for export/run planning.
    pub fn snapshot(&self, snapshot_id: &str) -> Result<SnapshotManifest, RegistryError> {
        self.eligibility(snapshot_id)?;
        self.state
            .snapshots
            .get(snapshot_id)
            .cloned()
            .ok_or_else(|| RegistryError::Missing(snapshot_id.into()))
    }
    /// General inspection/evaluation export after current eligibility validation.
    /// May include held-out records; training/teacher callers must use `training_records`.
    pub fn approved_records(
        &self,
        receipt: &Eligibility,
    ) -> Result<Vec<ApprovedRecord>, RegistryError> {
        self.validate_eligibility(receipt)?;
        let manifest = self.snapshot(&receipt.snapshot_id)?;
        manifest
            .records
            .iter()
            .map(|entry| self.record(&entry.id))
            .collect()
    }
    /// Training/teacher export preserves the exact snapshot identity and refuses
    /// Evaluation or mixed snapshots; build a separate Train snapshot to select data.
    pub fn training_records(
        &self,
        receipt: &Eligibility,
    ) -> Result<Vec<ApprovedRecord>, RegistryError> {
        self.validate_eligibility(receipt)?;
        let manifest = self.snapshot(&receipt.snapshot_id)?;
        validate_split_isolation(self.state.snapshots.values())?;
        train_only(&manifest)?;
        self.approved_records(receipt)
    }
    /// Explicit checkpoint for reopen verification; does not mutate registry revision.
    pub fn checkpoint(&mut self) -> Result<u64, RegistryError> {
        self.healthy()?;
        let result = self
            .store
            .checkpoint()
            .map_err(|e| RegistryError::Storage(e.to_string()));
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
}
/// Retained immutable manifests, including revoked history, own split assignment.
fn validate_split_isolation<'a>(
    manifests: impl IntoIterator<Item = &'a SnapshotManifest>,
) -> Result<(), RegistryError> {
    let mut families = BTreeMap::new();
    let mut records = BTreeMap::new();
    for manifest in manifests {
        for entry in &manifest.records {
            if families
                .insert(entry.family_id.as_str(), entry.split)
                .is_some_and(|old| old != entry.split)
            {
                return Err(RegistryError::Invalid("family split conflict".into()));
            }
            if records
                .insert(entry.id.as_str(), entry.split)
                .is_some_and(|old| old != entry.split)
            {
                return Err(RegistryError::Invalid("record split conflict".into()));
            }
        }
    }
    Ok(())
}

fn frame_size_bound(key_bytes: usize, value_bytes: usize) -> Result<u64, RegistryError> {
    key_bytes
        .checked_add(value_bytes)
        .and_then(|bytes| bytes.checked_mul(6))
        .and_then(|bytes| bytes.checked_add(64))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| RegistryError::Invalid("WAL frame size overflow".into()))
}
fn nonblank(s: &str) -> Result<(), RegistryError> {
    if s.trim().is_empty() {
        Err(RegistryError::Invalid("blank identity".into()))
    } else {
        Ok(())
    }
}
fn digest(s: &str) -> Result<(), RegistryError> {
    if s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        Ok(())
    } else {
        Err(RegistryError::Invalid("invalid digest".into()))
    }
}
fn training_use(s: &str) -> Result<(), RegistryError> {
    if s == "training" {
        Ok(())
    } else {
        Err(RegistryError::Invalid("use must be training".into()))
    }
}
fn insert<T: PartialEq>(
    map: &mut BTreeMap<String, T>,
    key: String,
    value: T,
) -> Result<bool, RegistryError> {
    if let Some(old) = map.get(&key) {
        if old == &value {
            Ok(false)
        } else {
            Err(RegistryError::Conflict(key))
        }
    } else {
        map.insert(key, value);
        Ok(true)
    }
}
fn train_only(manifest: &SnapshotManifest) -> Result<(), RegistryError> {
    if manifest
        .records
        .iter()
        .any(|record| record.split != Split::Train)
    {
        return Err(RegistryError::Invalid(
            "training snapshot contains held-out records".into(),
        ));
    }
    Ok(())
}
mod privacy;
#[cfg(all(test, unix))]
mod tests;

mod runs;
pub use runs::{QualifiedModel, select_model};
