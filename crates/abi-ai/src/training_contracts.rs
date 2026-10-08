//! Store-agnostic deterministic admission. Qualification digests bind claims;
//! callers must resolve their authorization and qualification in the registry.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

macro_rules! contract_struct {
    ($name:ident { $($field:ident: $ty:ty),* $(,)? }) => {
        #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        #[doc = concat!(stringify!($name), " training contract.")]
        pub struct $name { $(#[doc = concat!("Committed ", stringify!($field), " value.")] pub $field: $ty),* }
    };
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// Closed training contract values.
pub enum TrainingProfile {
    /// Committed contract value or admission refusal.
    Abbey,
    /// Committed contract value or admission refusal.
    Aviva,
    /// Committed contract value or admission refusal.
    Abi,
    /// Committed contract value or admission refusal.
    Quesar,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Closed training contract values.
pub enum Split {
    /// Committed contract value or admission refusal.
    Train,
    /// Committed contract value or admission refusal.
    Evaluation,
}
contract_struct!(SourceRef {
    repository_revision: String,
    document_digest: String,
    requirement_id: String
});
contract_struct!(Permission {
    approved_use_decision_id: String
});
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
/// Closed training contract values.
pub enum ApprovalKind {
    /// Committed contract value or admission refusal.
    Human {},
    /// Committed contract value or admission refusal.
    QualifiedEvaluator {
        /// Committed contract value or admission refusal.
        artifact_digest: String,
        /// Committed contract value or admission refusal.
        qualification_digest: String,
    },
}
contract_struct!(Approval {
    approver_identity: String,
    decision_id: String,
    evidence: ApprovalKind
});
contract_struct!(TeacherIdentity {
    base_digest: String,
    adapter_digest: String,
    tokenizer_digest: String,
    generation_configuration_digest: String,
    qualification_digest: String
});
contract_struct!(ApprovedRecord { id: String, profile: TrainingProfile, family_id: String, input: String, target: String, sources: Vec<SourceRef>, permission: Option<Permission>, approval: Option<Approval>, teacher: Option<TeacherIdentity> });
contract_struct!(SnapshotPolicy { family_splits: BTreeMap<String, Split>, permitted_specification_digests: BTreeSet<String> });
contract_struct!(SnapshotRecord {
    id: String,
    family_id: String,
    profile: TrainingProfile,
    digest: String,
    split: Split
});
contract_struct!(SnapshotManifest { version: u32, records: Vec<SnapshotRecord>, specification_digests: BTreeSet<String>, root_digest: String });
#[derive(Debug, Clone, PartialEq, Eq)]
/// Closed training contract values.
pub enum ContractError {
    /// Committed contract value or admission refusal.
    EmptyIdentity(&'static str),
    /// Committed contract value or admission refusal.
    InvalidDigest(&'static str),
    /// Committed contract value or admission refusal.
    MissingSources,
    /// Committed contract value or admission refusal.
    MissingPermission,
    /// Committed contract value or admission refusal.
    MissingApproval,
    /// Committed contract value or admission refusal.
    UnqualifiedTeacher,
    /// Committed contract value or admission refusal.
    TeacherSelfApproval,
    /// Committed contract value or admission refusal.
    DuplicateId(String),
    /// Committed contract value or admission refusal.
    MissingFamilySplit(String),
    /// Committed contract value or admission refusal.
    TeacherInEvaluation(String),
    /// Committed contract value or admission refusal.
    UnpermittedSpecification(String),
    /// Committed contract value or admission refusal.
    Serialization(String),
    /// Invalid training receipt or transition.
    InvalidRun(String),
}
impl std::fmt::Display for ContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ContractError {}
fn identity(value: &str, field: &'static str) -> Result<(), ContractError> {
    if value.trim().is_empty() {
        Err(ContractError::EmptyIdentity(field))
    } else {
        Ok(())
    }
}
fn digest(value: &str, field: &'static str) -> Result<(), ContractError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        Ok(())
    } else {
        Err(ContractError::InvalidDigest(field))
    }
}
/// Checks structural lineage and independence, not registry authorization.
pub fn validate_record(record: &ApprovedRecord) -> Result<(), ContractError> {
    for (value, field) in [
        (&record.id, "id"),
        (&record.family_id, "family_id"),
        (&record.input, "input"),
        (&record.target, "target"),
    ] {
        identity(value, field)?;
    }
    if record.sources.is_empty() {
        return Err(ContractError::MissingSources);
    }
    for source in &record.sources {
        identity(&source.repository_revision, "repository_revision")?;
        identity(&source.requirement_id, "requirement_id")?;
        digest(&source.document_digest, "document_digest")?;
    }
    identity(
        &record
            .permission
            .as_ref()
            .ok_or(ContractError::MissingPermission)?
            .approved_use_decision_id,
        "approved_use_decision_id",
    )?;
    let approval = record
        .approval
        .as_ref()
        .ok_or(ContractError::MissingApproval)?;
    identity(&approval.approver_identity, "approver_identity")?;
    identity(&approval.decision_id, "approval_decision_id")?;
    if let ApprovalKind::QualifiedEvaluator {
        artifact_digest,
        qualification_digest,
    } = &approval.evidence
    {
        digest(artifact_digest, "evaluator_artifact_digest")?;
        digest(qualification_digest, "evaluator_qualification_digest")?;
    }
    if let Some(teacher) = &record.teacher {
        if teacher.qualification_digest.is_empty() {
            return Err(ContractError::UnqualifiedTeacher);
        }
        for (value, field) in [
            (&teacher.base_digest, "teacher_base_digest"),
            (&teacher.adapter_digest, "teacher_adapter_digest"),
            (&teacher.tokenizer_digest, "teacher_tokenizer_digest"),
            (
                &teacher.generation_configuration_digest,
                "teacher_generation_configuration_digest",
            ),
            (
                &teacher.qualification_digest,
                "teacher_qualification_digest",
            ),
        ] {
            digest(value, field)?;
        }
        if approval.approver_identity == teacher.adapter_digest
            || matches!(&approval.evidence, ApprovalKind::QualifiedEvaluator { artifact_digest, .. } if artifact_digest == &teacher.adapter_digest)
        {
            return Err(ContractError::TeacherSelfApproval);
        }
    }
    Ok(())
}
fn json_bytes(value: &impl Serialize) -> Result<Vec<u8>, ContractError> {
    // BTreeMap-backed JSON values sort every object key recursively. JSON escaping
    // and explicit nulls preserve distinctions without delimiter ambiguity.
    let mut value =
        serde_json::to_value(value).map_err(|e| ContractError::Serialization(e.to_string()))?;
    value.sort_all_objects();
    serde_json::to_vec(&value).map_err(|e| ContractError::Serialization(e.to_string()))
}
/// Exact versioned canonical UTF-8 bytes. Sources have set semantics.
pub fn canonical_record_bytes(record: &ApprovedRecord) -> Result<Vec<u8>, ContractError> {
    validate_record(record)?;
    let mut record = record.clone();
    record.sources.sort_by(|a, b| {
        (
            &a.repository_revision,
            &a.document_digest,
            &a.requirement_id,
        )
            .cmp(&(
                &b.repository_revision,
                &b.document_digest,
                &b.requirement_id,
            ))
    });
    record.sources.dedup();
    json_bytes(&serde_json::json!({"version": 1, "record": record}))
}
/// Lowercase SHA-256 of exact supplied bytes.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        result.push(char::from(HEX[usize::from(byte >> 4)]));
        result.push(char::from(HEX[usize::from(byte & 15)]));
    }
    result
}

/// Builds a policy-assigned snapshot; equal duplicate ids collapse, changed ones refuse.
pub fn build_snapshot(
    records: &[ApprovedRecord],
    policy: &SnapshotPolicy,
) -> Result<SnapshotManifest, ContractError> {
    for spec in &policy.permitted_specification_digests {
        digest(spec, "specification_digest")?;
    }
    for family in policy.family_splits.keys() {
        identity(family, "policy_family_id")?;
    }
    let mut ordered = BTreeMap::new();
    let mut specifications = BTreeSet::new();
    for record in records {
        let bytes = canonical_record_bytes(record)?;
        let split = *policy
            .family_splits
            .get(&record.family_id)
            .ok_or_else(|| ContractError::MissingFamilySplit(record.family_id.clone()))?;
        if record.teacher.is_some() && split == Split::Evaluation {
            return Err(ContractError::TeacherInEvaluation(record.id.clone()));
        }
        for source in &record.sources {
            if !policy
                .permitted_specification_digests
                .contains(&source.document_digest)
            {
                return Err(ContractError::UnpermittedSpecification(
                    source.document_digest.clone(),
                ));
            }
            specifications.insert(source.document_digest.clone());
        }
        if let Some((existing, _)) = ordered.get(&record.id) {
            if existing != &bytes {
                return Err(ContractError::DuplicateId(record.id.clone()));
            }
        } else {
            let record_digest = sha256_hex(&bytes);
            ordered.insert(
                record.id.clone(),
                (
                    bytes,
                    SnapshotRecord {
                        id: record.id.clone(),
                        family_id: record.family_id.clone(),
                        profile: record.profile,
                        digest: record_digest,
                        split,
                    },
                ),
            );
        }
    }
    let mut manifest = SnapshotManifest {
        version: 1,
        records: ordered.into_values().map(|(_, entry)| entry).collect(),
        specification_digests: specifications,
        root_digest: String::new(),
    };
    // Root field excluded from its own preimage, avoiding a circular commitment.
    manifest.root_digest = sha256_hex(&json_bytes(
        &serde_json::json!({"version": manifest.version, "records": manifest.records, "specification_digests": manifest.specification_digests}),
    )?);
    Ok(manifest)
}
#[cfg(test)]
mod tests;

mod receipts;
pub use receipts::*;
