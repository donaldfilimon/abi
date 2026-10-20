//! Closed internal source compilation. These types do not alter the frozen wire corpus.
#![allow(missing_docs)]
use crate::recording::digest_json;
use crate::{
    ApprovalLevel, CapabilityPackage, CapabilityRegistry, Digest, Reversibility, RiskClass, Scope,
    SideEffect,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum DataClass {
    Public,
    Operational,
    Personal,
    Sensitive,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RetentionClass {
    Ephemeral,
    Session,
    Operational,
    Durable,
    MandatoryIncident,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ResidencyPolicy {
    LocalOnly,
    TenantRegion,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum StreamingContract {
    None,
    BoundedEvents,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CancellationContract {
    BeforeEffect,
    Cooperative,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum IdempotencyPolicy {
    NaturallyIdempotent,
    RequiresKey,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DelegationPolicy {
    Forbidden,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RevocationPolicy {
    RecheckBeforeEffect,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RollbackSpec {
    None,
    ReverseOrderStopOnFailure,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BudgetRequirement {
    pub max_input_bytes: u32,
    pub max_output_bytes: u32,
    pub max_events: u32,
    pub max_calls: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CompensationSpec {
    pub capability_id: String,
    pub version: String,
    pub pre_state: BTreeSet<String>,
    pub preauthorized: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterBinding {
    pub adapter_id: String,
    pub api_version: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReceiptSpec {
    pub redacted: bool,
    pub require_postcondition_evidence: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FixtureSet {
    pub deterministic: Digest,
    pub adversarial: Digest,
    pub failure: Digest,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PromotionThresholds {
    pub minimum_passed: u32,
    pub maximum_failed: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExpirationCriteria {
    pub expires_at_ms: u64,
    pub disable_on_schema_drift: bool,
    pub disable_on_missing_evidence: bool,
}

/// Complete source package; typed restrictions intentionally support no delegated authority.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PackageSource {
    pub id: String,
    pub version: String,
    pub input_schema: serde_json::Value,
    pub output_schema: serde_json::Value,
    pub error_schema: serde_json::Value,
    pub streaming: StreamingContract,
    pub cancellation: CancellationContract,
    pub required_platform_permissions: BTreeSet<String>,
    pub resource_scope: Scope,
    pub side_effect: SideEffect,
    pub reversibility: Reversibility,
    pub risk_class: RiskClass,
    pub data_classes: BTreeSet<DataClass>,
    pub residency: ResidencyPolicy,
    pub retention: RetentionClass,
    pub preconditions: BTreeSet<String>,
    pub invariants: BTreeSet<String>,
    pub deadline_ms: u64,
    pub rate_class: String,
    pub budgets: BudgetRequirement,
    pub confirmation_policy: ApprovalLevel,
    pub delegation_policy: DelegationPolicy,
    pub expiry_policy_ms: u64,
    pub revocation_policy: RevocationPolicy,
    pub approval_policy: ApprovalLevel,
    pub idempotency: IdempotencyPolicy,
    pub compensation: Option<CompensationSpec>,
    pub rollback: RollbackSpec,
    pub postconditions: BTreeSet<String>,
    pub adapter_bindings: Vec<AdapterBinding>,
    pub receipt_requirements: ReceiptSpec,
    pub fixtures: FixtureSet,
    pub promotion_thresholds: PromotionThresholds,
    pub expiration_criteria: ExpirationCriteria,
}

/// A constitution can only select a subset of existing exact capabilities and lower ceilings.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GuildConstitution {
    pub version: String,
    pub scope: Scope,
    pub allowed: BTreeSet<(String, String)>,
    pub risk_ceiling: RiskClass,
    pub approval_floor: ApprovalLevel,
    pub data_classes: BTreeSet<DataClass>,
    pub max_deadline_ms: u64,
    pub max_approval_ttl_ms: u64,
    pub budgets: BudgetRequirement,
}

pub struct CompiledSource {
    source: PackageSource,
    package: CapabilityPackage,
    schema_digest: Digest,
    validators: [jsonschema::Validator; 3],
}
impl CompiledSource {
    #[must_use]
    pub const fn source(&self) -> &PackageSource {
        &self.source
    }
    #[must_use]
    pub const fn package(&self) -> &CapabilityPackage {
        &self.package
    }
    #[must_use]
    pub const fn schema_digest(&self) -> Digest {
        self.schema_digest
    }
    #[must_use]
    pub fn validates_input(&self, value: &serde_json::Value) -> bool {
        self.validators[0].is_valid(value)
    }
    #[must_use]
    pub fn validates_output(&self, value: &serde_json::Value) -> bool {
        self.validators[1].is_valid(value)
    }
    #[must_use]
    pub fn validates_error(&self, value: &serde_json::Value) -> bool {
        self.validators[2].is_valid(value)
    }
}

#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq, Eq)]
pub enum SourceError {
    #[error("invalid_package_source")]
    Invalid,
    #[error("invalid_schema")]
    Schema,
    #[error("duplicate_package")]
    Duplicate,
    #[error("invalid_compensator")]
    Compensator,
    #[error("constitution_widening")]
    Widening,
}

/// Startup compiler owns immutable schemas and canonical source commitments.
pub struct SourceRegistry {
    packages: BTreeMap<(String, String), CompiledSource>,
}
impl SourceRegistry {
    pub fn compile(sources: impl IntoIterator<Item = PackageSource>) -> Result<Self, SourceError> {
        let mut packages = BTreeMap::new();
        for source in sources {
            validate_source(&source)?;
            let schemas = [
                &source.input_schema,
                &source.output_schema,
                &source.error_schema,
            ];
            let mut compiled = Vec::new();
            for schema in schemas {
                reject_external_references(schema)?;
                compiled.push(jsonschema::validator_for(schema).map_err(|_| SourceError::Schema)?);
            }
            let schema_digest = digest_json(&("abi.source.schemas.v1", schemas));
            let digest = digest_json(&("abi.source.package.v1", &source));
            let package = CapabilityPackage::new(
                &source.id,
                &source.version,
                digest,
                source.side_effect,
                source.reversibility,
                source.risk_class,
                source.confirmation_policy.max(source.approval_policy),
                &source.required_platform_permissions,
                &source.postconditions,
            )
            .map_err(|_| SourceError::Invalid)?;
            let key = (source.id.clone(), source.version.clone());
            let validators = compiled.try_into().map_err(|_| SourceError::Schema)?;
            if packages
                .insert(
                    key,
                    CompiledSource {
                        source,
                        package,
                        schema_digest,
                        validators,
                    },
                )
                .is_some()
            {
                return Err(SourceError::Duplicate);
            }
        }
        for entry in packages.values() {
            if let Some(compensation) = &entry.source.compensation {
                let inverse = packages
                    .get(&(
                        compensation.capability_id.clone(),
                        compensation.version.clone(),
                    ))
                    .ok_or(SourceError::Compensator)?;
                if inverse.source.risk_class > entry.source.risk_class
                    || inverse.source.resource_scope != entry.source.resource_scope
                    || inverse.source.risk_class == RiskClass::Prohibited
                {
                    return Err(SourceError::Compensator);
                }
            }
        }
        Ok(Self { packages })
    }
    #[must_use]
    pub fn get(&self, id: &str, version: &str) -> Option<&CompiledSource> {
        self.packages.get(&(id.into(), version.into()))
    }
    pub fn capability_registry(&self) -> Result<CapabilityRegistry, crate::RegistryError> {
        CapabilityRegistry::compile(self.packages.values().map(|p| p.package.clone()))
    }
    /// Produce a newly committed narrowed catalogue; old grants deliberately become stale.
    pub fn narrow(&self, constitution: &GuildConstitution) -> Result<Self, SourceError> {
        if constitution.version.is_empty() || constitution.version.len() > 64 {
            return Err(SourceError::Invalid);
        }
        let mut narrowed = Vec::new();
        for key in &constitution.allowed {
            let entry = self.packages.get(key).ok_or(SourceError::Widening)?;
            let mut source = entry.source.clone();
            if source.risk_class != RiskClass::Prohibited
                && (source.risk_class > constitution.risk_ceiling
                    || !source.data_classes.is_subset(&constitution.data_classes))
            {
                continue;
            }
            if source.risk_class == RiskClass::Prohibited
                || source.resource_scope != constitution.scope
                || constitution.approval_floor
                    < source.approval_policy.max(source.confirmation_policy)
                || constitution.max_deadline_ms > source.deadline_ms
                || constitution.max_approval_ttl_ms > source.expiry_policy_ms
                || !budget_narrows(&constitution.budgets, &source.budgets)
            {
                return Err(SourceError::Widening);
            }
            source.approval_policy = constitution.approval_floor;
            source.deadline_ms = constitution.max_deadline_ms;
            source.expiry_policy_ms = constitution.max_approval_ttl_ms;
            source.budgets.clone_from(&constitution.budgets);
            narrowed.push(source);
        }
        let mut compiled = Self::compile(narrowed)?;
        for entry in compiled.packages.values_mut() {
            entry.package.digest =
                digest_json(&("abi.source.guild.v1", entry.package.digest, constitution));
        }
        Ok(compiled)
    }
}
fn budget_narrows(a: &BudgetRequirement, b: &BudgetRequirement) -> bool {
    a.max_input_bytes <= b.max_input_bytes
        && a.max_output_bytes <= b.max_output_bytes
        && a.max_events <= b.max_events
        && a.max_calls <= b.max_calls
}
fn validate_source(s: &PackageSource) -> Result<(), SourceError> {
    let scope = &s.resource_scope;
    Scope::new(
        &scope.organization_id,
        &scope.deployment_id,
        &scope.tenant_id,
        &scope.platform,
        scope.guild_ref.as_deref(),
        scope.resource_digest,
        scope.subject_digest,
    )
    .map_err(|_| SourceError::Invalid)?;
    if serde_json::to_vec(s)
        .map_err(|_| SourceError::Invalid)?
        .len()
        > 1_048_576
        || s.rate_class.is_empty()
        || s.rate_class.len() > 64
        || s.adapter_bindings.iter().any(|a| {
            a.adapter_id.is_empty()
                || a.adapter_id.len() > 64
                || a.api_version.is_empty()
                || a.api_version.len() > 64
        })
    {
        return Err(SourceError::Invalid);
    }

    if s.deadline_ms == 0
        || s.expiry_policy_ms == 0
        || s.expiry_policy_ms > crate::lifecycle::MAX_APPROVAL_TTL_MS
        || s.budgets.max_calls == 0
        || s.budgets.max_input_bytes == 0
        || s.budgets.max_output_bytes == 0
        || s.budgets.max_events == 0
        || s.adapter_bindings.is_empty()
        || s.adapter_bindings.len() > 32
        || !s.receipt_requirements.redacted
        || s.expiration_criteria.expires_at_ms == 0
        || !s.expiration_criteria.disable_on_schema_drift
        || !s.expiration_criteria.disable_on_missing_evidence
        || (s.reversibility == Reversibility::Reversible && s.compensation.is_none())
        || (s.reversibility == Reversibility::Irreversible && s.compensation.is_some())
        || s.compensation.as_ref().is_some_and(|c| {
            c.pre_state.is_empty() || s.rollback != RollbackSpec::ReverseOrderStopOnFailure
        })
        || [&s.preconditions, &s.invariants, &s.postconditions]
            .iter()
            .any(|set| set.len() > 32 || set.iter().any(|v| v.is_empty() || v.len() > 64))
    {
        return Err(SourceError::Invalid);
    }
    Ok(())
}
fn reject_external_references(v: &serde_json::Value) -> Result<(), SourceError> {
    match v {
        serde_json::Value::Object(map) => {
            for (key, value) in map {
                if matches!(
                    key.as_str(),
                    "$id" | "$schema" | "$dynamicRef" | "$recursiveRef"
                ) || (key == "$ref" && !value.as_str().is_some_and(|r| r.starts_with('#')))
                {
                    return Err(SourceError::Schema);
                }
                reject_external_references(value)?;
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                reject_external_references(value)?;
            }
        }
        _ => (),
    }
    Ok(())
}

/// Trusted request-scoped admission facts; raw input is used only for schema validation.
pub struct SourceFacts<'a> {
    pub input: &'a serde_json::Value,
    pub predicates: &'a BTreeSet<String>,
    pub now_ms: u64,
    pub requested_deadline_ms: u64,
    pub requested_budgets: &'a BudgetRequirement,
    pub adapter: &'a AdapterBinding,
    pub idempotency_key: Option<Digest>,
    pub captured_pre_state: Option<Digest>,
}
impl CompiledSource {
    /// Enforce source-specific prerequisites before canonical grant evaluation.
    /// Input hashing remains the host's existing typed request construction contract.
    pub fn check_prerequisites(
        &self,
        request: &crate::Request,
        facts: &SourceFacts<'_>,
    ) -> Result<(), SourceError> {
        let s = &self.source;
        if request.scope != s.resource_scope
            || request.package_digest != self.package.digest
            || request.capability_id != s.id
            || request.capability_version != s.version
            || facts.now_ms >= s.expiration_criteria.expires_at_ms
            || facts.requested_deadline_ms == 0
            || facts.requested_deadline_ms > s.deadline_ms
            || !budget_narrows(facts.requested_budgets, &s.budgets)
            || facts.requested_budgets.max_input_bytes == 0
            || facts.requested_budgets.max_output_bytes == 0
            || facts.requested_budgets.max_events == 0
            || facts.requested_budgets.max_calls == 0
            || !s.preconditions.is_subset(facts.predicates)
            || !s.invariants.is_subset(facts.predicates)
            || !s.adapter_bindings.contains(facts.adapter)
            || (s.idempotency == IdempotencyPolicy::RequiresKey && facts.idempotency_key.is_none())
            || (s.compensation.is_some() && facts.captured_pre_state.is_none())
            || !self.validates_input(facts.input)
            || serde_json::to_vec(facts.input)
                .map_err(|_| SourceError::Invalid)?
                .len()
                > facts
                    .requested_budgets
                    .max_input_bytes
                    .min(s.budgets.max_input_bytes) as usize
        {
            return Err(SourceError::Invalid);
        }
        Ok(())
    }
}
