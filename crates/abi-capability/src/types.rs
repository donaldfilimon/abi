use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use thiserror::Error;

/// A fixed SHA-256 digest. Wire form is always `sha256:<64 lowercase hex>`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest([u8; 32]);

impl Digest {
    /// Construct a digest from its fixed bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Return the fixed digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("sha256:")?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl Serialize for Digest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        let hex = text
            .strip_prefix("sha256:")
            .filter(|value| value.len() == 64)
            .ok_or_else(|| serde::de::Error::custom("digest_shape"))?;
        let mut bytes = [0_u8; 32];
        for (index, byte) in bytes.iter_mut().enumerate() {
            let pair = &hex[index * 2..index * 2 + 2];
            *byte = u8::from_str_radix(pair, 16)
                .map_err(|_| serde::de::Error::custom("digest_shape"))?;
        }
        if hex.bytes().any(|byte| byte.is_ascii_uppercase()) {
            return Err(serde::de::Error::custom("digest_shape"));
        }
        Ok(Self(bytes))
    }
}

/// Closed package risk class.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskClass {
    /// Informational only.
    Informational,
    /// Low-risk bounded effect.
    Low,
    /// Medium-risk effect.
    Medium,
    /// High-risk effect.
    High,
    /// Representable for auditable refusal and never grantable.
    Prohibited,
}

/// Closed side-effect class; it is descriptive and never sufficient authority.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SideEffect {
    /// No state effect.
    None,
    /// ABI-owned local state.
    LocalState,
    /// Platform observation.
    PlatformRead,
    /// Platform mutation. Program 2 records this only.
    PlatformWrite,
}

/// Closed reversibility declaration.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Reversibility {
    /// A lossless compensator is declared.
    Reversible,
    /// Compensation may lose information.
    ReversibleWithLoss,
    /// No compensation is possible.
    Irreversible,
}

/// Deterministic approval ladder.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ApprovalLevel {
    /// A live exact grant is enough.
    A0None,
    /// The requesting actor confirms.
    A1Actor,
    /// A platform-permitted manager confirms.
    A2Manager,
    /// A guild administrator confirms.
    A3Admin,
    /// A guild owner confirms.
    A4Owner,
    /// Two distinct administrators or stronger confirm.
    A5DualControl,
}

/// Closed policy decision.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    /// All request-scoped checks passed.
    Allow,
    /// A matching grant exists but needs an approval.
    ApprovalRequired,
    /// Authority is absent or mismatched.
    Deny,
    /// Work must pause without an effect.
    Pause,
}

/// Closed redacted reason taxonomy.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReasonCode {
    /// All checks passed.
    Authorized,
    /// No matching grant exists.
    NoMatchingGrant,
    /// Grant starts in the future.
    GrantNotYetValid,
    /// Grant expired.
    GrantExpired,
    /// Grant is suspended.
    GrantSuspended,
    /// Grant is revoked.
    GrantRevoked,
    /// The request's revocation epoch is stale.
    RevocationEpochMismatch,
    /// Principal does not match.
    PrincipalMismatch,
    /// Exact scope does not match.
    ScopeMismatch,
    /// Capability is unknown.
    CapabilityUnknown,
    /// Capability version differs.
    CapabilityVersionMismatch,
    /// Package digest differs.
    PackageDigestMismatch,
    /// A policy version differs.
    PolicyVersionMismatch,
    /// A live platform permission is absent.
    PlatformPermissionMissing,
    /// Prohibited packages cannot execute.
    Prohibited,
    /// A required approval was absent.
    ApprovalMissing,
    /// Approval expired.
    ApprovalExpired,
    /// Approval does not meet the computed floor or binding.
    ApprovalInsufficient,
    /// Approval decision was previously consumed.
    ApprovalReplayed,
    /// Typed request fields do not match their domain-separated commitment.
    RequestDigestMismatch,
    /// A2+ proposer and approver are the same principal.
    SelfApproval,
    /// Cancellation was observed.
    Cancelled,
    /// Postconditions did not hold in recording facts.
    PostconditionFailed,
}

impl ReasonCode {
    /// Stable lower-snake-case wire label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Authorized => "authorized",
            Self::NoMatchingGrant => "no_matching_grant",
            Self::GrantNotYetValid => "grant_not_yet_valid",
            Self::GrantExpired => "grant_expired",
            Self::GrantSuspended => "grant_suspended",
            Self::GrantRevoked => "grant_revoked",
            Self::RevocationEpochMismatch => "revocation_epoch_mismatch",
            Self::PrincipalMismatch => "principal_mismatch",
            Self::ScopeMismatch => "scope_mismatch",
            Self::CapabilityUnknown => "capability_unknown",
            Self::CapabilityVersionMismatch => "capability_version_mismatch",
            Self::PackageDigestMismatch => "package_digest_mismatch",
            Self::PolicyVersionMismatch => "policy_version_mismatch",
            Self::PlatformPermissionMissing => "platform_permission_missing",
            Self::Prohibited => "prohibited",
            Self::ApprovalMissing => "approval_missing",
            Self::ApprovalExpired => "approval_expired",
            Self::ApprovalInsufficient => "approval_insufficient",
            Self::ApprovalReplayed => "approval_replayed",
            Self::RequestDigestMismatch => "request_digest_mismatch",
            Self::SelfApproval => "self_approval",
            Self::Cancelled => "cancelled",
            Self::PostconditionFailed => "postcondition_failed",
        }
    }
}

/// Closed principal class.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PrincipalKind {
    /// Requesting human.
    HumanSubject,
    /// Organization owner.
    OrganizationOwner,
    /// Guild owner.
    GuildOwner,
    /// Guild administrator.
    GuildAdministrator,
    /// Guild manager.
    GuildManager,
    /// Non-human service.
    Service,
}

/// A pseudonymous principal reference.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Principal {
    /// Opaque principal identifier.
    pub id: String,
    /// Closed principal kind.
    pub kind: PrincipalKind,
}

impl Principal {
    /// Construct a bounded pseudonymous reference.
    pub fn new(id: impl Into<String>, kind: PrincipalKind) -> Result<Self, TypeError> {
        Ok(Self {
            id: bounded_id(id.into())?,
            kind,
        })
    }
}

/// Exact organization/deployment/tenant/platform/resource/subject scope.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Scope {
    /// Organization boundary.
    pub organization_id: String,
    /// Deployment boundary.
    pub deployment_id: String,
    /// Tenant boundary.
    pub tenant_id: String,
    /// Closed adapter platform label.
    pub platform: String,
    /// Exact guild reference; absence is tenant-local, never wildcard.
    pub guild_ref: Option<String>,
    /// Resource selector commitment.
    pub resource_digest: Digest,
    /// Subject selector commitment.
    pub subject_digest: Digest,
}

impl Scope {
    /// Construct an exact bounded scope.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        organization_id: impl Into<String>,
        deployment_id: impl Into<String>,
        tenant_id: impl Into<String>,
        platform: impl Into<String>,
        guild_ref: Option<&str>,
        resource_digest: Digest,
        subject_digest: Digest,
    ) -> Result<Self, TypeError> {
        let platform = platform.into();
        if !matches!(platform.as_str(), "local" | "discord_recording") {
            return Err(TypeError::InvalidField);
        }
        Ok(Self {
            organization_id: bounded_id(organization_id.into())?,
            deployment_id: bounded_id(deployment_id.into())?,
            tenant_id: bounded_id(tenant_id.into())?,
            platform,
            guild_ref: guild_ref.map(str::to_owned).map(bounded_id).transpose()?,
            resource_digest,
            subject_digest,
        })
    }
}

/// A compiled capability declaration consumed by the registry.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CapabilityPackage {
    /// Stable dotted identifier.
    pub id: String,
    /// Exact semantic version.
    pub version: String,
    /// Canonical package commitment.
    pub digest: Digest,
    /// Declared side effect.
    pub side_effect: SideEffect,
    /// Declared compensation strength.
    pub reversibility: Reversibility,
    /// Reviewed risk class.
    pub risk: RiskClass,
    /// Reviewed minimum approval.
    pub approval_floor: ApprovalLevel,
    /// Necessary platform permissions.
    pub required_permissions: BTreeSet<String>,
    /// Typed postcondition identifiers.
    pub postconditions: BTreeSet<String>,
}

impl CapabilityPackage {
    /// Construct and validate a recording-compatible package.
    #[allow(clippy::too_many_arguments)]
    pub fn new<I, P>(
        id: impl Into<String>,
        version: impl Into<String>,
        digest: Digest,
        side_effect: SideEffect,
        reversibility: Reversibility,
        risk: RiskClass,
        approval_floor: ApprovalLevel,
        permissions: I,
        postconditions: P,
    ) -> Result<Self, TypeError>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
        P: IntoIterator,
        P::Item: AsRef<str>,
    {
        let id = id.into();
        if id.len() > 128
            || !id.contains('.')
            || !id
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'.')
        {
            return Err(TypeError::InvalidField);
        }
        let version = version.into();
        if version.split('.').count() != 3
            || !version
                .split('.')
                .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
        {
            return Err(TypeError::InvalidField);
        }
        let required_permissions = bounded_set(permissions)?;
        let postconditions = bounded_set(postconditions)?;
        if side_effect == SideEffect::PlatformWrite && postconditions.is_empty() {
            return Err(TypeError::MissingPostcondition);
        }
        Ok(Self {
            id,
            version,
            digest,
            side_effect,
            reversibility,
            risk,
            approval_floor,
            required_permissions,
            postconditions,
        })
    }
}

/// Grant lifecycle state.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RevocationState {
    /// The grant may be evaluated.
    Active,
    /// The grant is temporarily unusable.
    Suspended,
    /// The grant is terminally unusable.
    Revoked,
}

/// An exact, expiring, policy-versioned grant.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Grant {
    /// Opaque grant identifier.
    pub id: String,
    /// Issuer reference.
    pub issuer: Principal,
    /// Exact recipient.
    pub recipient: Principal,
    /// Exact scope.
    pub scope: Scope,
    /// Exact capability identifier.
    pub capability_id: String,
    /// Exact capability version.
    pub capability_version: String,
    /// Exact package commitment.
    pub package_digest: Digest,
    /// Inclusive start in injected milliseconds.
    pub not_before_ms: u64,
    /// Exclusive expiry in injected milliseconds.
    pub expires_at_ms: u64,
    /// Grant risk ceiling.
    pub risk_ceiling: RiskClass,
    /// Grant confirmation contribution.
    pub confirmation_level: ApprovalLevel,
    /// Current lifecycle.
    pub revocation: RevocationState,
    /// Monotonic revocation epoch.
    pub revocation_epoch: u64,
    /// Exact tenant policy version.
    pub policy_version: String,
    /// Exact guild constitution version.
    pub guild_constitution_version: String,
    /// Exact safety policy version.
    pub safety_policy_version: String,
}

impl Grant {
    /// Construct an active exact grant from a compiled package.
    #[allow(clippy::too_many_arguments)]
    pub fn active(
        id: impl Into<String>,
        issuer: Principal,
        recipient: Principal,
        scope: Scope,
        package: &CapabilityPackage,
        not_before_ms: u64,
        expires_at_ms: u64,
        confirmation_level: ApprovalLevel,
        revocation_epoch: u64,
        policy_version: impl Into<String>,
        guild_version: impl Into<String>,
        safety_version: impl Into<String>,
    ) -> Result<Self, TypeError> {
        let issuer_authorized = if scope.guild_ref.is_none() {
            issuer.kind == PrincipalKind::OrganizationOwner
        } else if package.risk <= RiskClass::Low {
            matches!(
                issuer.kind,
                PrincipalKind::GuildOwner | PrincipalKind::GuildAdministrator
            )
        } else {
            issuer.kind == PrincipalKind::GuildOwner
        };
        if expires_at_ms <= not_before_ms
            || package.risk == RiskClass::Prohibited
            || !issuer_authorized
            || (package.reversibility == Reversibility::Irreversible
                && issuer.kind != PrincipalKind::GuildOwner)
        {
            return Err(TypeError::InvalidField);
        }
        Ok(Self {
            id: bounded_id(id.into())?,
            issuer,
            recipient,
            scope,
            capability_id: package.id.clone(),
            capability_version: package.version.clone(),
            package_digest: package.digest,
            not_before_ms,
            expires_at_ms,
            risk_ceiling: package.risk,
            confirmation_level,
            revocation: RevocationState::Active,
            revocation_epoch,
            policy_version: bounded_id(policy_version.into())?,
            guild_constitution_version: bounded_id(guild_version.into())?,
            safety_policy_version: bounded_id(safety_version.into())?,
        })
    }
}

/// Approval lifecycle state.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ApprovalState {
    /// Explicitly approved.
    Approved,
    /// Explicitly denied.
    Denied,
    /// Cancelled before use.
    Cancelled,
    /// Expired before use.
    Expired,
    /// Already consumed.
    Consumed,
}

/// Exact digest-bound single-use approval.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Approval {
    /// Unique decision identifier.
    pub decision_id: String,
    /// Bound call identifier.
    pub call_id: String,
    /// Bound request digest.
    pub call_digest: Digest,
    /// Bound package.
    pub package_digest: Digest,
    /// Bound grant identifier.
    pub grant_id: String,
    /// Approver reference.
    pub approver: Principal,
    /// Second distinct approver required only for A5 dual control.
    pub coapprover: Option<Principal>,
    /// Satisfied approval level.
    pub level: ApprovalLevel,
    /// Exclusive expiry in injected milliseconds.
    pub expires_at_ms: u64,
    /// Approval state.
    pub state: ApprovalState,
}

/// An opaque tenant-scoped provider reference, never secret material.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CredentialRef {
    /// Opaque reference.
    pub reference_id: String,
    /// Exact tenant boundary.
    pub tenant_id: String,
    /// Closed provider reference.
    pub provider_id: String,
    /// Monotonic version.
    pub version: u64,
}

impl CredentialRef {
    /// Construct an opaque exact tenant/provider/version reference.
    pub fn new(
        reference_id: impl Into<String>,
        tenant_id: impl Into<String>,
        provider_id: impl Into<String>,
        version: u64,
    ) -> Result<Self, TypeError> {
        if version == 0 {
            return Err(TypeError::InvalidField);
        }
        Ok(Self {
            reference_id: bounded_id(reference_id.into())?,
            tenant_id: bounded_id(tenant_id.into())?,
            provider_id: bounded_id(provider_id.into())?,
            version,
        })
    }
}

/// A fully bound model proposal admitted to authorization.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Request {
    /// Attempt identifier.
    pub attempt_id: String,
    /// Call identifier.
    pub call_id: String,
    /// Requesting principal.
    pub principal: Principal,
    /// Exact scope.
    pub scope: Scope,
    /// Capability identifier.
    pub capability_id: String,
    /// Capability version.
    pub capability_version: String,
    /// Expected package commitment.
    pub package_digest: Digest,
    /// Canonical request commitment.
    pub request_digest: Digest,
    /// Parameter commitment; raw parameters are absent.
    pub parameter_digest: Digest,
    /// Exact tenant policy version.
    pub policy_version: String,
    /// Exact guild constitution version.
    pub guild_constitution_version: String,
    /// Exact safety policy version.
    pub safety_policy_version: String,
    /// Expected grant revocation epoch.
    pub revocation_epoch: u64,
}

impl Request {
    /// Construct a bounded, content-free request envelope.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        attempt_id: impl Into<String>,
        call_id: impl Into<String>,
        principal: Principal,
        scope: Scope,
        capability_id: impl Into<String>,
        capability_version: impl Into<String>,
        package_digest: Digest,
        parameter_digest: Digest,
        policy_version: impl Into<String>,
        guild_version: impl Into<String>,
        safety_version: impl Into<String>,
        revocation_epoch: u64,
    ) -> Result<Self, TypeError> {
        let mut request = Self {
            attempt_id: bounded_id(attempt_id.into())?,
            call_id: bounded_id(call_id.into())?,
            principal,
            scope,
            capability_id: capability_id.into(),
            capability_version: capability_version.into(),
            package_digest,
            request_digest: Digest::default(),
            parameter_digest,
            policy_version: bounded_id(policy_version.into())?,
            guild_constitution_version: bounded_id(guild_version.into())?,
            safety_policy_version: bounded_id(safety_version.into())?,
            revocation_epoch,
        };
        request.request_digest = request.computed_digest();
        Ok(request)
    }

    /// Recompute the domain-separated commitment over every typed request field
    /// except the commitment itself.
    #[must_use]
    pub fn computed_digest(&self) -> Digest {
        let encoded = serde_json::to_vec(&(
            &self.attempt_id,
            &self.call_id,
            &self.principal,
            &self.scope,
            &self.capability_id,
            &self.capability_version,
            self.package_digest,
            self.parameter_digest,
            &self.policy_version,
            &self.guild_constitution_version,
            &self.safety_policy_version,
            self.revocation_epoch,
        ))
        .expect("closed request fields serialize");
        let mut hasher = Sha256::new();
        hasher.update(b"abi-capability-request-v2\0");
        hasher.update(encoded);
        let mut output = [0_u8; 32];
        output.copy_from_slice(&hasher.finalize());
        Digest::from_bytes(output)
    }
}

/// Freshly observed platform and policy facts supplied to one request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlatformFacts {
    /// Current permissions.
    pub permissions: BTreeSet<String>,
    /// Current grant revocation epoch.
    pub revocation_epoch: u64,
    /// Current tenant policy version.
    pub policy_version: String,
    /// Current guild constitution version.
    pub guild_constitution_version: String,
    /// Current safety policy version.
    pub safety_policy_version: String,
    /// Current operational-regime approval floor.
    pub regime_approval_floor: ApprovalLevel,
    /// Current safety-policy approval floor.
    pub safety_approval_floor: ApprovalLevel,
}

impl PlatformFacts {
    /// Construct injected fresh facts.
    #[must_use]
    pub fn new(
        permissions: BTreeSet<&str>,
        revocation_epoch: u64,
        policy: &str,
        guild: &str,
        safety: &str,
    ) -> Self {
        Self {
            permissions: permissions.into_iter().map(str::to_owned).collect(),
            revocation_epoch,
            policy_version: policy.to_owned(),
            guild_constitution_version: guild.to_owned(),
            safety_policy_version: safety.to_owned(),
            regime_approval_floor: ApprovalLevel::A0None,
            safety_approval_floor: ApprovalLevel::A0None,
        }
    }

    /// Raise deterministic operational and safety approval contributions.
    #[must_use]
    pub const fn with_approval_floors(
        mut self,
        regime: ApprovalLevel,
        safety: ApprovalLevel,
    ) -> Self {
        self.regime_approval_floor = regime;
        self.safety_approval_floor = safety;
        self
    }
}

/// Injectable time source.
pub trait Clock: Send + Sync {
    /// Current Unix time in milliseconds.
    fn now_ms(&self) -> u64;
}

/// Deterministic test and replay clock.
#[derive(Clone, Copy, Debug)]
pub struct FixedClock(u64);

impl FixedClock {
    /// Construct a fixed clock.
    #[must_use]
    pub const fn new(now_ms: u64) -> Self {
        Self(now_ms)
    }
}

impl Clock for FixedClock {
    fn now_ms(&self) -> u64 {
        self.0
    }
}

/// Redacted terminal receipt.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    /// Attempt identifier.
    pub attempt_id: String,
    /// Request commitment.
    pub request_digest: Digest,
    /// Package commitment.
    pub package_digest: Digest,
    /// Closed decision.
    pub decision: Decision,
    /// Closed reason.
    pub reason: ReasonCode,
    /// Required approval.
    pub required_approval: ApprovalLevel,
    /// Result commitment when a recording was admitted.
    pub result_digest: Option<Digest>,
    /// Postconditions passed.
    pub postconditions_satisfied: bool,
    /// Explicit redaction marker.
    pub redacted: bool,
}

/// Closed construction failures.
#[derive(Debug, Error)]
pub enum TypeError {
    /// A bounded identifier or closed label is invalid.
    #[error("invalid_field")]
    InvalidField,
    /// Platform writes must declare postconditions.
    #[error("missing_postcondition")]
    MissingPostcondition,
}

fn bounded_id(value: String) -> Result<String, TypeError> {
    if value.is_empty()
        || value.len() > 64
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
        })
    {
        return Err(TypeError::InvalidField);
    }
    Ok(value)
}

fn bounded_set<I>(items: I) -> Result<BTreeSet<String>, TypeError>
where
    I: IntoIterator,
    I::Item: AsRef<str>,
{
    let items: BTreeSet<String> = items
        .into_iter()
        .map(|item| item.as_ref().to_owned())
        .collect();
    if items.len() > 32 || items.iter().any(|item| bounded_id(item.clone()).is_err()) {
        return Err(TypeError::InvalidField);
    }
    Ok(items)
}
