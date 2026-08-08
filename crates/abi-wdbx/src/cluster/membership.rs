//! Signed, monotonically chained cluster membership records.

use crate::v2::{ObjectKind, ObjectSecurity, open_object, seal_object};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use uuid::Uuid;

const MEMBERSHIP_FORMAT_VERSION: u16 = 1;
const GENESIS_DIGEST: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const MAX_ENDPOINT_BYTES: usize = 2_048;
const MAX_SIGNED_RECORD_BYTES: usize = 64 * 1024;

/// Administrative state of one cluster node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeState {
    /// Eligible for shard placement and quorum work.
    Active,
    /// Removed from placement while a resumable rebalance drains data.
    Draining,
    /// Permanently tombstoned in this membership lineage.
    Removed,
}

/// Stable member identity and its advertised transport endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeDescriptor {
    /// Random stable node identity.
    pub id: Uuid,
    /// Operator-configured endpoint. Transport validation happens separately.
    pub endpoint: String,
    /// Administrative lifecycle state.
    pub state: NodeState,
}

impl NodeDescriptor {
    /// Construct an active node after bounded endpoint validation.
    pub fn active(id: Uuid, endpoint: impl Into<String>) -> Result<Self, MembershipError> {
        let descriptor = Self {
            id,
            endpoint: endpoint.into(),
            state: NodeState::Active,
        };
        validate_descriptor(&descriptor)?;
        Ok(descriptor)
    }
}

/// One signed membership transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum MembershipChange {
    /// Add a previously unseen active node.
    Add {
        /// New member descriptor.
        node: NodeDescriptor,
    },
    /// Tombstone an active member and begin draining it.
    Remove {
        /// Member to remove from new placement.
        node_id: Uuid,
    },
}

/// Canonical payload authenticated by the WDBX object envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MembershipRecord {
    /// Membership payload format version.
    pub format_version: u16,
    /// Stable cluster identity.
    pub cluster_id: Uuid,
    /// Exactly one greater than the previously accepted generation.
    pub generation: u64,
    /// SHA-256 of the preceding canonical payload, or all zeroes at genesis.
    pub previous_digest: String,
    /// Administrative transition.
    pub change: MembershipChange,
}

/// Signed envelope plus its canonical payload digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedMembershipRecord {
    /// Authenticated envelope object identity.
    pub object_id: String,
    /// SHA-256 of the canonical record payload.
    pub digest: String,
    /// Complete independently authenticated object bytes.
    pub encoded: Vec<u8>,
}

/// Membership validation, signature, or state-transition failure.
#[derive(Debug, thiserror::Error)]
pub enum MembershipError {
    /// Record serialization failed.
    #[error("membership serialization failed: {0}")]
    Serialization(String),
    /// Authentication or signature verification failed.
    #[error("membership authentication failed: {0}")]
    Authentication(String),
    /// Record structure or bounded fields are invalid.
    #[error("invalid membership record: {0}")]
    Invalid(String),
    /// Record is stale, skips a generation, or forks the accepted lineage.
    #[error("membership lineage mismatch: {0}")]
    Lineage(String),
}

/// Verified membership state for one cluster lineage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MembershipLedger {
    cluster_id: Uuid,
    generation: u64,
    head_digest: String,
    nodes: BTreeMap<Uuid, NodeDescriptor>,
    tombstones: BTreeSet<Uuid>,
}

impl MembershipLedger {
    /// Start an empty membership lineage for `cluster_id`.
    #[must_use]
    pub fn new(cluster_id: Uuid) -> Self {
        Self {
            cluster_id,
            generation: 0,
            head_digest: GENESIS_DIGEST.to_owned(),
            nodes: BTreeMap::new(),
            tombstones: BTreeSet::new(),
        }
    }

    /// Stable cluster identity.
    #[must_use]
    pub const fn cluster_id(&self) -> Uuid {
        self.cluster_id
    }

    /// Latest accepted generation.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Digest of the latest accepted canonical record.
    #[must_use]
    pub fn head_digest(&self) -> &str {
        &self.head_digest
    }

    /// Current descriptor, including draining or removed state.
    #[must_use]
    pub fn node(&self, id: Uuid) -> Option<&NodeDescriptor> {
        self.nodes.get(&id)
    }

    /// Active nodes in stable UUID order.
    #[must_use]
    pub fn active_nodes(&self) -> Vec<NodeDescriptor> {
        self.nodes
            .values()
            .filter(|node| node.state == NodeState::Active)
            .cloned()
            .collect()
    }

    /// Whether an identity was permanently tombstoned in this lineage.
    #[must_use]
    pub fn is_tombstoned(&self, id: Uuid) -> bool {
        self.tombstones.contains(&id)
    }

    /// Build the only record that may validly follow this ledger.
    #[must_use]
    pub fn next_record(&self, change: MembershipChange) -> MembershipRecord {
        MembershipRecord {
            format_version: MEMBERSHIP_FORMAT_VERSION,
            cluster_id: self.cluster_id,
            generation: self.generation.saturating_add(1),
            previous_digest: self.head_digest.clone(),
            change,
        }
    }

    /// Verify a signed envelope before parsing and apply its exact next record.
    pub fn apply_signed(
        &mut self,
        encoded: &[u8],
        security: &ObjectSecurity,
    ) -> Result<SignedMembershipRecord, MembershipError> {
        if encoded.len() > MAX_SIGNED_RECORD_BYTES {
            return Err(MembershipError::Invalid(
                "signed record exceeds 64 KiB".into(),
            ));
        }
        let opened = open_object(encoded, security, true)
            .map_err(|error| MembershipError::Authentication(error.to_string()))?;
        if opened.kind != ObjectKind::Artifact {
            return Err(MembershipError::Invalid(
                "authenticated object kind is not artifact".into(),
            ));
        }
        let record: MembershipRecord = serde_json::from_slice(&opened.plaintext)
            .map_err(|error| MembershipError::Serialization(error.to_string()))?;
        let expected_id = object_id(record.cluster_id, record.generation);
        if opened.object_id != expected_id {
            return Err(MembershipError::Lineage(
                "object identity does not match cluster and generation".into(),
            ));
        }
        self.validate_next(&record)?;
        let digest = digest_hex(&opened.plaintext);
        self.apply_change(record.change)?;
        self.generation = record.generation;
        self.head_digest.clone_from(&digest);
        Ok(SignedMembershipRecord {
            object_id: opened.object_id,
            digest,
            encoded: encoded.to_vec(),
        })
    }

    fn validate_next(&self, record: &MembershipRecord) -> Result<(), MembershipError> {
        if record.format_version != MEMBERSHIP_FORMAT_VERSION {
            return Err(MembershipError::Invalid(
                "unsupported membership format version".into(),
            ));
        }
        if record.cluster_id != self.cluster_id {
            return Err(MembershipError::Lineage("cluster id differs".into()));
        }
        let expected_generation = self.generation.saturating_add(1);
        if record.generation != expected_generation {
            return Err(MembershipError::Lineage(format!(
                "expected generation {expected_generation}, got {}",
                record.generation
            )));
        }
        if record.previous_digest != self.head_digest {
            return Err(MembershipError::Lineage(
                "previous digest does not match accepted head".into(),
            ));
        }
        Ok(())
    }

    fn apply_change(&mut self, change: MembershipChange) -> Result<(), MembershipError> {
        match change {
            MembershipChange::Add { node } => {
                validate_descriptor(&node)?;
                if node.state != NodeState::Active {
                    return Err(MembershipError::Invalid("new node must be active".into()));
                }
                if self.nodes.contains_key(&node.id) || self.tombstones.contains(&node.id) {
                    return Err(MembershipError::Invalid(
                        "node id already exists or is tombstoned".into(),
                    ));
                }
                self.nodes.insert(node.id, node);
            }
            MembershipChange::Remove { node_id } => {
                let node = self.nodes.get_mut(&node_id).ok_or_else(|| {
                    MembershipError::Invalid("cannot remove an unknown node".into())
                })?;
                if node.state != NodeState::Active {
                    return Err(MembershipError::Invalid(
                        "node is already draining or removed".into(),
                    ));
                }
                node.state = NodeState::Draining;
                self.tombstones.insert(node_id);
            }
        }
        Ok(())
    }
}

/// Sign one canonical membership payload and verify the exact published bytes.
pub fn sign_membership_record(
    record: &MembershipRecord,
    security: &ObjectSecurity,
) -> Result<SignedMembershipRecord, MembershipError> {
    validate_record_shape(record)?;
    let plaintext = serde_json::to_vec(record)
        .map_err(|error| MembershipError::Serialization(error.to_string()))?;
    let object_id = object_id(record.cluster_id, record.generation);
    let encoded = seal_object(ObjectKind::Artifact, &object_id, &plaintext, security)
        .map_err(|error| MembershipError::Authentication(error.to_string()))?;
    let opened = open_object(&encoded, security, true)
        .map_err(|error| MembershipError::Authentication(error.to_string()))?;
    if opened.kind != ObjectKind::Artifact
        || opened.object_id != object_id
        || opened.plaintext != plaintext
    {
        return Err(MembershipError::Authentication(
            "published membership object did not verify byte-exact".into(),
        ));
    }
    Ok(SignedMembershipRecord {
        object_id,
        digest: digest_hex(&plaintext),
        encoded,
    })
}

fn validate_record_shape(record: &MembershipRecord) -> Result<(), MembershipError> {
    if record.format_version != MEMBERSHIP_FORMAT_VERSION || record.generation == 0 {
        return Err(MembershipError::Invalid(
            "invalid format version or zero generation".into(),
        ));
    }
    if record.previous_digest.len() != 64
        || !record
            .previous_digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(MembershipError::Invalid(
            "previous digest must be 64 hexadecimal characters".into(),
        ));
    }
    if let MembershipChange::Add { node } = &record.change {
        validate_descriptor(node)?;
    }
    Ok(())
}

fn validate_descriptor(node: &NodeDescriptor) -> Result<(), MembershipError> {
    if node.endpoint.is_empty()
        || node.endpoint.len() > MAX_ENDPOINT_BYTES
        || node.endpoint.chars().any(char::is_control)
    {
        return Err(MembershipError::Invalid(
            "node endpoint must be bounded non-control text".into(),
        ));
    }
    Ok(())
}

fn object_id(cluster_id: Uuid, generation: u64) -> String {
    format!("membership-{cluster_id}-{generation}")
}

fn digest_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::{generate_key_material, write_key_material};

    struct Keys {
        security: ObjectSecurity,
        root: std::path::PathBuf,
    }

    impl Keys {
        fn new(label: &str) -> Self {
            let root = abi_foundation::temp_path::temp_file_path(label, "keys");
            std::fs::create_dir_all(&root).unwrap();
            let (encryption, signing, verifying) = generate_key_material();
            let encryption_path = root.join("enc.key");
            let signing_path = root.join("sign.key");
            let verifying_path = root.join("verify.key");
            write_key_material(&encryption_path, &encryption).unwrap();
            write_key_material(&signing_path, &signing).unwrap();
            write_key_material(&verifying_path, &verifying).unwrap();
            let security = ObjectSecurity::from_key_files(
                Some(&encryption_path),
                Some(&signing_path),
                Some(&verifying_path),
            )
            .unwrap();
            Self { security, root }
        }
    }

    impl Drop for Keys {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.root).unwrap();
        }
    }

    #[test]
    fn signed_generations_apply_in_order_and_removal_is_a_tombstone() {
        let keys = Keys::new("membership-signed");
        let cluster_id = Uuid::new_v4();
        let node = NodeDescriptor::active(Uuid::new_v4(), "127.0.0.1:4001").unwrap();
        let mut ledger = MembershipLedger::new(cluster_id);
        let add = ledger.next_record(MembershipChange::Add { node: node.clone() });
        let signed = sign_membership_record(&add, &keys.security).unwrap();
        ledger
            .apply_signed(&signed.encoded, &keys.security)
            .unwrap();
        assert_eq!(ledger.generation(), 1);
        assert_eq!(ledger.active_nodes(), vec![node.clone()]);

        let remove = ledger.next_record(MembershipChange::Remove { node_id: node.id });
        let signed = sign_membership_record(&remove, &keys.security).unwrap();
        ledger
            .apply_signed(&signed.encoded, &keys.security)
            .unwrap();
        assert_eq!(ledger.active_nodes(), Vec::<NodeDescriptor>::new());
        assert!(ledger.is_tombstoned(node.id));
        assert_eq!(ledger.node(node.id).unwrap().state, NodeState::Draining);
    }

    #[test]
    fn unsigned_wrong_key_tampered_replayed_gapped_and_forked_records_fail_closed() {
        let keys = Keys::new("membership-primary");
        let wrong = Keys::new("membership-wrong");
        let cluster_id = Uuid::new_v4();
        let mut ledger = MembershipLedger::new(cluster_id);
        let node = NodeDescriptor::active(Uuid::new_v4(), "127.0.0.1:4002").unwrap();
        let record = ledger.next_record(MembershipChange::Add { node });

        assert!(sign_membership_record(&record, &ObjectSecurity::plaintext()).is_err());
        let signed = sign_membership_record(&record, &keys.security).unwrap();
        assert!(
            ledger
                .apply_signed(&signed.encoded, &wrong.security)
                .is_err()
        );
        for index in [0, signed.encoded.len() / 2, signed.encoded.len() - 1] {
            let mut tampered = signed.encoded.clone();
            tampered[index] ^= 1;
            assert!(ledger.apply_signed(&tampered, &keys.security).is_err());
        }
        ledger
            .apply_signed(&signed.encoded, &keys.security)
            .unwrap();
        assert!(
            ledger
                .apply_signed(&signed.encoded, &keys.security)
                .is_err()
        );

        let extra = NodeDescriptor::active(Uuid::new_v4(), "127.0.0.1:4003").unwrap();
        let mut gapped = ledger.next_record(MembershipChange::Add {
            node: extra.clone(),
        });
        gapped.generation += 1;
        let signed_gap = sign_membership_record(&gapped, &keys.security).unwrap();
        assert!(
            ledger
                .apply_signed(&signed_gap.encoded, &keys.security)
                .is_err()
        );

        let mut forked = ledger.next_record(MembershipChange::Add { node: extra });
        forked.previous_digest = GENESIS_DIGEST.into();
        let signed_fork = sign_membership_record(&forked, &keys.security).unwrap();
        assert!(
            ledger
                .apply_signed(&signed_fork.encoded, &keys.security)
                .is_err()
        );
        assert_eq!(ledger.generation(), 1);
    }
}
