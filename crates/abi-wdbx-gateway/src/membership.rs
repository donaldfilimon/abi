//! Durable signed gateway-local membership lineage.
//!
//! This reuses WDBX's canonical signed membership records but intentionally
//! does not claim separate-host consensus. The gateway owns a local signing
//! keypair under its configured store root and verifies the complete lineage
//! before serving requests after a restart.

use abi_wdbx::cluster::{
    MembershipChange, MembershipLedger, NodeDescriptor, NodeState, sign_membership_record,
};
use abi_wdbx::v2::{ObjectSecurity, generate_key_material, write_key_material};
use std::fs::{File, OpenOptions};
use std::io::{Read as _, Seek as _, Write as _};
use std::path::{Path, PathBuf};
use uuid::Uuid;

const MAX_RECORD_BYTES: usize = 64 * 1024;
const MAX_RECORDS: usize = 1_024;

#[derive(Debug, thiserror::Error)]
pub(crate) enum MembershipStoreError {
    #[error("membership I/O failed")]
    Io,
    #[error("membership key configuration is incomplete")]
    IncompleteKeys,
    #[error("membership lineage is malformed: {0}")]
    Malformed(&'static str),
    #[error("membership authentication failed")]
    Authentication,
}

pub(crate) struct MembershipStore {
    ledger: MembershipLedger,
    security: ObjectSecurity,
    log_path: PathBuf,
}

pub(crate) struct MembershipOutcome {
    pub(crate) changed: bool,
    pub(crate) active_members: usize,
    pub(crate) generation: u64,
    pub(crate) head_digest: String,
    pub(crate) tombstoned: bool,
}

impl MembershipStore {
    pub(crate) fn open(store_root: &Path) -> Result<Self, MembershipStoreError> {
        let root = store_root.join("gateway-membership");
        std::fs::create_dir_all(&root).map_err(|_| MembershipStoreError::Io)?;
        let security = open_security(&root)?;
        let cluster_id = open_cluster_id(&root)?;
        let log_path = root.join("records.bin");
        let mut ledger = MembershipLedger::new(cluster_id);
        for encoded in read_records(&log_path)? {
            ledger
                .apply_signed(&encoded, &security)
                .map_err(|_| MembershipStoreError::Authentication)?;
        }
        Ok(Self {
            ledger,
            security,
            log_path,
        })
    }

    pub(crate) fn active_len(&self) -> usize {
        self.ledger.active_nodes().len()
    }

    pub(crate) fn add(
        &mut self,
        id: Uuid,
        endpoint: String,
        maximum_members: usize,
    ) -> Result<MembershipOutcome, MembershipStoreError> {
        if let Some(existing) = self.ledger.node(id) {
            if existing.state == NodeState::Active && existing.endpoint == endpoint {
                return Ok(self.outcome(false, id));
            }
            return Err(MembershipStoreError::Malformed(
                "node already exists or is tombstoned",
            ));
        }
        if self.active_len() >= maximum_members {
            return Err(MembershipStoreError::Malformed(
                "gateway membership limit reached",
            ));
        }
        let node = NodeDescriptor::active(id, endpoint)
            .map_err(|_| MembershipStoreError::Malformed("invalid node endpoint"))?;
        self.publish(MembershipChange::Add { node })?;
        Ok(self.outcome(true, id))
    }

    pub(crate) fn remove(&mut self, id: Uuid) -> Result<MembershipOutcome, MembershipStoreError> {
        if self.ledger.node(id).is_none() {
            return Ok(self.outcome(false, id));
        }
        self.publish(MembershipChange::Remove { node_id: id })?;
        Ok(self.outcome(true, id))
    }

    fn publish(&mut self, change: MembershipChange) -> Result<(), MembershipStoreError> {
        let record = self.ledger.next_record(change);
        let signed = sign_membership_record(&record, &self.security)
            .map_err(|_| MembershipStoreError::Authentication)?;
        let mut next = self.ledger.clone();
        next.apply_signed(&signed.encoded, &self.security)
            .map_err(|_| MembershipStoreError::Authentication)?;
        append_record(&self.log_path, &signed.encoded)?;
        self.ledger = next;
        Ok(())
    }

    fn outcome(&self, changed: bool, id: Uuid) -> MembershipOutcome {
        MembershipOutcome {
            changed,
            active_members: self.active_len(),
            generation: self.ledger.generation(),
            head_digest: self.ledger.head_digest().to_owned(),
            tombstoned: self.ledger.is_tombstoned(id),
        }
    }
}

fn open_security(root: &Path) -> Result<ObjectSecurity, MembershipStoreError> {
    let signing = root.join("signing.key");
    let verifying = root.join("verify.key");
    match (signing.is_file(), verifying.is_file()) {
        (false, false) => {
            let (_, signing_material, verifying_material) = generate_key_material();
            write_key_material(&signing, &signing_material)
                .map_err(|_| MembershipStoreError::Io)?;
            if write_key_material(&verifying, &verifying_material).is_err() {
                let _ = std::fs::remove_file(&signing);
                return Err(MembershipStoreError::Io);
            }
        }
        (true, true) => {}
        _ => return Err(MembershipStoreError::IncompleteKeys),
    }
    ObjectSecurity::from_key_files(None, Some(&signing), Some(&verifying))
        .map_err(|_| MembershipStoreError::Authentication)
}

fn open_cluster_id(root: &Path) -> Result<Uuid, MembershipStoreError> {
    let path = root.join("cluster-id");
    if path.is_file() {
        let value = std::fs::read_to_string(path).map_err(|_| MembershipStoreError::Io)?;
        return Uuid::parse_str(value.trim())
            .map_err(|_| MembershipStoreError::Malformed("invalid cluster id"));
    }
    let id = Uuid::new_v4();
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|_| MembershipStoreError::Io)?;
    writeln!(file, "{id}").map_err(|_| MembershipStoreError::Io)?;
    file.sync_all().map_err(|_| MembershipStoreError::Io)?;
    Ok(id)
}

fn read_records(path: &Path) -> Result<Vec<Vec<u8>>, MembershipStoreError> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let mut file = File::open(path).map_err(|_| MembershipStoreError::Io)?;
    let mut records = Vec::new();
    let file_len = file.metadata().map_err(|_| MembershipStoreError::Io)?.len();
    loop {
        let position = file
            .stream_position()
            .map_err(|_| MembershipStoreError::Io)?;
        if position == file_len {
            break;
        }
        if file_len.saturating_sub(position) < 4 {
            return Err(MembershipStoreError::Malformed("torn length prefix"));
        }
        let mut length = [0_u8; 4];
        file.read_exact(&mut length)
            .map_err(|_| MembershipStoreError::Io)?;
        let length = usize::try_from(u32::from_le_bytes(length))
            .map_err(|_| MembershipStoreError::Malformed("record length overflow"))?;
        if length == 0 || length > MAX_RECORD_BYTES || records.len() >= MAX_RECORDS {
            return Err(MembershipStoreError::Malformed("record bounds exceeded"));
        }
        let mut encoded = vec![0_u8; length];
        file.read_exact(&mut encoded)
            .map_err(|_| MembershipStoreError::Malformed("torn membership record"))?;
        records.push(encoded);
    }
    Ok(records)
}

fn append_record(path: &Path, encoded: &[u8]) -> Result<(), MembershipStoreError> {
    let length = u32::try_from(encoded.len())
        .map_err(|_| MembershipStoreError::Malformed("record is too large"))?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|_| MembershipStoreError::Io)?;
    file.write_all(&length.to_le_bytes())
        .and_then(|()| file.write_all(encoded))
        .and_then(|()| file.sync_all())
        .map_err(|_| MembershipStoreError::Io)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("gateway-membership-{}", Uuid::new_v4()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn signed_membership_survives_reopen_and_rejects_tampering() {
        let scratch = Scratch::new();
        let node = Uuid::new_v4();
        let mut store = MembershipStore::open(&scratch.0).unwrap();
        let added = store.add(node, "loopback://node-a".into(), 3).unwrap();
        assert_eq!((added.generation, added.active_members), (1, 1));
        drop(store);

        let mut reopened = MembershipStore::open(&scratch.0).unwrap();
        assert_eq!(reopened.active_len(), 1);
        let removed = reopened.remove(node).unwrap();
        assert!(removed.tombstoned);
        assert_eq!((removed.generation, removed.active_members), (2, 0));
        drop(reopened);
        assert_eq!(MembershipStore::open(&scratch.0).unwrap().active_len(), 0);

        let log = scratch.0.join("gateway-membership/records.bin");
        let mut bytes = std::fs::read(&log).unwrap();
        let last = bytes.last_mut().unwrap();
        *last ^= 0x40;
        std::fs::write(log, bytes).unwrap();
        assert!(matches!(
            MembershipStore::open(&scratch.0),
            Err(MembershipStoreError::Authentication)
        ));
    }
}
