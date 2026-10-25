//! Ordered, independently encrypted frames; only commitments survive relay restart.
use crate::{
    Authority, Error, JobBinding, NodeCertificate, NodeReply, Relay, RelayKey, Result,
    SealedPayload,
    authority::{audit, hash},
    enrollment::json,
    relay::{node_receipt, owned_node},
    seal,
};
use ed25519_dalek::{Signature, Signer as _, SigningKey, VerifyingKey};
use rusqlite::{OptionalExtension as _, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
/// Versioned encrypted result delta, authenticated by the enrolled worker key.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamFrame {
    /// Zero-based strict sequence; maximum sixteen frames per job.
    pub sequence: u32,
    /// Client-encrypted bytes with exact binding and sequence as associated data.
    pub payload: SealedPayload,
    /// Node signature over binding, sequence and ciphertext.
    pub signature: Vec<u8>,
}
impl StreamFrame {
    fn aad(binding: &JobBinding, sequence: u32) -> Result<Vec<u8>> {
        json(&("abi.relay-stream.v1", binding, sequence)).map(String::into_bytes)
    }
    fn signing_bytes(&self, binding: &JobBinding) -> Result<Vec<u8>> {
        json(&(
            "abi.relay-stream-signature.v1",
            binding,
            self.sequence,
            &self.payload,
        ))
        .map(String::into_bytes)
    }
    /// Encrypt one local provider delta for the original client, then sign exact context.
    pub fn seal(
        binding: &JobBinding,
        sequence: u32,
        bytes: &[u8],
        key: &SigningKey,
    ) -> Result<Self> {
        if sequence >= 16 || bytes.is_empty() || bytes.len() > binding.max_result_bytes {
            return Err(Error::Invalid);
        }
        let payload = seal(&binding.reply_key, &Self::aad(binding, sequence)?, bytes)?;
        let mut frame = Self {
            sequence,
            payload,
            signature: Vec::new(),
        };
        frame.signature = key.sign(&frame.signing_bytes(binding)?).to_bytes().to_vec();
        Ok(frame)
    }
    /// Verify worker identity and context before exposing any plaintext on the client.
    pub fn open(
        &self,
        binding: &JobBinding,
        node_key: &[u8; 32],
        key: &RelayKey,
    ) -> Result<Vec<u8>> {
        self.verify(binding, node_key)?;
        key.open(&Self::aad(binding, self.sequence)?, &self.payload)
    }
    pub(crate) fn verify(&self, binding: &JobBinding, key: &[u8; 32]) -> Result<()> {
        self.payload.validate()?;
        if self.sequence >= 16 || self.payload.ciphertext.len() > binding.max_result_bytes + 16 {
            return Err(Error::Invalid);
        }
        VerifyingKey::from_bytes(key)
            .map_err(|_| Error::Denied)?
            .verify_strict(
                &self.signing_bytes(binding)?,
                &Signature::from_slice(&self.signature).map_err(|_| Error::Denied)?,
            )
            .map_err(|_| Error::Denied)
    }
}
impl Relay {
    pub(crate) fn stream(
        &mut self,
        authority: &mut Authority,
        cert: &NodeCertificate,
        id: &str,
        frame: &StreamFrame,
        now: i64,
    ) -> Result<NodeReply> {
        let memory = self.memory_bytes();
        let tx = authority
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        owned_node(
            &tx,
            &cert.node_id,
            None,
            &authority.key.verifying_key().to_bytes(),
            &authority.audience,
            now,
        )?;
        let receipt = node_receipt(&tx, cert, id)?;
        frame.verify(&receipt.binding, &cert.claim.public_key)?;
        let digest = hash(&json(frame)?);
        let previous: Option<String> = tx
            .query_row(
                "SELECT digest FROM relay_frames WHERE job=?1 AND sequence=?2",
                params![id, frame.sequence],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(previous) = previous {
            if previous != digest {
                return Err(Error::Denied);
            }
            return Ok(NodeReply {
                delivery: None,
                recovery: None,
                dispatch: None,
                receipt: Some(receipt),
                permit: None,
            });
        }
        if receipt.state != "dispatched" {
            return Err(Error::Denied);
        }
        let (count, bytes): (u32, i64) = tx.query_row(
            "SELECT count(*),coalesce(sum(bytes),0) FROM relay_frames WHERE job=?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        let size = frame.payload.ciphertext.len() - 16;
        if frame.sequence != count {
            return Err(Error::Denied);
        }
        let size_sql = i64::try_from(size).map_err(|_| Error::Invalid)?;
        let max_sql =
            i64::try_from(receipt.binding.max_result_bytes).map_err(|_| Error::Invalid)?;
        if bytes + size_sql > max_sql || memory + frame.payload.bytes() > self.limits.memory_bytes {
            return Err(Error::Full);
        }
        tx.execute(
            "INSERT INTO relay_frames(job,sequence,digest,bytes) VALUES(?1,?2,?3,?4)",
            params![id, frame.sequence, digest, size_sql],
        )?;
        audit(&tx, now, "job.frame", &cert.claim.account_id, id)?;
        tx.commit()?;
        self.frames
            .entry(id.into())
            .or_default()
            .push(frame.clone());
        Ok(NodeReply {
            delivery: None,
            recovery: None,
            dispatch: None,
            receipt: Some(receipt),
            permit: None,
        })
    }
    pub(crate) fn stopped(
        authority: &mut Authority,
        cert: &NodeCertificate,
        id: &str,
        now: i64,
    ) -> Result<NodeReply> {
        let tx = authority
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        owned_node(
            &tx,
            &cert.node_id,
            None,
            &authority.key.verifying_key().to_bytes(),
            &authority.audience,
            now,
        )?;
        let mut receipt = node_receipt(&tx, cert, id)?;
        if !matches!(receipt.state.as_str(), "cancel_requested" | "cancelled") {
            return Err(Error::Denied);
        }
        if receipt.state == "cancel_requested" {
            tx.execute("UPDATE relay_jobs SET state='cancelled' WHERE id=?1", [id])?;
            audit(&tx, now, "job.stopped", &cert.claim.account_id, id)?;
            tx.commit()?;
            receipt.state = "cancelled".into();
        }
        Ok(NodeReply {
            delivery: None,
            recovery: None,
            dispatch: None,
            receipt: Some(receipt),
            permit: None,
        })
    }
}
