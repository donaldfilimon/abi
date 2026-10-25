//! Pairing requires browser ownership plus a fresh local key signature.
use crate::{
    Authority,
    authority::{audit, hash, id, secret},
    types::{
        Account, Error, Node, NodeCertificate, NodeChallenge, PairClaim, PairRequest, PairTicket,
        Result,
    },
};
use ed25519_dalek::{Signature, Signer as _, VerifyingKey};
use rusqlite::{OptionalExtension as _, TransactionBehavior, params};

impl Authority {
    /// Begin a bounded, expiring node enrollment. Does not enroll or authorize a node.
    pub fn begin_pairing(&mut self, request: &PairRequest, now: i64) -> Result<PairTicket> {
        validate_request(request)?;
        let ticket = PairTicket {
            enrollment_id: id(),
            user_code: secret(),
            poll_secret: secret(),
            expires_at: now + 300,
        };
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute("DELETE FROM pairs WHERE expires<=?1", [now])?;
        let count: i64 = tx.query_row("SELECT count(*) FROM pairs", [], |r| r.get(0))?;
        if count >= 1024 {
            return Err(Error::Full);
        }
        tx.execute(
            "INSERT INTO pairs(id,code,poll,request,expires) VALUES(?1,?2,?3,?4,?5)",
            params![
                ticket.enrollment_id,
                hash(&ticket.user_code),
                hash(&ticket.poll_secret),
                json(request)?,
                ticket.expires_at
            ],
        )?;
        tx.commit()?;
        Ok(ticket)
    }
    pub(crate) fn claim_pairing(
        &mut self,
        account: &Account,
        code: &str,
        now: i64,
    ) -> Result<PairClaim> {
        if code.len() > 128 {
            return Err(Error::Denied);
        }
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (enrollment_id,request,expires):(String,String,i64)=tx.query_row("SELECT id,request,expires FROM pairs WHERE code=?1 AND expires>?2 AND claim IS NULL",params![hash(code),now],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?.ok_or(Error::Denied)?;
        let request: PairRequest = decode(&request)?;
        let owned: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM accounts WHERE id=?1 AND tenant=?2)",
            params![account.account_id, account.tenant_id],
            |r| r.get(0),
        )?;
        if !owned {
            return Err(Error::Denied);
        }
        let claim = PairClaim {
            protocol: "abi.local-enrollment.v1".into(),
            enrollment_id,
            account_id: account.account_id.clone(),
            tenant_id: account.tenant_id.clone(),
            public_key: request.public_key,
            encryption_public_key: request.encryption_public_key,
            capabilities: request.capabilities,
            expires_at: expires,
        };
        tx.execute(
            "UPDATE pairs SET claim=?1,account=?2,code=NULL WHERE id=?3",
            params![json(&claim)?, account.account_id, claim.enrollment_id],
        )?;
        audit(
            &tx,
            now,
            "pairing.claimed",
            &account.account_id,
            &claim.enrollment_id,
        )?;
        tx.commit()?;
        Ok(claim)
    }
    /// Poll from the node using the separately issued node-only secret.
    pub fn poll_pairing(
        &self,
        enrollment: &str,
        poll_secret: &str,
        now: i64,
    ) -> Result<Option<PairClaim>> {
        if enrollment.len() > 64 || poll_secret.len() > 128 {
            return Err(Error::Denied);
        }
        let claim: Option<String> = self
            .db
            .query_row(
                "SELECT claim FROM pairs WHERE id=?1 AND poll=?2 AND expires>?3 AND consumed=0",
                params![enrollment, hash(poll_secret), now],
                |r| r.get(0),
            )
            .optional()?
            .ok_or(Error::Denied)?;
        claim.map(|s| decode(&s)).transpose()
    }
    /// Consume local approval once. The signature binds every field in the preview.
    /// A retry returns the existing credential, never creates another node.
    pub fn confirm_pairing(
        &mut self,
        enrollment: &str,
        poll_secret: &str,
        signature: &[u8],
        now: i64,
    ) -> Result<NodeCertificate> {
        if enrollment.len() > 64 || poll_secret.len() > 128 {
            return Err(Error::Denied);
        }
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (encoded,consumed):(String,bool)=tx.query_row("SELECT claim,consumed FROM pairs WHERE id=?1 AND poll=?2 AND expires>?3 AND claim IS NOT NULL",params![enrollment,hash(poll_secret),now],|r|Ok((r.get(0)?,r.get(1)?))).optional()?.ok_or(Error::Denied)?;
        let claim: PairClaim = decode(&encoded)?;
        verify(&claim.public_key, &claim.signing_bytes()?, signature)?;
        if consumed {
            let encoded: String = tx
                .query_row(
                    "SELECT certificate FROM nodes WHERE id=?1 AND revoked=0",
                    [enrollment],
                    |r| r.get(0),
                )
                .optional()?
                .ok_or(Error::Denied)?;
            let cert = decode(&encoded)?;
            tx.commit()?;
            return Ok(cert);
        }
        let count: i64 = tx.query_row(
            "SELECT count(*) FROM nodes WHERE account=?1 AND revoked=0",
            [&claim.account_id],
            |r| r.get(0),
        )?;
        if count >= 32 {
            return Err(Error::Full);
        }
        let mut certificate = NodeCertificate {
            protocol: "abi.node-certificate.v1".into(),
            audience: self.audience.clone(),
            node_id: enrollment.into(),
            claim,
            expires_at: now + 3600,
            signature: Vec::new(),
        };
        certificate.signature = self
            .key
            .sign(&certificate.signing_bytes()?)
            .to_bytes()
            .to_vec();
        tx.execute(
            "INSERT INTO nodes VALUES(?1,?2,?3,0)",
            params![
                certificate.node_id,
                certificate.claim.account_id,
                json(&certificate)?
            ],
        )?;
        tx.execute("UPDATE pairs SET consumed=1 WHERE id=?1", [enrollment])?;
        audit(
            &tx,
            now,
            "node.enrolled",
            &certificate.claim.account_id,
            &certificate.node_id,
        )?;
        tx.commit()?;
        Ok(certificate)
    }
    pub(crate) fn nodes(&self, account: &Account) -> Result<Vec<Node>> {
        let mut stmt = self
            .db
            .prepare("SELECT certificate,revoked FROM nodes WHERE account=?1 ORDER BY id")?;
        let rows = stmt.query_map([&account.account_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, bool>(1)?))
        })?;
        rows.map(|r| {
            let (encoded, revoked) = r?;
            let cert: NodeCertificate = decode(&encoded)?;
            Ok(Node {
                node_id: cert.node_id,
                capabilities: cert.claim.capabilities,
                expires_at: cert.expires_at,
                revoked,
                encryption_public_key: cert.claim.encryption_public_key,
            })
        })
        .collect()
    }
    pub(crate) fn revoke(&mut self, account: &Account, node: &str, now: i64) -> Result<()> {
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        if tx.execute(
            "UPDATE nodes SET revoked=1 WHERE id=?1 AND account=?2",
            params![node, account.account_id],
        )? != 1
        {
            return Err(Error::Denied);
        }
        tx.execute("DELETE FROM challenges WHERE node=?1", [node])?;
        audit(&tx, now, "node.revoked", &account.account_id, node)?;
        tx.commit()?;
        Ok(())
    }
    /// Construct a local signed-nonce preview without reserving server resources.
    /// Clients may also construct this contract locally. No HTTP challenge issuance is needed.
    pub fn challenge(&self, node: &str, now: i64) -> Result<NodeChallenge> {
        if node.len() > 64 {
            return Err(Error::Denied);
        }
        let encoded: String = self
            .db
            .query_row(
                "SELECT certificate FROM nodes WHERE id=?1 AND revoked=0",
                [node],
                |r| r.get(0),
            )
            .optional()?
            .ok_or(Error::Denied)?;
        let cert: NodeCertificate = decode(&encoded)?;
        cert.verify(&self.public_key(), &self.audience, now)?;
        Ok(NodeChallenge {
            protocol: "abi.node-proof.v1".into(),
            audience: self.audience.clone(),
            node_id: node.into(),
            nonce: secret(),
            expires_at: now + 60,
        })
    }
    /// Authenticate a node-signed, expiring nonce once against fresh authority state.
    /// Nonces are recorded only after possession verification. Returns metadata,
    /// never a reusable dispatch authorization or a browser session.
    pub fn authenticate_node(
        &mut self,
        challenge: &NodeChallenge,
        signature: &[u8],
        now: i64,
    ) -> Result<NodeCertificate> {
        if challenge.nonce.len() < 32
            || challenge.nonce.len() > 128
            || challenge.node_id.len() > 64
            || challenge.protocol != "abi.node-proof.v1"
            || challenge.audience != self.audience
            || challenge.expires_at <= now
            || challenge.expires_at > now + 60
        {
            return Err(Error::Denied);
        }
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let encoded:String=tx.query_row("SELECT n.certificate FROM nodes n JOIN accounts a ON a.id=n.account WHERE n.id=?1 AND n.revoked=0",[&challenge.node_id],|r|r.get(0)).optional()?.ok_or(Error::Denied)?;
        let certificate: NodeCertificate = decode(&encoded)?;
        certificate.verify(&self.key.verifying_key().to_bytes(), &self.audience, now)?;
        verify(
            &certificate.claim.public_key,
            &challenge.signing_bytes()?,
            signature,
        )?;
        tx.execute("DELETE FROM challenges WHERE expires<=?1", [now])?;
        let count: i64 = tx.query_row(
            "SELECT count(*) FROM challenges WHERE node=?1",
            [&challenge.node_id],
            |r| r.get(0),
        )?;
        if count >= 64 {
            return Err(Error::Full);
        }
        let inserted = tx.execute(
            "INSERT OR IGNORE INTO challenges VALUES(?1,?2,?3,?4)",
            params![
                hash(&challenge.nonce),
                challenge.node_id,
                json(challenge)?,
                challenge.expires_at
            ],
        )?;
        if inserted != 1 {
            return Err(Error::Denied);
        }
        tx.commit()?;
        Ok(certificate)
    }
}
fn validate_request(request: &PairRequest) -> Result<()> {
    VerifyingKey::from_bytes(&request.public_key).map_err(|_| Error::Invalid)?;
    if request.capabilities.is_empty()
        || request.capabilities.len() > 4
        || request.capabilities.windows(2).any(|p| p[0] >= p[1])
        || request.capabilities.iter().any(|c| {
            !matches!(
                c.as_str(),
                "model.infer" | "memory.retrieve" | "tool.script" | "tool.execute"
            )
        })
    {
        return Err(Error::Invalid);
    }
    Ok(())
}
fn verify(key: &[u8; 32], bytes: &[u8], signature: &[u8]) -> Result<()> {
    VerifyingKey::from_bytes(key)
        .map_err(|_| Error::Denied)?
        .verify_strict(
            bytes,
            &Signature::from_slice(signature).map_err(|_| Error::Denied)?,
        )
        .map_err(|_| Error::Denied)
}
pub(crate) fn json<T: serde::Serialize>(value: &T) -> Result<String> {
    serde_json::to_string(value).map_err(|_| Error::Invalid)
}
pub(crate) fn decode<T: serde::de::DeserializeOwned>(value: &str) -> Result<T> {
    serde_json::from_str(value).map_err(|_| Error::Denied)
}
