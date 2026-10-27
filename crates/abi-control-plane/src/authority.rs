//! Durable account authority. SQLite transactions atomically mutate metadata and audit.
use crate::types::{Account, Error, Result, Session, VerifiedIdentity};
use ed25519_dalek::SigningKey;
use rusqlite::{Connection, OptionalExtension as _, Transaction, TransactionBehavior, params};
use sha2::{Digest as _, Sha256};
use std::path::Path;

/// Private metadata store. Keep its directory private; do not replace a live file.
/// Unsupported platform ACL validation refuses startup. No automatic reset or repair.
pub struct Authority {
    pub(crate) db: Connection,
    pub(crate) key: SigningKey,
    pub(crate) audience: String,
}
impl Authority {
    /// Open an existing store, or explicitly initialize a new private database.
    /// The audience is bound on first initialization and checked on every reopen.
    pub fn open(path: &Path, initialize: bool, audience: &str) -> Result<Self> {
        if audience.is_empty() || audience.len() > 512 {
            return Err(Error::Invalid);
        }
        let (file, canonical) = crate::private::prepare(path, initialize)?;
        let db = Connection::open_with_flags(
            &canonical,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE | rusqlite::OpenFlags::SQLITE_OPEN_NOFOLLOW,
        )?;
        db.busy_timeout(std::time::Duration::from_secs(3))?;
        db.execute_batch("PRAGMA foreign_keys=ON; PRAGMA secure_delete=ON; PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA trusted_schema=OFF; PRAGMA max_page_count=16384;")?;
        if initialize {
            let secret = secret_bytes();
            db.execute_batch(include_str!("schema.sql"))?;
            db.execute(
                "INSERT INTO config VALUES(1,1,?1,?2)",
                params![audience, &secret],
            )?;
            file.sync_all()?;
            std::fs::File::open(canonical.parent().ok_or(Error::Invalid)?)?.sync_all()?;
        }
        let (version, stored_audience, bytes): (i64, String, Vec<u8>) = db.query_row(
            "SELECT version,audience,signing_key FROM config WHERE id=1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )?;
        if version != 1 || stored_audience != audience {
            return Err(Error::Denied);
        }
        let key = SigningKey::from_bytes(&bytes.try_into().map_err(|_| Error::Denied)?);
        Ok(Self {
            db,
            key,
            audience: audience.into(),
        })
    }
    /// Public authority key for node credential verification (never the signing secret).
    #[must_use]
    pub fn public_key(&self) -> [u8; 32] {
        self.key.verifying_key().to_bytes()
    }
    pub(crate) fn account(&mut self, identity: &VerifiedIdentity, now: i64) -> Result<Account> {
        if identity.subject.is_empty()
            || identity.subject.len() > 1024
            || identity.issuer.len() > 512
        {
            return Err(Error::Denied);
        }
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let account = tx
            .query_row(
                "SELECT id,tenant FROM accounts WHERE issuer=?1 AND subject=?2",
                params![identity.issuer, identity.subject],
                account_row,
            )
            .optional()?;
        let account = if let Some(account) = account {
            account
        } else {
            if count(&tx, "accounts")? >= 10_000 {
                return Err(Error::Full);
            }
            let account = Account {
                account_id: id(),
                tenant_id: id(),
                active_job_limit: 2,
                queued_job_limit: 10,
            };
            tx.execute(
                "INSERT INTO accounts VALUES(?1,?2,?3,?4)",
                params![
                    account.account_id,
                    account.tenant_id,
                    identity.issuer,
                    identity.subject
                ],
            )?;
            audit(
                &tx,
                now,
                "account.created",
                &account.account_id,
                &account.account_id,
            )?;
            account
        };
        tx.commit()?;
        Ok(account)
    }
    pub(crate) fn new_session(&mut self, account: &Account, now: i64) -> Result<Session> {
        let session = Session {
            token: secret(),
            csrf: secret(),
        };
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute("DELETE FROM sessions WHERE expires<=?1", [now])?;
        let n: i64 = tx.query_row(
            "SELECT count(*) FROM sessions WHERE account=?1",
            [&account.account_id],
            |r| r.get(0),
        )?;
        if n >= 16 {
            return Err(Error::Full);
        }
        tx.execute(
            "INSERT INTO sessions VALUES(?1,?2,?3,?4)",
            params![
                hash(&session.token),
                hash(&session.csrf),
                account.account_id,
                now + 28_800
            ],
        )?;
        audit(&tx, now, "session.created", &account.account_id, "")?;
        tx.commit()?;
        Ok(session)
    }
    pub(crate) fn session(&self, token: &str, csrf: Option<&str>, now: i64) -> Result<Account> {
        if token.len() > 128 || csrf.is_some_and(|s| s.len() > 128) {
            return Err(Error::Denied);
        }
        let row: Option<(Account,String)> = self.db.query_row("SELECT a.id,a.tenant,s.csrf FROM sessions s JOIN accounts a ON a.id=s.account WHERE s.token=?1 AND s.expires>?2", params![hash(token),now], |r| Ok((account_row(r)?,r.get(2)?))).optional()?;
        let (account, expected) = row.ok_or(Error::Denied)?;
        if csrf.is_some_and(|s| hash(s) != expected) {
            return Err(Error::Denied);
        }
        Ok(account)
    }
    pub(crate) fn logout(&mut self, token: &str, now: i64) -> Result<()> {
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let actor: String = tx
            .query_row(
                "SELECT account FROM sessions WHERE token=?1",
                [hash(token)],
                |r| r.get(0),
            )
            .optional()?
            .ok_or(Error::Denied)?;
        tx.execute("DELETE FROM sessions WHERE token=?1", [hash(token)])?;
        audit(&tx, now, "session.deleted", &actor, "")?;
        tx.commit()?;
        Ok(())
    }
    pub(crate) fn delete_account(&mut self, account: &Account, now: i64) -> Result<()> {
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        if tx.execute(
            "DELETE FROM accounts WHERE id=?1 AND tenant=?2",
            params![account.account_id, account.tenant_id],
        )? != 1
        {
            return Err(Error::Denied);
        }
        audit(
            &tx,
            now,
            "account.deleted",
            &account.account_id,
            &account.account_id,
        )?;
        tx.commit()?;
        Ok(())
    }
}
fn account_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Account> {
    Ok(Account {
        account_id: row.get(0)?,
        tenant_id: row.get(1)?,
        active_job_limit: 2,
        queued_job_limit: 10,
    })
}
pub(crate) fn id() -> String {
    uuid::Uuid::new_v4().to_string()
}
pub(crate) fn secret_bytes() -> [u8; 32] {
    let mut value = [0; 32];
    value[..16].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
    value[16..].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
    value
}
pub(crate) fn secret() -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(secret_bytes())
}
pub(crate) fn hash(value: &str) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(Sha256::digest(value.as_bytes()))
}
pub(crate) fn audit(
    tx: &Transaction<'_>,
    now: i64,
    kind: &str,
    actor: &str,
    resource: &str,
) -> Result<()> {
    tx.execute(
        "INSERT INTO audit(timestamp,kind,actor,resource) VALUES(?1,?2,?3,?4)",
        params![now, kind, actor, resource],
    )?;
    Ok(())
}
fn count(tx: &Transaction<'_>, table: &str) -> Result<i64> {
    // Table names are internal literals, never request input.
    Ok(tx.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))?)
}
#[cfg(all(test, unix))]
mod metrics_tests;
#[cfg(all(test, unix))]
mod tests;
