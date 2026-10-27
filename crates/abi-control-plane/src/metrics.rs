//! Content-free operator metrics: fixed keys and counts only, never identifiers or payloads.
use crate::{Authority, Result};
use rusqlite::{OptionalExtension as _, params};
use serde::Serialize;
use std::collections::BTreeMap;

/// Relay job states an operator may see; any other stored value counts as `other`.
const JOB_STATES: [&str; 8] = [
    "queued",
    "offered",
    "dispatched",
    "cancel_requested",
    "cancelled",
    "completed",
    "expired",
    "unresolved",
];
/// Audit kinds written by the authority and relay; unknown kinds count as `other`.
const AUDIT_KINDS: [&str; 11] = [
    "session.created",
    "session.deleted",
    "node.revoked",
    "job.queued",
    "job.offered",
    "job.dispatched",
    "job.frame",
    "job.finished",
    "job.cancelled",
    "job.stopped",
    "relay.recovered",
];

/// Aggregate service state. Keys come from closed internal lists, never from stored text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OperatorMetrics {
    /// Registered accounts.
    pub accounts: u64,
    /// Unexpired browser sessions.
    pub sessions_active: u64,
    /// Enrolled, unrevoked nodes.
    pub nodes_active: u64,
    /// Revoked nodes.
    pub nodes_revoked: u64,
    /// Unconsumed, unexpired pairing requests.
    pub pairings_pending: u64,
    /// Relay jobs by state.
    pub jobs: BTreeMap<&'static str, u64>,
    /// Audit window the event counts cover, in milliseconds before `now`.
    pub audit_window_ms: i64,
    /// Audit events by kind within the window.
    pub audit_events: BTreeMap<&'static str, u64>,
}

impl Authority {
    /// Read-only aggregate snapshot for operators. Contains no private content.
    pub fn operator_metrics(&self, now: i64, window_ms: i64) -> Result<OperatorMetrics> {
        let count = |sql: &str, p: &[&dyn rusqlite::ToSql]| -> Result<u64> {
            let n: i64 = self.db.query_row(sql, p, |r| r.get(0))?;
            Ok(u64::try_from(n).unwrap_or(0))
        };
        let relay_present = self
            .db
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='relay_jobs'",
                [],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        let jobs = if relay_present {
            grouped(
                &self.db,
                "SELECT state,count(*) FROM relay_jobs GROUP BY state",
                &[],
                &JOB_STATES,
            )?
        } else {
            BTreeMap::new()
        };
        Ok(OperatorMetrics {
            accounts: count("SELECT count(*) FROM accounts", &[])?,
            sessions_active: count("SELECT count(*) FROM sessions WHERE expires>?1", &[&now])?,
            nodes_active: count("SELECT count(*) FROM nodes WHERE revoked=0", &[])?,
            nodes_revoked: count("SELECT count(*) FROM nodes WHERE revoked<>0", &[])?,
            pairings_pending: count(
                "SELECT count(*) FROM pairs WHERE consumed=0 AND expires>?1",
                &[&now],
            )?,
            jobs,
            audit_window_ms: window_ms,
            audit_events: grouped(
                &self.db,
                "SELECT kind,count(*) FROM audit WHERE timestamp>?1 GROUP BY kind",
                params![now.saturating_sub(window_ms)],
                &AUDIT_KINDS,
            )?,
        })
    }
}

fn grouped(
    db: &rusqlite::Connection,
    sql: &str,
    p: &[&dyn rusqlite::ToSql],
    allowed: &[&'static str],
) -> Result<BTreeMap<&'static str, u64>> {
    let mut statement = db.prepare(sql)?;
    let rows = statement.query_map(p, |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
    let mut out = BTreeMap::new();
    for row in rows {
        let (label, n) = row?;
        let key = allowed
            .iter()
            .copied()
            .find(|k| *k == label)
            .unwrap_or("other");
        *out.entry(key).or_insert(0) += u64::try_from(n).unwrap_or(0);
    }
    Ok(out)
}
