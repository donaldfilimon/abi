//! Recorded license acceptance.
//!
//! Many model licenses require an affirmative act by a named person before the
//! weights may be used. That act is recorded here as durable evidence — who
//! accepted, which license, for which model at which *revision*, and when.
//!
//! Two properties matter and are both tested:
//!
//! - **Acceptance is revision-scoped.** Accepting a license for one revision
//!   does not authorize a different revision, because a republished model can
//!   change its terms.
//! - **The ledger is append-only.** Records are appended as one JSON object per
//!   line; nothing rewrites or deletes earlier lines. Consent evidence that can
//!   be silently edited is not evidence.
//!
//! Nothing here interprets license *text* or judges whether a use is permitted.
//! It records that a human said yes.

use crate::error::ModelError;
use crate::manifest::ModelManifest;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// One recorded acceptance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptanceRecord {
    /// Model identifier the acceptance covers.
    pub model_id: String,
    /// Exact revision the acceptance covers.
    pub revision: String,
    /// License identifier that was accepted.
    pub license: String,
    /// Who accepted, as supplied by the caller.
    pub accepted_by: String,
    /// Wall-clock acceptance time, Unix milliseconds.
    pub accepted_at_unix_ms: i64,
}

impl AcceptanceRecord {
    /// True when this record authorizes exactly this model, revision and license.
    #[must_use]
    pub fn covers(&self, manifest: &ModelManifest) -> bool {
        self.model_id == manifest.id
            && self.revision == manifest.revision.as_str()
            && self.license == manifest.license
    }
}

/// An append-only log of license acceptances.
#[derive(Debug, Clone, Default)]
pub struct AcceptanceLedger {
    /// Backing file, or `None` for an in-memory ledger.
    path: Option<PathBuf>,
    /// Records in acceptance order.
    records: Vec<AcceptanceRecord>,
}

impl AcceptanceLedger {
    /// An unbacked ledger that forgets everything when dropped.
    #[must_use]
    pub fn in_memory() -> Self {
        Self::default()
    }

    /// Load a ledger from disk, treating an absent file as an empty ledger.
    pub fn load(path: impl Into<PathBuf>) -> Result<Self, ModelError> {
        let path = path.into();
        let mut records = Vec::new();
        match std::fs::File::open(&path) {
            Ok(file) => {
                for line in BufReader::new(file).lines() {
                    let line = line.map_err(|source| ModelError::io(&path, source))?;
                    if line.trim().is_empty() {
                        continue;
                    }
                    let record = serde_json::from_str(&line).map_err(|source| {
                        ModelError::json(format!("acceptance ledger {}", path.display()), source)
                    })?;
                    records.push(record);
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => return Err(ModelError::io(&path, source)),
        }
        Ok(Self {
            path: Some(path),
            records,
        })
    }

    /// The backing file, if this ledger is persisted.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Every record, oldest first.
    #[must_use]
    pub fn records(&self) -> &[AcceptanceRecord] {
        &self.records
    }

    /// Record an acceptance and, when persisted, append it to disk.
    ///
    /// The record is appended to the file *before* it is added to the in-memory
    /// list, so a write failure cannot leave the process believing in an
    /// acceptance that was never durably recorded.
    pub fn accept(
        &mut self,
        manifest: &ModelManifest,
        accepted_by: &str,
    ) -> Result<AcceptanceRecord, ModelError> {
        let record = AcceptanceRecord {
            model_id: manifest.id.clone(),
            revision: manifest.revision.as_str().to_owned(),
            license: manifest.license.clone(),
            accepted_by: accepted_by.to_owned(),
            accepted_at_unix_ms: abi_foundation::time::unix_ms(),
        };
        if let Some(path) = &self.path {
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent).map_err(|source| ModelError::io(parent, source))?;
            }
            let line = serde_json::to_string(&record)
                .map_err(|source| ModelError::json("acceptance record", source))?;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|source| ModelError::io(path, source))?;
            writeln!(file, "{line}").map_err(|source| ModelError::io(path, source))?;
        }
        self.records.push(record.clone());
        Ok(record)
    }

    /// Whether this exact model, revision and license has been accepted.
    #[must_use]
    pub fn is_accepted(&self, manifest: &ModelManifest) -> bool {
        self.records.iter().any(|record| record.covers(manifest))
    }

    /// Require acceptance, distinguishing "never accepted" from "accepted under
    /// different terms".
    ///
    /// The second case matters: a republished model that changed its license
    /// must not inherit consent given for the old one.
    pub fn ensure_accepted(&self, manifest: &ModelManifest) -> Result<(), ModelError> {
        if self.is_accepted(manifest) {
            return Ok(());
        }
        let conflicting = self.records.iter().find(|record| {
            record.model_id == manifest.id && record.revision == manifest.revision.as_str()
        });
        if let Some(record) = conflicting {
            return Err(ModelError::LicenseMismatch {
                model: manifest.id.clone(),
                recorded: record.license.clone(),
                declared: manifest.license.clone(),
            });
        }
        Err(ModelError::LicenseNotAccepted {
            model: manifest.id.clone(),
            revision: manifest.revision.as_str().to_owned(),
            license: manifest.license.clone(),
        })
    }
}
