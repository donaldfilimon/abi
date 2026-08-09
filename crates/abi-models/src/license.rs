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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{OTHER_REVISION, Scratch, manifest, manifest_value, parse};
    use serde_json::json;

    /// The fixture manifest republished at a different revision.
    fn at_other_revision() -> ModelManifest {
        let mut value = manifest_value();
        value["revision"] = json!(OTHER_REVISION);
        parse(&value)
    }

    #[test]
    fn an_unaccepted_model_is_refused() {
        let ledger = AcceptanceLedger::in_memory();
        let manifest = manifest();
        assert!(!ledger.is_accepted(&manifest));

        let error = ledger
            .ensure_accepted(&manifest)
            .expect_err("an unaccepted model must not be usable");
        assert!(
            matches!(error, ModelError::LicenseNotAccepted { ref license, .. }
                if license == "example-community-license-1.0"),
            "{error:?}"
        );
    }

    #[test]
    fn acceptance_records_who_what_and_when() {
        let mut ledger = AcceptanceLedger::in_memory();
        let manifest = manifest();
        let before = abi_foundation::time::unix_ms();

        let record = ledger
            .accept(&manifest, "operator@example.invalid")
            .expect("accepts");
        assert_eq!(record.model_id, manifest.id);
        assert_eq!(record.revision, manifest.revision.as_str());
        assert_eq!(record.license, manifest.license);
        assert_eq!(record.accepted_by, "operator@example.invalid");
        assert!(record.accepted_at_unix_ms >= before);

        ledger.ensure_accepted(&manifest).expect("now accepted");
        assert_eq!(ledger.records().len(), 1);
    }

    #[test]
    fn acceptance_does_not_carry_to_another_revision() {
        let mut ledger = AcceptanceLedger::in_memory();
        ledger.accept(&manifest(), "operator").expect("accepts");

        let republished = at_other_revision();
        assert!(!ledger.is_accepted(&republished));
        let error = ledger
            .ensure_accepted(&republished)
            .expect_err("a different revision must require its own acceptance");
        assert!(
            matches!(error, ModelError::LicenseNotAccepted { ref revision, .. }
                if revision == OTHER_REVISION),
            "{error:?}"
        );
    }

    #[test]
    fn acceptance_does_not_carry_across_a_changed_license() {
        let mut ledger = AcceptanceLedger::in_memory();
        ledger.accept(&manifest(), "operator").expect("accepts");

        // Same model, same revision, relicensed by the publisher.
        let mut value = manifest_value();
        value["license"] = json!("example-restrictive-license-2.0");
        let relicensed = parse(&value);

        let error = ledger
            .ensure_accepted(&relicensed)
            .expect_err("consent for the old terms must not transfer");
        assert!(
            matches!(error, ModelError::LicenseMismatch { ref recorded, ref declared, .. }
                if recorded == "example-community-license-1.0"
                    && declared == "example-restrictive-license-2.0"),
            "{error:?}"
        );
    }

    #[test]
    fn a_missing_ledger_file_loads_as_empty() {
        let scratch = Scratch::new("ledger_absent");
        let path = scratch.join("nested/accepted.jsonl");
        let ledger = AcceptanceLedger::load(&path).expect("an absent ledger is simply empty");
        assert!(ledger.records().is_empty());
        assert_eq!(ledger.path(), Some(path.as_path()));
        assert!(!path.exists(), "loading must not create the file");
    }

    #[test]
    fn acceptance_persists_across_a_reload() {
        let scratch = Scratch::new("ledger_persist");
        let path = scratch.join("accepted.jsonl");
        let manifest = manifest();

        let mut ledger = AcceptanceLedger::load(&path).expect("loads");
        ledger.accept(&manifest, "operator").expect("accepts");

        let reloaded = AcceptanceLedger::load(&path).expect("reloads");
        assert_eq!(reloaded.records().len(), 1);
        reloaded
            .ensure_accepted(&manifest)
            .expect("acceptance survived the reload");
    }

    #[test]
    fn the_ledger_is_append_only() {
        let scratch = Scratch::new("ledger_append");
        let path = scratch.join("accepted.jsonl");

        let mut ledger = AcceptanceLedger::load(&path).expect("loads");
        ledger
            .accept(&manifest(), "first-operator")
            .expect("accepts");
        let after_first = std::fs::read_to_string(&path).expect("readable");

        ledger
            .accept(&at_other_revision(), "second-operator")
            .expect("accepts");
        let after_second = std::fs::read_to_string(&path).expect("readable");

        assert!(
            after_second.starts_with(&after_first),
            "an earlier record must never be rewritten"
        );
        assert_eq!(after_second.lines().count(), 2);
        assert_eq!(
            AcceptanceLedger::load(&path)
                .expect("reloads")
                .records()
                .len(),
            2
        );
    }

    #[test]
    fn a_corrupt_ledger_line_is_reported_rather_than_skipped() {
        let scratch = Scratch::new("ledger_corrupt");
        let path = scratch.write("accepted.jsonl", b"{\"model_id\": \"x\"}\n");
        let error = AcceptanceLedger::load(&path).expect_err("must not silently drop the record");
        assert!(matches!(error, ModelError::Json { .. }), "{error:?}");
    }
}
