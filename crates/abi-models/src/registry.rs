//! A validated collection of model manifests.
//!
//! The registry is deliberately thin: it holds manifests keyed by id, refuses
//! duplicates, and is the only place a [`UsableModel`] can be produced.
//!
//! That last point is the design decision worth noting. License acceptance is
//! not a helper a caller may forget to call — [`ModelRegistry::resolve`] and
//! [`ModelRegistry::resolve_for`] are the only constructors of `UsableModel`,
//! and both consult the [`AcceptanceLedger`] first. "Usable" and
//! "license accepted for this exact revision" are therefore the same statement,
//! enforced by the type system rather than by convention.

use crate::error::ModelError;
use crate::license::AcceptanceLedger;
use crate::manifest::ModelManifest;
use crate::signed::{PublisherTrustStore, SignedManifestEnvelope};
use std::collections::BTreeMap;
use std::path::Path;

/// A set of validated manifests, keyed by model id.
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    /// Ordered by id so listings and errors are deterministic.
    models: BTreeMap<String, ModelManifest>,
}

impl ModelRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a manifest, refusing a duplicate identifier.
    pub fn insert(&mut self, manifest: ModelManifest) -> Result<(), ModelError> {
        if self.models.contains_key(&manifest.id) {
            return Err(ModelError::DuplicateModel {
                id: manifest.id.clone(),
            });
        }
        self.models.insert(manifest.id.clone(), manifest);
        Ok(())
    }

    /// Load a registry from a JSON array of manifest documents.
    ///
    /// Each element is validated individually, so one bad manifest names itself
    /// in the resulting error rather than failing the whole document anonymously.
    pub fn from_json_array(context: &str, document: &str) -> Result<Self, ModelError> {
        let values: Vec<serde_json::Value> =
            serde_json::from_str(document).map_err(|source| ModelError::json(context, source))?;
        let mut registry = Self::new();
        for (index, value) in values.iter().enumerate() {
            let element = format!("{context}[{index}]");
            let manifest = ModelManifest::from_json(&element, &value.to_string())?;
            registry.insert(manifest)?;
        }
        Ok(registry)
    }

    /// Load a JSON array of publisher-signed manifest envelopes.
    pub fn from_signed_json_array(
        context: &str,
        document: &str,
        trust_store: &PublisherTrustStore,
    ) -> Result<Self, ModelError> {
        let values: Vec<serde_json::Value> =
            serde_json::from_str(document).map_err(|source| ModelError::json(context, source))?;
        let mut registry = Self::new();
        for (index, value) in values.iter().enumerate() {
            let element = format!("{context}[{index}]");
            let envelope = SignedManifestEnvelope::from_json(&element, &value.to_string())?;
            let manifest = envelope.verify(&format!("{element}.manifest_json"), trust_store)?;
            registry.insert(manifest)?;
        }
        Ok(registry)
    }

    /// Load every `*.json` manifest in a directory.
    ///
    /// Entries are sorted before parsing, because directory iteration order is
    /// not defined by the OS and an unsorted walk would make duplicate-id errors
    /// nondeterministic.
    pub fn from_dir(dir: &Path) -> Result<Self, ModelError> {
        let mut paths = Vec::new();
        let entries = std::fs::read_dir(dir).map_err(|source| ModelError::io(dir, source))?;
        for entry in entries {
            let entry = entry.map_err(|source| ModelError::io(dir, source))?;
            let path = entry.path();
            if path.extension().and_then(std::ffi::OsStr::to_str) == Some("json") {
                paths.push(path);
            }
        }
        paths.sort();

        let mut registry = Self::new();
        for path in paths {
            let text =
                std::fs::read_to_string(&path).map_err(|source| ModelError::io(&path, source))?;
            let manifest = ModelManifest::from_json(&path.display().to_string(), &text)?;
            registry.insert(manifest)?;
        }
        Ok(registry)
    }

    /// Look up a manifest.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&ModelManifest> {
        self.models.get(id)
    }

    /// Look up a manifest, erroring when it is absent.
    pub fn manifest(&self, id: &str) -> Result<&ModelManifest, ModelError> {
        self.models
            .get(id)
            .ok_or_else(|| ModelError::UnknownModel { id: id.to_owned() })
    }

    /// Every model id, sorted.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.models.keys().map(String::as_str)
    }

    /// Number of registered models.
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Whether the registry holds no models.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// Resolve a model for use, requiring recorded license acceptance.
    ///
    /// Use [`Self::resolve_for`] when the caller must prove which principal
    /// supplied the acceptance.
    pub fn resolve<'a>(
        &'a self,
        id: &str,
        ledger: &AcceptanceLedger,
    ) -> Result<UsableModel<'a>, ModelError> {
        let manifest = self.manifest(id)?;
        ledger.ensure_accepted(manifest)?;
        Ok(UsableModel { manifest })
    }

    /// Resolve a model for one exact accepting principal.
    pub fn resolve_for<'a>(
        &'a self,
        id: &str,
        ledger: &AcceptanceLedger,
        principal: &str,
    ) -> Result<UsableModel<'a>, ModelError> {
        let manifest = self.manifest(id)?;
        ledger.ensure_accepted_for(manifest, principal)?;
        Ok(UsableModel { manifest })
    }
}

/// A manifest whose license has been accepted for its exact revision.
///
/// Holding one of these is proof the acceptance check ran and passed; there is
/// no other way to obtain the type.
#[derive(Debug, Clone, Copy)]
pub struct UsableModel<'a> {
    /// The underlying manifest.
    manifest: &'a ModelManifest,
}

impl<'a> UsableModel<'a> {
    /// The manifest behind this authorization.
    #[must_use]
    pub fn manifest(&self) -> &'a ModelManifest {
        self.manifest
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{OTHER_REVISION, Scratch, manifest, manifest_value, parse, set_revision};
    use serde_json::json;

    /// A second manifest with a distinct id.
    fn second_manifest() -> ModelManifest {
        let mut value = manifest_value();
        value["id"] = json!("another-7b");
        set_revision(&mut value, OTHER_REVISION);
        parse(&value)
    }

    #[test]
    fn a_json_array_loads_into_a_sorted_registry() {
        let document = json!([second_manifest(), manifest()]).to_string();
        let registry = ModelRegistry::from_json_array("array", &document).expect("loads");
        assert_eq!(registry.len(), 2);
        assert!(!registry.is_empty());
        assert_eq!(
            registry.ids().collect::<Vec<_>>(),
            ["another-7b", "example-2b"]
        );
        assert_eq!(
            registry.get("example-2b").expect("present").id,
            "example-2b"
        );
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let document = json!([manifest(), manifest()]).to_string();
        let error = ModelRegistry::from_json_array("array", &document)
            .expect_err("two manifests may not claim one id");
        assert!(
            matches!(error, ModelError::DuplicateModel { ref id } if id == "example-2b"),
            "{error:?}"
        );
    }

    #[test]
    fn an_invalid_element_names_its_position() {
        let mut bad = manifest_value();
        bad["revision"] = json!("main");
        let document = json!([manifest(), bad]).to_string();
        let error = ModelRegistry::from_json_array("catalog", &document)
            .expect_err("a floating ref anywhere fails the whole load");
        assert!(
            matches!(error, ModelError::FloatingRevision { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn a_directory_of_manifests_loads_in_sorted_order() {
        let scratch = Scratch::new("registry_dir");
        // Written in reverse order, and with a non-JSON file that must be ignored.
        scratch.write(
            "z-second.json",
            second_manifest().to_json().expect("json").as_bytes(),
        );
        scratch.write(
            "a-first.json",
            manifest().to_json().expect("json").as_bytes(),
        );
        scratch.write("notes.txt", b"not a manifest");

        let registry = ModelRegistry::from_dir(scratch.path()).expect("loads");
        assert_eq!(registry.len(), 2);
        assert_eq!(
            registry.ids().collect::<Vec<_>>(),
            ["another-7b", "example-2b"]
        );
    }

    #[test]
    fn a_missing_directory_is_an_io_error() {
        let scratch = Scratch::new("registry_absent");
        let error = ModelRegistry::from_dir(&scratch.join("nope")).expect_err("no such directory");
        assert!(matches!(error, ModelError::Io { .. }), "{error:?}");
    }

    #[test]
    fn an_unknown_id_is_an_error_not_a_none() {
        let registry = ModelRegistry::new();
        assert!(registry.get("absent").is_none());
        let error = registry.manifest("absent").expect_err("unknown");
        assert!(matches!(error, ModelError::UnknownModel { ref id } if id == "absent"));
    }

    #[test]
    fn resolving_an_unaccepted_model_is_refused() {
        let mut registry = ModelRegistry::new();
        registry.insert(manifest()).expect("inserts");
        let ledger = AcceptanceLedger::in_memory();

        let error = registry
            .resolve("example-2b", &ledger)
            .expect_err("a model with no recorded acceptance must not be usable");
        assert!(
            matches!(error, ModelError::LicenseNotAccepted { .. }),
            "{error:?}"
        );
    }

    #[test]
    fn resolving_succeeds_only_after_acceptance() {
        let mut registry = ModelRegistry::new();
        registry.insert(manifest()).expect("inserts");
        let mut ledger = AcceptanceLedger::in_memory();
        assert!(registry.resolve("example-2b", &ledger).is_err());

        ledger.accept(&manifest(), "operator").expect("accepts");
        let usable = registry.resolve("example-2b", &ledger).expect("now usable");
        assert_eq!(usable.manifest().id, "example-2b");
        assert_eq!(
            usable.manifest().revision.as_str(),
            manifest().revision.as_str()
        );
    }

    #[test]
    fn resolving_an_unknown_model_reports_the_id_not_the_license() {
        let registry = ModelRegistry::new();
        let ledger = AcceptanceLedger::in_memory();
        let error = registry.resolve("ghost", &ledger).expect_err("unknown");
        assert!(
            matches!(error, ModelError::UnknownModel { .. }),
            "{error:?}"
        );
    }
}
