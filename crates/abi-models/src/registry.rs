//! A validated collection of model manifests.
//!
//! The registry is deliberately thin: it holds manifests keyed by id, refuses
//! duplicates, and is the only place a [`UsableModel`] can be produced.
//!
//! That last point is the design decision worth noting. License acceptance is
//! not a helper a caller may forget to call — [`ModelRegistry::resolve`] is the
//! sole constructor of `UsableModel`, and it consults the
//! [`AcceptanceLedger`](crate::license::AcceptanceLedger) first. "Usable" and
//! "license accepted for this exact revision" are therefore the same statement,
//! enforced by the type system rather than by convention.

use crate::error::ModelError;
use crate::license::AcceptanceLedger;
use crate::manifest::ModelManifest;
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
    /// This is the only constructor of [`UsableModel`].
    pub fn resolve<'a>(
        &'a self,
        id: &str,
        ledger: &AcceptanceLedger,
    ) -> Result<UsableModel<'a>, ModelError> {
        let manifest = self.manifest(id)?;
        ledger.ensure_accepted(manifest)?;
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
