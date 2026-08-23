use crate::{CapabilityPackage, RiskClass, SideEffect};
use std::collections::BTreeMap;
use thiserror::Error;

/// Startup-compiled immutable package registry.
#[derive(Clone, Debug)]
pub struct CapabilityRegistry {
    packages: BTreeMap<(String, String), CapabilityPackage>,
}

impl CapabilityRegistry {
    /// Compile exact package identities before any request is admitted.
    pub fn compile<I>(packages: I) -> Result<Self, RegistryError>
    where
        I: IntoIterator<Item = CapabilityPackage>,
    {
        let mut compiled = BTreeMap::new();
        for package in packages {
            if package.side_effect == SideEffect::PlatformWrite && package.postconditions.is_empty()
            {
                return Err(RegistryError::MissingPostcondition);
            }
            let key = (package.id.clone(), package.version.clone());
            if compiled.insert(key, package).is_some() {
                return Err(RegistryError::DuplicatePackage);
            }
        }
        Ok(Self { packages: compiled })
    }

    /// Resolve an exact identifier and version.
    #[must_use]
    pub fn get(&self, id: &str, version: &str) -> Option<&CapabilityPackage> {
        self.packages.get(&(id.to_owned(), version.to_owned()))
    }

    /// Whether an identifier exists at any version.
    #[must_use]
    pub fn contains_id(&self, id: &str) -> bool {
        self.packages.keys().any(|(candidate, _)| candidate == id)
    }

    /// Prohibited packages are representable but never executable.
    #[must_use]
    pub fn is_prohibited(package: &CapabilityPackage) -> bool {
        package.risk == RiskClass::Prohibited
    }
}

/// Closed registry compilation failures.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RegistryError {
    /// Duplicate exact identity.
    #[error("duplicate_package")]
    DuplicatePackage,
    /// A platform write has no typed postcondition.
    #[error("missing_postcondition")]
    MissingPostcondition,
}
