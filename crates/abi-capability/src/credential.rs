use crate::{CredentialRef, Digest};
use std::collections::BTreeMap;

/// Synthetic resolver storing opaque handles only, with exact tenant/version matching.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TenantCredentialResolver {
    handles: BTreeMap<(String, String, u64), Digest>,
}

impl TenantCredentialResolver {
    /// Construct a local resolver from synthetic opaque-handle commitments.
    #[must_use]
    pub fn new<I>(entries: I) -> Self
    where
        I: IntoIterator<Item = (CredentialRef, Digest)>,
    {
        Self {
            handles: entries
                .into_iter()
                .map(|(reference, handle)| {
                    (
                        (
                            reference.reference_id,
                            reference.tenant_id,
                            reference.version,
                        ),
                        handle,
                    )
                })
                .collect(),
        }
    }

    /// Resolve only an exact reference, tenant, and version; there is no fallback.
    #[must_use]
    pub fn resolve(
        &self,
        reference: &CredentialRef,
        tenant_id: &str,
        version: u64,
    ) -> Option<Digest> {
        if reference.tenant_id != tenant_id || reference.version != version {
            return None;
        }
        self.handles
            .get(&(
                reference.reference_id.clone(),
                tenant_id.to_owned(),
                version,
            ))
            .copied()
    }
}
