//! Bounded `spawn_blocking` adapter around the synchronous WDBX facade.

use crate::GatewayError;
use crate::membership::MembershipStore;
use abi_wdbx::v3::episode::{EpisodeSigner, EpisodeStore, SignerKeyId, StorePolicy};
use abi_wdbx::{StorePaths, VersionedError, VersionedStore};
use ed25519_dalek::VerifyingKey;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

pub(crate) struct GatewayState {
    pub(crate) store: VersionedStore,
    pub(crate) membership: MembershipStore,
    /// Canonical episode ledger; `None` until an episode policy is configured.
    pub(crate) episodes: Option<EpisodeStore>,
    /// The key the episode ledger signs with, used to check signatures on
    /// verification; `None` when appends are unsigned.
    pub(crate) episode_verifier: Option<(SignerKeyId, VerifyingKey)>,
}

/// Subdirectory of the store path that holds the v3 episode ledger.
const EPISODE_DIRECTORY: &str = "episodes";

/// Cloneable handle for bounded synchronous store work.
#[derive(Clone)]
pub struct StoreExecutor {
    state: Arc<Mutex<GatewayState>>,
    permits: Arc<Semaphore>,
}

impl std::fmt::Debug for StoreExecutor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StoreExecutor")
            .finish_non_exhaustive()
    }
}

impl StoreExecutor {
    /// Open a scratch or configured store and cap simultaneous blocking tasks.
    /// No episode store is opened; see [`Self::open_with_episodes`].
    pub fn open(path: &Path, maximum_jobs: usize) -> Result<Self, GatewayError> {
        Self::open_with_episodes(path, maximum_jobs, None)
    }

    /// Open the store and, when a policy is supplied, the canonical episode
    /// ledger under `<path>/episodes` bound to that exact policy. Appended
    /// episodes are unsigned; see [`Self::open_with_signed_episodes`].
    pub fn open_with_episodes(
        path: &Path,
        maximum_jobs: usize,
        episode_policy: Option<StorePolicy>,
    ) -> Result<Self, GatewayError> {
        Self::open_with_signed_episodes(path, maximum_jobs, episode_policy, None)
    }

    /// Like [`Self::open_with_episodes`], but when `episode_signer` is
    /// supplied the episode ledger signs every record it appends.
    ///
    /// The signer must hold a key that signs nothing else. The gateway's
    /// membership ledger already signs with its own key under the store root,
    /// so a signer whose key equals that one is refused: one key signing both
    /// domains would let an episode signature be presented as a membership
    /// signature. A signer without a policy is refused too, because no episode
    /// store would exist for it to sign.
    ///
    /// Once a ledger holds signed records, a gateway build that predates
    /// episode signing can no longer open it.
    pub fn open_with_signed_episodes(
        path: &Path,
        maximum_jobs: usize,
        episode_policy: Option<StorePolicy>,
        episode_signer: Option<EpisodeSigner>,
    ) -> Result<Self, GatewayError> {
        if episode_signer.is_some() && episode_policy.is_none() {
            return Err(GatewayError::Configuration(
                "an episode signing key requires an episode policy".into(),
            ));
        }
        let store = VersionedStore::open(StorePaths::new(path))
            .map_err(|error| GatewayError::Store(error.to_string()))?;
        let membership =
            MembershipStore::open(path).map_err(|error| GatewayError::Store(error.to_string()))?;
        if let Some(signer) = &episode_signer {
            reject_membership_key(path, signer)?;
        }
        let episode_verifier = episode_signer
            .as_ref()
            .map(|signer| (signer.key_id().clone(), signer.verifying_key()));
        let episodes = episode_policy
            .map(|policy| {
                let directory = path.join(EPISODE_DIRECTORY);
                match episode_signer {
                    Some(signer) => EpisodeStore::open_with_signer(directory, policy, signer),
                    None => EpisodeStore::open(directory, policy),
                }
                .map_err(|error| GatewayError::Store(format!("episode store: {error}")))
            })
            .transpose()?;
        Ok(Self {
            state: Arc::new(Mutex::new(GatewayState {
                store,
                membership,
                episodes,
                episode_verifier,
            })),
            permits: Arc::new(Semaphore::new(maximum_jobs)),
        })
    }

    /// Execute one admitted versioned-store job without blocking a Tokio worker thread.
    pub(crate) async fn run<R, F>(&self, job: F) -> Result<R, GatewayError>
    where
        R: Send + 'static,
        F: FnOnce(&mut GatewayState) -> Result<R, VersionedError> + Send + 'static,
    {
        self.run_gateway(move |state| {
            job(state).map_err(|error| GatewayError::Store(error.to_string()))
        })
        .await
    }

    /// Execute one admitted job whose failure is already a gateway error.
    pub(crate) async fn run_gateway<R, F>(&self, job: F) -> Result<R, GatewayError>
    where
        R: Send + 'static,
        F: FnOnce(&mut GatewayState) -> Result<R, GatewayError> + Send + 'static,
    {
        let permit = Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .map_err(|_| GatewayError::Store("blocking executor closed".into()))?;
        let state = Arc::clone(&self.state);
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let mut guard = state
                .lock()
                .map_err(|_| GatewayError::Store("store actor lock poisoned".into()))?;
            job(&mut guard)
        })
        .await
        .map_err(|error| GatewayError::Store(format!("blocking store job failed: {error}")))?
    }
}

/// Refuse an episode signer that holds the membership ledger's signing key.
///
/// Compared by key, not by path, so a copied key file is caught too.
fn reject_membership_key(store_root: &Path, signer: &EpisodeSigner) -> Result<(), GatewayError> {
    let membership_key = crate::membership::signing_key_path(store_root);
    let membership = EpisodeSigner::from_key_file(&membership_key)
        .map_err(|error| GatewayError::Store(format!("membership signing key: {error}")))?;
    if membership.verifying_key() == signer.verifying_key() {
        return Err(GatewayError::Configuration(
            "the episode signing key must differ from the membership signing key".into(),
        ));
    }
    Ok(())
}
