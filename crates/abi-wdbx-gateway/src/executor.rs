//! Bounded `spawn_blocking` adapter around the synchronous WDBX facade.

use crate::GatewayError;
use crate::membership::MembershipStore;
use abi_wdbx::v3::episode::{EpisodeStore, StorePolicy};
use abi_wdbx::{StorePaths, VersionedError, VersionedStore};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

pub(crate) struct GatewayState {
    pub(crate) store: VersionedStore,
    pub(crate) membership: MembershipStore,
    /// Canonical episode ledger; `None` until an episode policy is configured.
    pub(crate) episodes: Option<EpisodeStore>,
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
    /// ledger under `<path>/episodes` bound to that exact policy.
    pub fn open_with_episodes(
        path: &Path,
        maximum_jobs: usize,
        episode_policy: Option<StorePolicy>,
    ) -> Result<Self, GatewayError> {
        let store = VersionedStore::open(StorePaths::new(path))
            .map_err(|error| GatewayError::Store(error.to_string()))?;
        let membership =
            MembershipStore::open(path).map_err(|error| GatewayError::Store(error.to_string()))?;
        let episodes = episode_policy
            .map(|policy| {
                EpisodeStore::open(path.join(EPISODE_DIRECTORY), policy)
                    .map_err(|error| GatewayError::Store(format!("episode store: {error}")))
            })
            .transpose()?;
        Ok(Self {
            state: Arc::new(Mutex::new(GatewayState {
                store,
                membership,
                episodes,
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
