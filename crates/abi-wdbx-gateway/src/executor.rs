//! Bounded `spawn_blocking` adapter around the synchronous WDBX facade.

use crate::GatewayError;
use crate::membership::MembershipStore;
use abi_wdbx::{StorePaths, VersionedError, VersionedStore};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

pub(crate) struct GatewayState {
    pub(crate) store: VersionedStore,
    pub(crate) membership: MembershipStore,
}

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
    pub fn open(path: &Path, maximum_jobs: usize) -> Result<Self, GatewayError> {
        let store = VersionedStore::open(StorePaths::new(path))
            .map_err(|error| GatewayError::Store(error.to_string()))?;
        let membership =
            MembershipStore::open(path).map_err(|error| GatewayError::Store(error.to_string()))?;
        Ok(Self {
            state: Arc::new(Mutex::new(GatewayState { store, membership })),
            permits: Arc::new(Semaphore::new(maximum_jobs)),
        })
    }

    /// Execute one admitted job without blocking a Tokio worker thread.
    pub(crate) async fn run<R, F>(&self, job: F) -> Result<R, GatewayError>
    where
        R: Send + 'static,
        F: FnOnce(&mut GatewayState) -> Result<R, VersionedError> + Send + 'static,
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
            job(&mut guard).map_err(|error| GatewayError::Store(error.to_string()))
        })
        .await
        .map_err(|error| GatewayError::Store(format!("blocking store job failed: {error}")))?
    }
}
