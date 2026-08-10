use abi_worker::{
    CoordinatorTrustStore, JobEnvelope, JobRequest, MemoryControlStore, WorkerCapability,
    WorkerIdentity, WorkerLimits, WorkerManager,
};
use ed25519_dalek::SigningKey;
use serde_json::json;
use uuid::Uuid;

pub const NOW: i64 = 1_000_000;
pub const WORKER: &str = "worker-1";
pub const NAMESPACE: &str = "safe-production";
pub const COORDINATOR: &str = "abbey-coordinator";
pub const KEY_ID: &str = "coordinator-key-1";
pub const PRINCIPAL: &str = "service-account-1";

pub fn signing_key() -> SigningKey {
    SigningKey::from_bytes(&[17; 32])
}

pub fn trust() -> CoordinatorTrustStore {
    let key = signing_key();
    let mut trust = CoordinatorTrustStore::new();
    trust
        .insert(KEY_ID, COORDINATOR, key.verifying_key().as_bytes())
        .expect("fixture trust key");
    trust
}

pub fn identity() -> WorkerIdentity {
    WorkerIdentity::new(WORKER, NAMESPACE, [COORDINATOR.to_owned()]).expect("fixture identity")
}

pub fn capability() -> WorkerCapability {
    WorkerCapability::new("model.infer", "revision-1", 2).expect("fixture capability")
}

pub fn manager() -> WorkerManager<MemoryControlStore> {
    manager_with(WorkerLimits::default(), MemoryControlStore::new())
}

pub fn manager_with(
    limits: WorkerLimits,
    store: MemoryControlStore,
) -> WorkerManager<MemoryControlStore> {
    WorkerManager::new(identity(), [capability()], trust(), limits, store).expect("fixture manager")
}

pub fn request() -> JobRequest {
    JobRequest::new(
        "model.infer",
        "revision-1",
        "infer",
        json!({"prompt": "hello", "tokens": 4}),
        256,
        4,
    )
    .expect("fixture request")
}

pub fn envelope() -> JobEnvelope {
    envelope_for(Uuid::new_v4(), Uuid::new_v4(), request())
}

pub fn envelope_for(job_id: Uuid, nonce: Uuid, request: JobRequest) -> JobEnvelope {
    JobEnvelope::sign(
        job_id,
        PRINCIPAL,
        WORKER,
        NAMESPACE,
        COORDINATOR,
        NOW,
        NOW + 10_000,
        nonce,
        request,
        KEY_ID,
        &signing_key(),
    )
    .expect("fixture signed envelope")
}
