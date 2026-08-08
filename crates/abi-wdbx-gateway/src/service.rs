//! Tonic WDBX service with admission checks ahead of store dispatch.

use crate::proto::wdbx_gateway_server::WdbxGateway;
use crate::proto::{
    GetKvRequest, GetKvResponse, MembershipAction, MembershipChangeRequest,
    MembershipChangeResponse, MutationEvent, PutKvRequest, PutKvResponse, PutVectorRequest,
    PutVectorResponse, ResolveConflictRequest, ResolveConflictResponse, SearchHit, SearchRequest,
    SearchResponse, StatsRequest, StatsResponse, WatchMutationsRequest,
};
use crate::{BearerToken, EventHub, GatewayError, Limits, StoreExecutor};
use abi_wdbx::{RecordId, V2Mutation};
use futures_util::Stream;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tonic::{Request, Response, Status};
use uuid::Uuid;

#[derive(Debug)]
struct RateWindow {
    start: Instant,
    used: u32,
}

/// Authenticated gRPC implementation.
#[derive(Clone)]
pub struct GatewayService {
    token: Arc<BearerToken>,
    limits: Limits,
    executor: StoreExecutor,
    events: Arc<EventHub>,
    rate: Arc<Mutex<RateWindow>>,
}

impl std::fmt::Debug for GatewayService {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GatewayService")
            .field("limits", &self.limits)
            .field("executor", &self.executor)
            .finish_non_exhaustive()
    }
}

impl GatewayService {
    /// Construct a service from validated shared components.
    #[must_use]
    pub fn new(
        token: Arc<BearerToken>,
        limits: Limits,
        executor: StoreExecutor,
        events: Arc<EventHub>,
    ) -> Self {
        Self {
            token,
            limits,
            executor,
            events,
            rate: Arc::new(Mutex::new(RateWindow {
                start: Instant::now(),
                used: 0,
            })),
        }
    }

    fn admit<T: prost::Message>(&self, request: &Request<T>) -> Result<(), Status> {
        let mut rate = self
            .rate
            .lock()
            .map_err(|_| Status::internal("rate limiter unavailable"))?;
        if rate.start.elapsed() >= Duration::from_secs(1) {
            rate.start = Instant::now();
            rate.used = 0;
        }
        if rate.used >= self.limits.requests_per_second {
            return Err(Status::resource_exhausted("request rate exceeded"));
        }
        rate.used += 1;
        drop(rate);
        let mut authorization_values = request.metadata().get_all("authorization").iter();
        let authorization = authorization_values
            .next()
            .and_then(|value| value.to_str().ok());
        if authorization_values.next().is_some() {
            return Err(Status::unauthenticated("invalid bearer credentials"));
        }
        if !self.token.authorizes(authorization) {
            return Err(Status::unauthenticated("invalid bearer credentials"));
        }
        if request.get_ref().encoded_len() > self.limits.message_bytes {
            return Err(Status::resource_exhausted("request exceeds message limit"));
        }
        Ok(())
    }

    fn validate_key(&self, key: &str) -> Result<(), Status> {
        if key.is_empty() || key.len() > self.limits.key_bytes {
            return Err(Status::invalid_argument("key length is outside limits"));
        }
        Ok(())
    }
}

impl From<GatewayError> for Status {
    fn from(error: GatewayError) -> Self {
        Self::internal(error.to_string())
    }
}

type WatchStream = Pin<Box<dyn Stream<Item = Result<MutationEvent, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl WdbxGateway for GatewayService {
    type WatchMutationsStream = WatchStream;

    async fn put_vector(
        &self,
        request: Request<PutVectorRequest>,
    ) -> Result<Response<PutVectorResponse>, Status> {
        self.admit(&request)?;
        let vectors = request.into_inner().vectors;
        if vectors.is_empty() || vectors.len() > self.limits.batch_items {
            return Err(Status::invalid_argument("vector batch is outside limits"));
        }
        if vectors.iter().any(|vector| {
            vector.values.is_empty()
                || vector.values.len() > abi_wdbx::v2::MAX_V2_VECTOR_DIMENSIONS
                || vector.values.iter().any(|value| !value.is_finite())
        }) {
            return Err(Status::invalid_argument(
                "vector dimensions or values are invalid",
            ));
        }
        let ids = (0..vectors.len())
            .map(|_| RecordId::new_v2())
            .collect::<Vec<_>>();
        let mutations = ids
            .iter()
            .copied()
            .zip(vectors)
            .map(|(id, vector)| V2Mutation::PutVector {
                id,
                values: vector.values,
            })
            .collect();
        let count = ids.len();
        let transaction = self
            .executor
            .run(move |state| state.store.commit_export(mutations))
            .await
            .map_err(Status::from)?;
        self.events
            .publish("put_vector", transaction.transaction_id(), count);
        Ok(Response::new(PutVectorResponse {
            ids: ids.into_iter().map(|id| id.to_string()).collect(),
        }))
    }

    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        self.admit(&request)?;
        let input = request.into_inner();
        if input.query.is_empty()
            || input.query.len() > abi_wdbx::v2::MAX_V2_VECTOR_DIMENSIONS
            || input.query.iter().any(|value| !value.is_finite())
        {
            return Err(Status::invalid_argument("query vector is invalid"));
        }
        let limit = usize::try_from(input.limit).unwrap_or(usize::MAX);
        if limit == 0 || limit > self.limits.search_limit {
            return Err(Status::invalid_argument("search limit is outside limits"));
        }
        let hits: Vec<SearchHit> = self
            .executor
            .run(move |state| {
                state.store.search(&input.query, limit).map(|results| {
                    results
                        .into_iter()
                        .map(|result| SearchHit {
                            id: result.id.to_string(),
                            score: result.score,
                        })
                        .collect()
                })
            })
            .await
            .map_err(Status::from)?;
        self.events.publish("search", "", hits.len());
        Ok(Response::new(SearchResponse { hits }))
    }

    async fn put_kv(
        &self,
        request: Request<PutKvRequest>,
    ) -> Result<Response<PutKvResponse>, Status> {
        self.admit(&request)?;
        let entries = request.into_inner().entries;
        if entries.is_empty() || entries.len() > self.limits.batch_items {
            return Err(Status::invalid_argument("KV batch is outside limits"));
        }
        for entry in &entries {
            self.validate_key(&entry.key)?;
            if entry.value.len() > self.limits.value_bytes {
                return Err(Status::invalid_argument("value exceeds limit"));
            }
        }
        let count = entries.len();
        let mutations = entries
            .into_iter()
            .map(|entry| V2Mutation::PutKv {
                key: entry.key,
                value: entry.value,
            })
            .collect();
        let transaction = self
            .executor
            .run(move |state| state.store.commit_export(mutations))
            .await
            .map_err(Status::from)?;
        self.events
            .publish("put_kv", transaction.transaction_id(), count);
        Ok(Response::new(PutKvResponse {
            transaction_ids: vec![transaction.transaction_id().to_string()],
        }))
    }

    async fn get_kv(
        &self,
        request: Request<GetKvRequest>,
    ) -> Result<Response<GetKvResponse>, Status> {
        self.admit(&request)?;
        self.validate_key(&request.get_ref().key)?;
        let key = request.into_inner().key;
        let found = self
            .executor
            .run(move |state| Ok(state.store.get_conflicts(&key)))
            .await
            .map_err(Status::from)?;
        let response = found.map_or(
            GetKvResponse {
                found: false,
                value: String::new(),
                version_ids: Vec::new(),
            },
            |set| {
                let mut version_ids = set
                    .conflicts
                    .into_iter()
                    .map(|version| version.version_id.to_string())
                    .collect::<Vec<_>>();
                version_ids.push(set.preferred.version_id.to_string());
                version_ids.sort();
                GetKvResponse {
                    found: true,
                    value: set.preferred.value,
                    version_ids,
                }
            },
        );
        Ok(Response::new(response))
    }

    async fn resolve_conflict(
        &self,
        request: Request<ResolveConflictRequest>,
    ) -> Result<Response<ResolveConflictResponse>, Status> {
        self.admit(&request)?;
        let input = request.into_inner();
        self.validate_key(&input.key)?;
        if input.value.len() > self.limits.value_bytes
            || input.version_ids.is_empty()
            || input.version_ids.len() > self.limits.batch_items
        {
            return Err(Status::invalid_argument("resolution is outside limits"));
        }
        let versions = input
            .version_ids
            .iter()
            .map(|value| {
                Uuid::parse_str(value).map_err(|_| Status::invalid_argument("invalid version id"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let transaction = self
            .executor
            .run(move |state| state.store.resolve(&input.key, &versions, input.value))
            .await
            .map_err(Status::from)?;
        self.events.publish("resolve_conflict", transaction, 1);
        Ok(Response::new(ResolveConflictResponse {
            transaction_id: transaction.to_string(),
        }))
    }

    async fn stats(
        &self,
        request: Request<StatsRequest>,
    ) -> Result<Response<StatsResponse>, Status> {
        self.admit(&request)?;
        let response = self
            .executor
            .run(|state| {
                let stats = state.store.stats();
                Ok(StatsResponse {
                    kv_entries: stats.kv_entries as u64,
                    vectors: stats.vectors as u64,
                    blocks: stats.blocks as u64,
                    spatial_records: stats.spatial_records as u64,
                    temporal_nodes: stats.temporal_nodes as u64,
                    temporal_edges: stats.temporal_edges as u64,
                    members: state.membership.active_len() as u64,
                })
            })
            .await
            .map_err(Status::from)?;
        Ok(Response::new(response))
    }

    async fn membership_change(
        &self,
        request: Request<MembershipChangeRequest>,
    ) -> Result<Response<MembershipChangeResponse>, Status> {
        self.admit(&request)?;
        let input = request.into_inner();
        self.validate_key(&input.node_id)?;
        let parsed_node = Uuid::parse_str(&input.node_id)
            .map_err(|_| Status::invalid_argument("node id must be a UUID"))?;
        let action = MembershipAction::try_from(input.action)
            .map_err(|_| Status::invalid_argument("unknown membership action"))?;
        let maximum_members = self.limits.membership_entries;
        let outcome = self
            .executor
            .run(move |state| {
                let result = match action {
                    MembershipAction::Add => {
                        state
                            .membership
                            .add(parsed_node, input.endpoint, maximum_members)
                    }
                    MembershipAction::Remove => state.membership.remove(parsed_node),
                    MembershipAction::Unspecified => {
                        return Err(abi_wdbx::VersionedError::V2(
                            abi_wdbx::V2Error::InvalidMutation(
                                "membership action is required".into(),
                            ),
                        ));
                    }
                };
                result.map_err(|error| {
                    abi_wdbx::VersionedError::V2(abi_wdbx::V2Error::InvalidMutation(
                        error.to_string(),
                    ))
                })
            })
            .await
            .map_err(Status::from)?;
        if outcome.changed {
            self.events.publish("membership_change", "", 1);
        }
        Ok(Response::new(MembershipChangeResponse {
            changed: outcome.changed,
            members: outcome.active_members as u64,
            generation: outcome.generation,
            head_digest: outcome.head_digest,
            tombstoned: outcome.tombstoned,
        }))
    }

    async fn watch_mutations(
        &self,
        request: Request<WatchMutationsRequest>,
    ) -> Result<Response<Self::WatchMutationsStream>, Status> {
        self.admit(&request)?;
        let permit = self
            .events
            .try_stream_permit()
            .ok_or_else(|| Status::resource_exhausted("stream limit reached"))?;
        let receiver = self.events.subscribe();
        let idle = self.events.idle();
        let stream = futures_util::stream::unfold(
            (receiver, permit, false),
            move |(mut receiver, permit, done)| async move {
                if done {
                    return None;
                }
                let item = match tokio::time::timeout(idle, receiver.recv()).await {
                    Ok(Ok(notice)) => Ok(notice.into()),
                    Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => {
                        Err(Status::resource_exhausted("slow mutation consumer"))
                    }
                    Ok(Err(tokio::sync::broadcast::error::RecvError::Closed)) => return None,
                    Err(_) => Err(Status::deadline_exceeded("mutation stream idle timeout")),
                };
                let terminal = item.is_err();
                Some((item, (receiver, permit, terminal)))
            },
        );
        Ok(Response::new(Box::pin(stream)))
    }
}
