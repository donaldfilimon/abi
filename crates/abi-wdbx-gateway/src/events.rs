//! Bounded, metadata-only mutation notification hub and WebSocket route.

use crate::{BearerToken, Limits, proto};
use axum::Router;
use axum::extract::State;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade, close_code};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse as _, Response};
use axum::routing::get;
use futures_util::StreamExt as _;
use serde::Serialize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, broadcast};

/// Metadata-only mutation notification. It never contains keys, values, or vectors.
#[derive(Debug, Clone, Serialize)]
pub struct MutationNotice {
    /// Monotonic process-local event sequence.
    pub sequence: u64,
    /// Coarse operation name.
    pub kind: String,
    /// Store transaction identity, empty for gateway-only membership changes.
    pub transaction_id: String,
    /// Number of mutated items.
    pub item_count: u32,
    /// Publication time as Unix milliseconds.
    pub unix_ms: i64,
}

impl From<MutationNotice> for proto::MutationEvent {
    fn from(value: MutationNotice) -> Self {
        Self {
            sequence: value.sequence,
            kind: value.kind,
            transaction_id: value.transaction_id,
            item_count: value.item_count,
            unix_ms: value.unix_ms,
        }
    }
}

/// Bounded broadcast queue shared by gRPC and WebSocket subscribers.
#[derive(Debug)]
pub struct EventHub {
    sender: broadcast::Sender<MutationNotice>,
    next_sequence: AtomicU64,
    streams: Arc<Semaphore>,
    idle: Duration,
}

impl EventHub {
    /// Create an event hub from validated resource limits.
    #[must_use]
    pub fn new(limits: &Limits) -> Self {
        let (sender, _) = broadcast::channel(limits.event_queue);
        Self {
            sender,
            next_sequence: AtomicU64::new(1),
            streams: Arc::new(Semaphore::new(limits.concurrent_streams)),
            idle: Duration::from_secs(limits.idle_seconds),
        }
    }

    /// Publish secret-free mutation metadata. No-subscriber sends are benign.
    pub fn publish(&self, kind: &str, transaction_id: impl Into<String>, item_count: usize) {
        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| {
                i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
            });
        let notice = MutationNotice {
            sequence,
            kind: kind.to_owned(),
            transaction_id: transaction_id.into(),
            item_count: u32::try_from(item_count).unwrap_or(u32::MAX),
            unix_ms,
        };
        let _ = self.sender.send(notice);
    }

    /// Subscribe to the bounded queue.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<MutationNotice> {
        self.sender.subscribe()
    }

    /// Try to reserve one concurrent stream slot.
    pub fn try_stream_permit(&self) -> Option<OwnedSemaphorePermit> {
        Arc::clone(&self.streams).try_acquire_owned().ok()
    }

    /// Configured idle timeout.
    #[must_use]
    pub const fn idle(&self) -> Duration {
        self.idle
    }
}

#[derive(Clone)]
struct WsState {
    token: Arc<BearerToken>,
    origins: Arc<[String]>,
    hub: Arc<EventHub>,
    message_bytes: usize,
    requests_per_second: u32,
    rate: Arc<std::sync::Mutex<UpgradeWindow>>,
}

#[derive(Debug)]
struct UpgradeWindow {
    start: Instant,
    used: u32,
}

/// Build the authenticated `/v1/events` WebSocket application.
pub fn websocket_router(
    token: Arc<BearerToken>,
    origins: Vec<String>,
    hub: Arc<EventHub>,
    message_bytes: usize,
    requests_per_second: u32,
) -> Router {
    Router::new()
        .route("/v1/events", get(upgrade_events))
        .with_state(WsState {
            token,
            origins: origins.into(),
            hub,
            message_bytes,
            requests_per_second,
            rate: Arc::new(std::sync::Mutex::new(UpgradeWindow {
                start: Instant::now(),
                used: 0,
            })),
        })
}

async fn upgrade_events(
    State(state): State<WsState>,
    headers: HeaderMap,
    websocket: WebSocketUpgrade,
) -> Response {
    if !admit_upgrade(&state) {
        return (StatusCode::TOO_MANY_REQUESTS, "upgrade rate exceeded").into_response();
    }
    if let Err(status) = validate_websocket_headers(&state.token, &state.origins, &headers) {
        return (
            status,
            status.canonical_reason().unwrap_or("request rejected"),
        )
            .into_response();
    }
    let Some(permit) = state.hub.try_stream_permit() else {
        return (StatusCode::TOO_MANY_REQUESTS, "stream limit reached").into_response();
    };
    websocket
        .max_message_size(state.message_bytes)
        .on_upgrade(move |socket| serve_socket(socket, state, permit))
}

fn admit_upgrade(state: &WsState) -> bool {
    let Ok(mut rate) = state.rate.lock() else {
        return false;
    };
    if rate.start.elapsed() >= Duration::from_secs(1) {
        rate.start = Instant::now();
        rate.used = 0;
    }
    if rate.used >= state.requests_per_second {
        return false;
    }
    rate.used += 1;
    true
}

/// Validate bearer authentication and an exact browser-origin allowlist.
pub fn validate_websocket_headers(
    token: &BearerToken,
    origins: &[String],
    headers: &HeaderMap,
) -> Result<(), StatusCode> {
    let mut authorization_values = headers.get_all(axum::http::header::AUTHORIZATION).iter();
    let authorization = authorization_values
        .next()
        .and_then(|value| value.to_str().ok());
    if authorization_values.next().is_some() {
        return Err(StatusCode::UNAUTHORIZED);
    }
    if !token.authorizes(authorization) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    let mut origin_values = headers.get_all(axum::http::header::ORIGIN).iter();
    let origin = origin_values.next().and_then(|value| value.to_str().ok());
    if origin_values.next().is_some()
        || !origin.is_some_and(|value| {
            !value.eq_ignore_ascii_case("null") && origins.iter().any(|allowed| allowed == value)
        })
    {
        return Err(StatusCode::FORBIDDEN);
    }
    Ok(())
}

async fn serve_socket(mut socket: WebSocket, state: WsState, _permit: OwnedSemaphorePermit) {
    let mut receiver = state.hub.subscribe();
    loop {
        tokio::select! {
            incoming = socket.next() => match incoming {
                Some(Ok(Message::Close(_)) | Err(_)) | None => break,
                Some(Ok(Message::Text(text))) if text.len() > state.message_bytes => {
                    close_socket(&mut socket, close_code::SIZE, "message too large").await;
                    break;
                }
                Some(Ok(Message::Binary(bytes))) if bytes.len() > state.message_bytes => {
                    close_socket(&mut socket, close_code::SIZE, "message too large").await;
                    break;
                }
                Some(Ok(_)) => {}
            },
            event = tokio::time::timeout(state.hub.idle(), receiver.recv()) => match event {
                Ok(Ok(notice)) => {
                    let Ok(json) = serde_json::to_string(&notice) else { break; };
                    match tokio::time::timeout(
                        state.hub.idle(),
                        socket.send(Message::Text(json.into())),
                    ).await {
                        Ok(Ok(())) => {}
                        Ok(Err(_)) => break,
                        Err(_) => {
                            close_socket(&mut socket, close_code::AGAIN, "slow consumer").await;
                            break;
                        }
                    }
                }
                Ok(Err(broadcast::error::RecvError::Lagged(_))) => {
                    close_socket(&mut socket, close_code::AGAIN, "slow consumer").await;
                    break;
                }
                Ok(Err(broadcast::error::RecvError::Closed)) => break,
                Err(_) => {
                    close_socket(&mut socket, close_code::AWAY, "idle timeout").await;
                    break;
                }
            }
        }
    }
}

async fn close_socket(socket: &mut WebSocket, code: u16, reason: &'static str) {
    let _ = socket
        .send(Message::Close(Some(CloseFrame {
            code,
            reason: reason.into(),
        })))
        .await;
}
