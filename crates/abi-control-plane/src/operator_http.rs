//! Content-free health and operator metrics. Metrics need a configured operator token digest.
use crate::{
    Error, OperatorMetrics, Result,
    service::{App, now},
};
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};

/// Operator metrics window: one day of audit events.
const WINDOW_SECS: i64 = 86_400;

/// Operator-only state: the token digest (absent disables metrics) and a process-local
/// count of `401` responses.
#[derive(Default)]
pub(crate) struct Operator {
    pub(crate) digest: Option<[u8; 32]>,
    pub(crate) denied: AtomicU64,
}

#[derive(Serialize)]
struct Snapshot {
    metrics: OperatorMetrics,
    auth_denied_total: u64,
}

pub(crate) fn routes() -> Router<App> {
    Router::new()
        .route("/healthz", get(health))
        .route("/v1/operator/metrics", get(metrics))
}

/// Count every unauthorized response without recording which caller or route it was.
pub(crate) async fn count_denied(
    State(app): State<App>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let response = next.run(request).await;
    if response.status() == StatusCode::UNAUTHORIZED {
        app.operator.denied.fetch_add(1, Ordering::Relaxed);
    }
    response
}

async fn health(State(app): State<App>) -> Response {
    let authority = app.authority.clone();
    let ok = tokio::task::spawn_blocking(move || {
        authority
            .lock()
            .ok()
            .and_then(|a| a.db.query_row("SELECT 1", [], |r| r.get::<_, i64>(0)).ok())
            == Some(1)
    })
    .await
    .unwrap_or(false);
    if ok {
        Json(serde_json::json!({"status":"ok"})).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status":"unavailable"})),
        )
            .into_response()
    }
}

async fn metrics(State(app): State<App>, headers: HeaderMap) -> Result<Response> {
    let Some(expected) = app.operator.digest else {
        return Ok(StatusCode::NOT_FOUND.into_response());
    };
    let token = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .filter(|t| !t.is_empty() && t.len() <= 256)
        .ok_or(Error::Denied)?;
    let actual: [u8; 32] = Sha256::digest(token.as_bytes()).into();
    if !constant_time_eq(&actual, &expected) {
        return Err(Error::Denied);
    }
    let authority = app.authority.clone();
    let metrics = tokio::task::spawn_blocking(move || {
        authority
            .lock()
            .map_err(|_| Error::Unavailable)?
            .operator_metrics(now()?, WINDOW_SECS)
    })
    .await
    .map_err(|_| Error::Unavailable)??;
    Ok(Json(Snapshot {
        metrics,
        auth_denied_total: app.operator.denied.load(Ordering::Relaxed),
    })
    .into_response())
}

fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a.iter().zip(b).fold(0_u8, |acc, (x, y)| acc | (x ^ y)) == 0
}
