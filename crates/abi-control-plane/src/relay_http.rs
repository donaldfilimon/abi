//! Authenticated outbound node polling and session-scoped encrypted job routing.
use crate::{
    Error, JobReceipt, JobView, NodeMessage, NodeReply, Result, Submission,
    service::{App, credentials, now, store},
};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, Query, State},
    http::HeaderMap,
    routing::{get, post},
};
pub(crate) fn routes() -> Router<App> {
    Router::new()
        .route("/v1/jobs", post(submit))
        .route("/v2/jobs", post(durable_submit))
        .route("/v2/jobs/{id}/delivery", get(delivery))
        .route("/v2/jobs/{id}/delivery/ack", post(delivery_ack))
        .route("/v1/jobs/{id}", get(view))
        .route("/v1/jobs/{id}/cancel", post(cancel))
        .route("/v1/relay/node", post(node))
        .layer(DefaultBodyLimit::max(384 * 1024))
}
async fn submit(
    State(app): State<App>,
    headers: HeaderMap,
    Json(submission): Json<Submission>,
) -> Result<Json<JobReceipt>> {
    let (token, csrf) = credentials(&app, &headers, true)?;
    let relay = app.relay.clone();
    store(&app, move |s| {
        let time = now()?;
        let account = s.session(&token, csrf.as_deref(), time)?;
        relay
            .lock()
            .map_err(|_| Error::Denied)?
            .submit(s, &account, submission, time)
            .map(Json)
    })
    .await
}
async fn view(
    State(app): State<App>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<JobView>> {
    let (token, _) = credentials(&app, &headers, false)?;
    let relay = app.relay.clone();
    store(&app, move |s| {
        let time = now()?;
        let account = s.session(&token, None, time)?;
        relay
            .lock()
            .map_err(|_| Error::Denied)?
            .view(s, &account, &id, time)
            .map(Json)
    })
    .await
}
async fn cancel(
    State(app): State<App>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<JobReceipt>> {
    let (token, csrf) = credentials(&app, &headers, true)?;
    let relay = app.relay.clone();
    store(&app, move |s| {
        let time = now()?;
        let account = s.session(&token, csrf.as_deref(), time)?;
        relay
            .lock()
            .map_err(|_| Error::Denied)?
            .cancel(s, &account, &id, time)
            .map(Json)
    })
    .await
}
async fn node(State(app): State<App>, Json(message): Json<NodeMessage>) -> Result<Json<NodeReply>> {
    let relay = app.relay.clone();
    store(&app, move |s| {
        relay
            .lock()
            .map_err(|_| Error::Denied)?
            .node(s, &message, now()?)
            .map(Json)
    })
    .await
}

#[derive(serde::Deserialize)]
struct Cursor {
    #[serde(default)]
    after: u32,
}
async fn durable_submit(
    State(app): State<App>,
    headers: HeaderMap,
    Json(submission): Json<crate::DurableSubmission>,
) -> Result<Json<JobReceipt>> {
    let (token, csrf) = credentials(&app, &headers, true)?;
    let relay = app.relay.clone();
    store(&app, move |s| {
        let time = now()?;
        let account = s.session(&token, csrf.as_deref(), time)?;
        relay
            .lock()
            .map_err(|_| Error::Denied)?
            .submit_durable(s, &account, submission, time)
            .map(Json)
    })
    .await
}
async fn delivery(
    State(app): State<App>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Query(cursor): Query<Cursor>,
) -> Result<Json<crate::DeliveryPage>> {
    let (token, _) = credentials(&app, &headers, false)?;
    let relay = app.relay.clone();
    store(&app, move |s| {
        let time = now()?;
        let account = s.session(&token, None, time)?;
        relay
            .lock()
            .map_err(|_| Error::Denied)?
            .delivery(s, &account, &id, cursor.after, time)
            .map(Json)
    })
    .await
}
async fn delivery_ack(
    State(app): State<App>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(ack): Json<crate::DeliveryAck>,
) -> Result<Json<bool>> {
    let (token, csrf) = credentials(&app, &headers, true)?;
    let relay = app.relay.clone();
    store(&app, move |s| {
        let time = now()?;
        let account = s.session(&token, csrf.as_deref(), time)?;
        relay
            .lock()
            .map_err(|_| Error::Denied)?
            .acknowledge_delivery(s, &account, &id, &ack, time)?;
        Ok(Json(true))
    })
    .await
}
