//! Loopback listener for a TLS reverse proxy; no forwarded authentication headers.
use crate::{Authority, Error, OidcConfig, Result, oidc::Oidc};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, Query, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Redirect, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::{
    path::Path as FilePath,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone)]
pub(crate) struct App {
    pub(crate) authority: Arc<Mutex<Authority>>,
    pub(crate) oidc: Arc<Oidc>,
    pub(crate) relay: Arc<Mutex<crate::Relay>>,
    pub(crate) operator: Arc<crate::operator_http::Operator>,
}
/// Build account/enrollment APIs, browser client and encrypted tenant relay.
/// The listener must be loopback behind an HTTPS reverse proxy outside local development.
pub async fn router(path: &FilePath, config: OidcConfig) -> Result<Router> {
    router_with_limits(path, config, crate::RelayLimits::default()).await
}
/// Build with validated finite per-account scheduling and global ciphertext budgets.
pub async fn router_with_limits(
    path: &FilePath,
    config: OidcConfig,
    limits: crate::RelayLimits,
) -> Result<Router> {
    router_with_operator(path, config, limits, None).await
}
/// Build with an optional operator token SHA-256 digest; without it the metrics route is 404.
pub async fn router_with_operator(
    path: &FilePath,
    config: OidcConfig,
    limits: crate::RelayLimits,
    operator_token_sha256: Option<[u8; 32]>,
) -> Result<Router> {
    let oidc = Oidc::discover(config).await?;
    let mut authority = Authority::open(path, false, &oidc.config.origin)?;
    let relay = crate::Relay::open(&mut authority, limits, now()?)?;
    Ok(routes(App {
        authority: Arc::new(Mutex::new(authority)),
        oidc: Arc::new(oidc),
        relay: Arc::new(Mutex::new(relay)),
        operator: Arc::new(crate::operator_http::Operator {
            digest: operator_token_sha256,
            ..Default::default()
        }),
    }))
}
fn routes(app: App) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/app.js", get(script))
        .route("/app.css", get(styles))
        .route("/auth/login", get(login))
        .route("/auth/callback", get(callback))
        .route("/v1/authority", get(authority_key))
        .route("/v1/account", get(account).delete(delete_account))
        .route("/v1/logout", post(logout))
        .route("/v1/enrollments", post(begin_pairing))
        .route("/v1/enrollments/claim", post(claim_pairing))
        .route("/v1/enrollments/{id}", get(poll_pairing))
        .route("/v1/enrollments/{id}/confirm", post(confirm_pairing))
        .route("/v1/nodes/{id}/revoke", post(revoke))
        .route("/v1/nodes/authenticate", post(authenticate))
        .merge(crate::relay_http::routes())
        .merge(crate::operator_http::routes())
        .layer(DefaultBodyLimit::max(16 * 1024))
        .layer(axum::middleware::from_fn_with_state(
            app.clone(),
            crate::operator_http::count_denied,
        ))
        .layer(axum::middleware::from_fn(security_headers))
        .with_state(app)
}
impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = match self {
            Self::Denied => StatusCode::UNAUTHORIZED,
            Self::Invalid => StatusCode::BAD_REQUEST,
            Self::Full => StatusCode::TOO_MANY_REQUESTS,
            Self::Unavailable | Self::Storage(_) | Self::Io(_) => StatusCode::SERVICE_UNAVAILABLE,
        };
        (
            status,
            Json(
                serde_json::json!({"error":status.canonical_reason().unwrap_or("request failed")}),
            ),
        )
            .into_response()
    }
}
pub(crate) fn now() -> Result<i64> {
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| Error::Denied)?
            .as_secs(),
    )
    .map_err(|_| Error::Denied)
}
pub(crate) async fn store<T: Send + 'static>(
    app: &App,
    f: impl FnOnce(&mut Authority) -> Result<T> + Send + 'static,
) -> Result<T> {
    let authority = app.authority.clone();
    let relay = app.relay.clone();
    tokio::task::spawn_blocking(move || {
        let mut guard = authority.lock().map_err(|_| Error::Denied)?;
        relay
            .lock()
            .map_err(|_| Error::Denied)?
            .cleanup(&mut guard, now()?)?;
        let result = f(&mut guard);
        relay
            .lock()
            .map_err(|_| Error::Denied)?
            .cleanup(&mut guard, now()?)?;
        result
    })
    .await
    .map_err(|_| Error::Denied)?
}
fn cookie_name(app: &App, kind: &str) -> String {
    format!(
        "{}abi_{kind}",
        if app.oidc.config.loopback_development {
            ""
        } else {
            "__Host-"
        }
    )
}
fn cookie(headers: &HeaderMap, name: &str) -> Result<String> {
    let mut found = None;
    for value in headers.get_all(header::COOKIE) {
        for item in value.to_str().map_err(|_| Error::Denied)?.split(';') {
            if let Some((key, value)) = item.trim().split_once('=')
                && key == name
            {
                if found.is_some() || value.len() > 128 {
                    return Err(Error::Denied);
                }
                found = Some(value.to_owned());
            }
        }
    }
    found.ok_or(Error::Denied)
}
fn set_cookie(
    app: &App,
    kind: &str,
    value: &str,
    age: u32,
    http_only: bool,
) -> Result<header::HeaderValue> {
    let value = format!(
        "{}={value}; Path=/; Max-Age={age}; SameSite=Lax{}{}",
        cookie_name(app, kind),
        if http_only { "; HttpOnly" } else { "" },
        if app.oidc.config.loopback_development {
            ""
        } else {
            "; Secure"
        }
    );
    value.parse().map_err(|_| Error::Invalid)
}
pub(crate) fn credentials(
    app: &App,
    headers: &HeaderMap,
    write: bool,
) -> Result<(String, Option<String>)> {
    let token = cookie(headers, &cookie_name(app, "session"))?;
    let csrf = if write {
        if headers.get(header::ORIGIN).and_then(|s| s.to_str().ok())
            != Some(app.oidc.config.origin.as_str())
        {
            return Err(Error::Denied);
        }
        Some(
            headers
                .get("x-csrf-token")
                .ok_or(Error::Denied)?
                .to_str()
                .map_err(|_| Error::Denied)?
                .to_owned(),
        )
    } else {
        None
    };
    Ok((token, csrf))
}
async fn login(State(app): State<App>) -> Result<Response> {
    let (url, cookie) = app.oidc.begin(now()?)?;
    let mut response = Redirect::to(&url).into_response();
    response.headers_mut().append(
        header::SET_COOKIE,
        set_cookie(&app, "login", &cookie, 300, true)?,
    );
    Ok(response)
}
#[derive(Deserialize)]
struct Callback {
    state: String,
    code: String,
}
async fn callback(
    State(app): State<App>,
    headers: HeaderMap,
    Query(query): Query<Callback>,
) -> Result<Response> {
    let identity = app
        .oidc
        .finish(
            &query.state,
            &query.code,
            &cookie(&headers, &cookie_name(&app, "login"))?,
            now()?,
        )
        .await?;
    let session = store(&app, move |s| {
        let now = now()?;
        let account = s.account(&identity, now)?;
        s.new_session(&account, now)
    })
    .await?;
    let mut response = Redirect::to("/").into_response();
    response.headers_mut().append(
        header::SET_COOKIE,
        set_cookie(&app, "session", &session.token, 28_800, true)?,
    );
    response.headers_mut().append(
        header::SET_COOKIE,
        set_cookie(&app, "csrf", &session.csrf, 28_800, false)?,
    );
    response
        .headers_mut()
        .append(header::SET_COOKIE, set_cookie(&app, "login", "", 0, true)?);
    Ok(response)
}
#[derive(Serialize)]
struct AccountView {
    account: crate::Account,
    nodes: Vec<crate::Node>,
}
async fn authority_key(State(app): State<App>) -> Result<Json<[u8; 32]>> {
    store(&app, |s| Ok(Json(s.public_key()))).await
}
async fn account(State(app): State<App>, headers: HeaderMap) -> Result<Json<AccountView>> {
    let (token, _) = credentials(&app, &headers, false)?;
    let limits = app.relay.lock().map_err(|_| Error::Denied)?.limits;
    store(&app, move |s| {
        let mut account = s.session(&token, None, now()?)?;
        account.active_job_limit = limits.active_jobs;
        account.queued_job_limit = limits.queued_jobs;
        let nodes = s.nodes(&account)?;
        Ok(Json(AccountView { account, nodes }))
    })
    .await
}
async fn delete_account(State(app): State<App>, headers: HeaderMap) -> Result<StatusCode> {
    let (token, csrf) = credentials(&app, &headers, true)?;
    store(&app, move |s| {
        let now = now()?;
        let account = s.session(&token, csrf.as_deref(), now)?;
        s.delete_account(&account, now)
    })
    .await?;
    Ok(StatusCode::NO_CONTENT)
}
async fn logout(State(app): State<App>, headers: HeaderMap) -> Result<Response> {
    let (token, csrf) = credentials(&app, &headers, true)?;
    store(&app, move |s| {
        let now = now()?;
        s.session(&token, csrf.as_deref(), now)?;
        s.logout(&token, now)
    })
    .await?;
    let mut response = StatusCode::NO_CONTENT.into_response();
    response.headers_mut().append(
        header::SET_COOKIE,
        set_cookie(&app, "session", "", 0, true)?,
    );
    response
        .headers_mut()
        .append(header::SET_COOKIE, set_cookie(&app, "csrf", "", 0, false)?);
    Ok(response)
}
async fn begin_pairing(
    State(app): State<App>,
    Json(request): Json<crate::PairRequest>,
) -> Result<Json<crate::PairTicket>> {
    store(&app, move |s| s.begin_pairing(&request, now()?).map(Json)).await
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClaimRequest {
    user_code: String,
}
async fn claim_pairing(
    State(app): State<App>,
    headers: HeaderMap,
    Json(request): Json<ClaimRequest>,
) -> Result<Json<crate::PairClaim>> {
    let (token, csrf) = credentials(&app, &headers, true)?;
    store(&app, move |s| {
        let now = now()?;
        let account = s.session(&token, csrf.as_deref(), now)?;
        s.claim_pairing(&account, &request.user_code, now).map(Json)
    })
    .await
}
fn bearer(headers: &HeaderMap) -> Result<String> {
    let value = headers
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .ok_or(Error::Denied)?;
    if value.len() > 128 {
        return Err(Error::Denied);
    }
    Ok(value.into())
}
async fn poll_pairing(
    State(app): State<App>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<Option<crate::PairClaim>>> {
    let secret = bearer(&headers)?;
    store(&app, move |s| {
        s.poll_pairing(&id, &secret, now()?).map(Json)
    })
    .await
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Signature {
    signature: Vec<u8>,
}
async fn confirm_pairing(
    State(app): State<App>,
    Path(id): Path<String>,
    headers: HeaderMap,
    Json(request): Json<Signature>,
) -> Result<Json<crate::NodeCertificate>> {
    let secret = bearer(&headers)?;
    store(&app, move |s| {
        s.confirm_pairing(&id, &secret, &request.signature, now()?)
            .map(Json)
    })
    .await
}
async fn revoke(
    State(app): State<App>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<StatusCode> {
    let (token, csrf) = credentials(&app, &headers, true)?;
    store(&app, move |s| {
        let now = now()?;
        let account = s.session(&token, csrf.as_deref(), now)?;
        s.revoke(&account, &id, now)
    })
    .await?;
    Ok(StatusCode::NO_CONTENT)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Proof {
    challenge: crate::NodeChallenge,
    signature: Vec<u8>,
}
async fn authenticate(
    State(app): State<App>,
    Json(request): Json<Proof>,
) -> Result<Json<crate::NodeCertificate>> {
    store(&app, move |s| {
        s.authenticate_node(&request.challenge, &request.signature, now()?)
            .map(Json)
    })
    .await
}
async fn index() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
        include_str!("../web/index.html"),
    )
}
async fn script() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        include_str!("../web/app.js"),
    )
}
async fn styles() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("../web/app.css"),
    )
}
async fn security_headers(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let mut response = next.run(request).await;
    for (name, value) in [
        ("cache-control", "no-store"),
        ("referrer-policy", "no-referrer"),
        ("x-content-type-options", "nosniff"),
        (
            "content-security-policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
        ),
    ] {
        response.headers_mut().insert(
            axum::http::HeaderName::from_static(name),
            header::HeaderValue::from_static(value),
        );
    }
    response
}
#[cfg(all(test, unix))]
mod tests;
