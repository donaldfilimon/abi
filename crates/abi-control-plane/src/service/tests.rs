use super::*;
use axum::{body::Body, http::Request};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD as B64};
use ed25519_dalek::{Signer as _, SigningKey};
use http_body_util::BodyExt as _;
use sha2::{Digest as _, Sha256};
use tower::ServiceExt as _;

#[test]
fn cookie_parser_refuses_ambiguity() {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::COOKIE,
        "abi_session=first; abi_session=second".parse().unwrap(),
    );
    assert!(cookie(&headers, "abi_session").is_err());
}
#[derive(Clone)]
struct Issuer {
    redirect_origin: String,
    base: String,
    key: SigningKey,
}
struct Fixture {
    dir: tempfile::TempDir,
    app: Router,
    authority: Arc<Mutex<Authority>>,
    relay: Arc<Mutex<crate::Relay>>,
    issuer: tokio::task::JoinHandle<()>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.issuer.abort();
    }
}
async fn fixture() -> Fixture {
    fixture_with_origin("http://127.0.0.1:3000".into()).await
}
async fn fixture_with_origin(origin: String) -> Fixture {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let issuer = Issuer {
        redirect_origin: origin.clone(),
        base: format!("http://{}", listener.local_addr().unwrap()),
        key: SigningKey::from_bytes(&[42; 32]),
    };
    let issuer_app = Router::new()
        .route("/.well-known/openid-configuration", get(discovery))
        .route("/jwks", get(jwks))
        .route("/token", post(token))
        .with_state(issuer.clone());
    let task = tokio::spawn(async move {
        axum::serve(listener, issuer_app).await.unwrap();
    });
    let config = OidcConfig {
        issuer: issuer.base,
        client_id: "abi-client".into(),
        client_secret: None,
        origin,
        loopback_development: true,
    };
    let oidc = Oidc::discover(config).await.unwrap();
    let dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(
        dir.path(),
        std::os::unix::fs::PermissionsExt::from_mode(0o700),
    )
    .unwrap();
    let authority = Arc::new(Mutex::new(
        Authority::open(
            &dir.path().join("control.sqlite"),
            true,
            &oidc.config.origin,
        )
        .unwrap(),
    ));
    let relay = crate::Relay::open(
        &mut authority.lock().unwrap(),
        crate::RelayLimits::default(),
        now().unwrap(),
    )
    .unwrap();
    let relay = Arc::new(Mutex::new(relay));
    let app = routes(App {
        authority: authority.clone(),
        oidc: Arc::new(oidc),
        relay: relay.clone(),
    });
    Fixture {
        dir,
        app,
        authority,
        relay,
        issuer: task,
    }
}
async fn discovery(State(issuer): State<Issuer>) -> Json<serde_json::Value> {
    Json(
        serde_json::json!({"issuer":issuer.base,"authorization_endpoint":format!("{}/authorize",issuer.base),"token_endpoint":format!("{}/token",issuer.base),"jwks_uri":format!("{}/jwks",issuer.base),"response_types_supported":["code"],"subject_types_supported":["public"],"id_token_signing_alg_values_supported":["EdDSA"],"code_challenge_methods_supported":["S256"]}),
    )
}
async fn jwks(State(issuer): State<Issuer>) -> Json<serde_json::Value> {
    Json(
        serde_json::json!({"keys":[{"kty":"OKP","crv":"Ed25519","kid":"test-key","use":"sig","alg":"EdDSA","x":B64.encode(issuer.key.verifying_key().to_bytes())}]}),
    )
}
async fn token(
    State(issuer): State<Issuer>,
    axum::Form(form): axum::Form<std::collections::HashMap<String, String>>,
) -> Response {
    let code: serde_json::Value =
        serde_json::from_slice(&B64.decode(&form["code"]).unwrap()).unwrap();
    if B64.encode(Sha256::digest(form["code_verifier"].as_bytes()))
        != code["challenge"].as_str().unwrap()
    {
        return StatusCode::BAD_REQUEST.into_response();
    }
    assert_eq!(
        form["redirect_uri"],
        format!("{}/auth/callback", issuer.redirect_origin)
    );
    let bad = code["bad"].as_str().unwrap();
    let clock = now().unwrap();
    let mut claims = serde_json::json!({"iss":issuer.base,"sub":code["sub"],"aud":"abi-client","iat":clock,"exp":clock+300,"nonce":code["nonce"],"email":"shared@example.test","email_verified":true});
    match bad {
        "issuer" => claims["iss"] = "https://forged.example".into(),
        "audience" => claims["aud"] = "other-client".into(),
        "nonce" => claims["nonce"] = "forged-nonce".into(),
        "expired" => claims["exp"] = (clock - 100).into(),
        "unverified" => claims["email_verified"] = false.into(),
        "azp" => claims["azp"] = "other-client".into(),
        "multi-audience" => claims["aud"] = serde_json::json!(["abi-client", "other-client"]),
        "future-iat" => claims["iat"] = (clock + 120).into(),
        "nbf" => claims["nbf"] = (clock + 120).into(),
        "access-hash" => claims["at_hash"] = "forged".into(),
        "missing-email" => {
            claims.as_object_mut().unwrap().remove("email");
        }
        _ => {}
    }
    let header = if bad == "critical-header" {
        B64.encode(br#"{"alg":"EdDSA","kid":"test-key","crit":["custom"],"custom":true}"#)
    } else {
        B64.encode(br#"{"alg":"EdDSA","kid":"test-key"}"#)
    };
    let input = format!(
        "{header}.{}",
        B64.encode(serde_json::to_vec(&claims).unwrap())
    );
    let signing_key = if bad == "signature" {
        SigningKey::from_bytes(&[43; 32])
    } else {
        issuer.key
    };
    let jwt = format!(
        "{input}.{}",
        B64.encode(signing_key.sign(input.as_bytes()).to_bytes())
    );
    Json(serde_json::json!({"access_token":"discarded-access-token","token_type":"Bearer","id_token":jwt,"expires_in":300})).into_response()
}
async fn request(
    app: &Router,
    method: &str,
    path: &str,
    cookies: &str,
    origin: Option<&str>,
    csrf: Option<&str>,
    body: Option<serde_json::Value>,
) -> Response {
    let mut request = Request::builder()
        .method(method)
        .uri(path)
        .header(header::COOKIE, cookies);
    if let Some(origin) = origin {
        request = request.header(header::ORIGIN, origin);
    }
    if let Some(csrf) = csrf {
        request = request.header("x-csrf-token", csrf);
    }
    if body.is_some() {
        request = request.header(header::CONTENT_TYPE, "application/json");
    }
    app.clone()
        .oneshot(
            request
                .body(Body::from(body.map_or_else(String::new, |v| v.to_string())))
                .unwrap(),
        )
        .await
        .unwrap()
}
async fn json(response: Response) -> serde_json::Value {
    serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
}
struct Browser {
    cookie: String,
    csrf: String,
}
async fn login_flow(f: &Fixture, subject: &str, bad: &str) -> (String, String) {
    let response = request(&f.app, "GET", "/auth/login", "", None, None, None).await;
    assert_eq!(response.status(), StatusCode::SEE_OTHER);
    let cookie = response.headers()[header::SET_COOKIE]
        .to_str()
        .unwrap()
        .split(';')
        .next()
        .unwrap();
    let url =
        oauth2::url::Url::parse(response.headers()[header::LOCATION].to_str().unwrap()).unwrap();
    let query: std::collections::HashMap<_, _> = url.query_pairs().into_owned().collect();
    assert_eq!(query["code_challenge_method"], "S256");
    let code=B64.encode(serde_json::to_vec(&serde_json::json!({"sub":subject,"nonce":query["nonce"],"challenge":query["code_challenge"],"bad":bad})).unwrap());
    let mut callback = oauth2::url::Url::parse("http://127.0.0.1:3000/auth/callback").unwrap();
    callback
        .query_pairs_mut()
        .append_pair("state", &query["state"])
        .append_pair("code", &code);
    (
        format!("{}?{}", callback.path(), callback.query().unwrap()),
        cookie.to_owned(),
    )
}
async fn login(f: &Fixture, subject: &str, bad: &str) -> Response {
    let (path, cookie) = login_flow(f, subject, bad).await;
    request(&f.app, "GET", &path, &cookie, None, None, None).await
}
async fn browser(f: &Fixture, subject: &str) -> Browser {
    let response = login(f, subject, "").await;
    assert_eq!(response.status(), StatusCode::SEE_OTHER);
    let cookies = response
        .headers()
        .get_all(header::SET_COOKIE)
        .iter()
        .map(|s| s.to_str().unwrap().split(';').next().unwrap().to_owned())
        .collect::<Vec<_>>();
    let csrf = cookies
        .iter()
        .find_map(|s| s.strip_prefix("abi_csrf="))
        .unwrap()
        .to_owned();
    Browser {
        cookie: cookies.join("; "),
        csrf,
    }
}
#[tokio::test]
async fn real_oidc_code_exchange_refuses_forged_unverified_and_stale_tokens() {
    let f = fixture().await;
    for bad in [
        "signature",
        "issuer",
        "audience",
        "nonce",
        "expired",
        "unverified",
        "missing-email",
        "azp",
        "multi-audience",
        "future-iat",
        "nbf",
        "access-hash",
        "critical-header",
    ] {
        let response = login(&f, "alice", bad).await;
        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "accepted {bad}"
        );
    }
    let alice = browser(&f, "alice").await;
    let view = json(
        request(
            &f.app,
            "GET",
            "/v1/account",
            &alice.cookie,
            None,
            None,
            None,
        )
        .await,
    )
    .await;
    assert!(view["account"]["tenant_id"].as_str().is_some());
    assert_eq!(
        f.authority
            .lock()
            .unwrap()
            .db
            .query_row::<i64, _, _>("SELECT count(*) FROM accounts", [], |r| r.get(0))
            .unwrap(),
        1
    );
}
#[tokio::test]
async fn two_browser_tenants_pair_independently_and_refuse_cross_tenant_effects() {
    let f = fixture().await;
    let alice = browser(&f, "alice").await;
    let bob = browser(&f, "bob").await;
    let a = json(
        request(
            &f.app,
            "GET",
            "/v1/account",
            &alice.cookie,
            None,
            None,
            None,
        )
        .await,
    )
    .await;
    let b = json(request(&f.app, "GET", "/v1/account", &bob.cookie, None, None, None).await).await;
    assert_ne!(a["account"]["tenant_id"], b["account"]["tenant_id"]);
    let key = SigningKey::from_bytes(&[33; 32]);
    let response=request(&f.app,"POST","/v1/enrollments","",None,None,Some(serde_json::json!({"public_key":key.verifying_key().to_bytes(),"capabilities":["model.infer"]}))).await;
    assert_eq!(response.status(), StatusCode::OK);
    let ticket: crate::PairTicket = serde_json::from_value(json(response).await).unwrap();
    let body = serde_json::json!({"user_code":ticket.user_code});
    for (origin, csrf) in [
        (None, Some(alice.csrf.as_str())),
        (Some("https://evil.example"), Some(alice.csrf.as_str())),
        (Some("http://127.0.0.1:3000"), Some("forged")),
    ] {
        assert_eq!(
            request(
                &f.app,
                "POST",
                "/v1/enrollments/claim",
                &alice.cookie,
                origin,
                csrf,
                Some(body.clone())
            )
            .await
            .status(),
            StatusCode::UNAUTHORIZED
        );
    }
    let response = request(
        &f.app,
        "POST",
        "/v1/enrollments/claim",
        &alice.cookie,
        Some("http://127.0.0.1:3000"),
        Some(&alice.csrf),
        Some(body.clone()),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let claim: crate::PairClaim = serde_json::from_value(json(response).await).unwrap();
    assert_eq!(claim.tenant_id, a["account"]["tenant_id"].as_str().unwrap());
    assert_eq!(
        request(
            &f.app,
            "POST",
            "/v1/enrollments/claim",
            &bob.cookie,
            Some("http://127.0.0.1:3000"),
            Some(&bob.csrf),
            Some(body)
        )
        .await
        .status(),
        StatusCode::UNAUTHORIZED
    );
    let certificate = f
        .authority
        .lock()
        .unwrap()
        .confirm_pairing(
            &ticket.enrollment_id,
            &ticket.poll_secret,
            &key.sign(&claim.signing_bytes().unwrap()).to_bytes(),
            now().unwrap(),
        )
        .unwrap();
    verify_revocation_and_deletion(&f, &alice, &bob, &certificate).await;
}
async fn verify_revocation_and_deletion(
    f: &Fixture,
    alice: &Browser,
    bob: &Browser,
    certificate: &crate::NodeCertificate,
) {
    let path = format!("/v1/nodes/{}/revoke", certificate.node_id);
    assert_eq!(
        request(
            &f.app,
            "POST",
            &path,
            &bob.cookie,
            Some("http://127.0.0.1:3000"),
            Some(&bob.csrf),
            None
        )
        .await
        .status(),
        StatusCode::UNAUTHORIZED
    );
    let bob_view =
        json(request(&f.app, "GET", "/v1/account", &bob.cookie, None, None, None).await).await;
    assert_eq!(bob_view["nodes"].as_array().unwrap().len(), 0);
    assert_eq!(
        request(
            &f.app,
            "POST",
            &path,
            &alice.cookie,
            Some("http://127.0.0.1:3000"),
            Some(&alice.csrf),
            None
        )
        .await
        .status(),
        StatusCode::NO_CONTENT
    );
    assert!(
        f.authority
            .lock()
            .unwrap()
            .challenge(&certificate.node_id, now().unwrap())
            .is_err()
    );
    assert_eq!(
        request(
            &f.app,
            "DELETE",
            "/v1/account",
            &alice.cookie,
            Some("http://127.0.0.1:3000"),
            Some(&alice.csrf),
            None
        )
        .await
        .status(),
        StatusCode::NO_CONTENT
    );
    assert_eq!(
        request(
            &f.app,
            "GET",
            "/v1/account",
            &alice.cookie,
            None,
            None,
            None
        )
        .await
        .status(),
        StatusCode::UNAUTHORIZED
    );
}
#[tokio::test]
async fn browser_assets_and_request_budgets_are_served_with_security_headers() {
    let f = fixture().await;
    for path in ["/", "/app.js", "/app.css"] {
        let response = request(&f.app, "GET", path, "", None, None, None).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["cache-control"], "no-store");
        assert!(
            response.headers()["content-security-policy"]
                .to_str()
                .unwrap()
                .contains("frame-ancestors 'none'")
        );
    }
    let response = request(
        &f.app,
        "POST",
        "/v1/enrollments",
        "",
        None,
        None,
        Some(serde_json::json!({"oversize":"x".repeat(20000)})),
    )
    .await;
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}
#[tokio::test]
async fn oauth_callback_is_browser_bound_and_consumed_once() {
    let f = fixture().await;
    let (path, cookie) = login_flow(&f, "alice", "").await;
    assert_eq!(
        request(&f.app, "GET", &path, "abi_login=forged", None, None, None)
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        request(&f.app, "GET", &path, &cookie, None, None, None)
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );
    let (path, cookie) = login_flow(&f, "alice", "").await;
    assert_eq!(
        request(&f.app, "GET", &path, &cookie, None, None, None)
            .await
            .status(),
        StatusCode::SEE_OTHER
    );
    assert_eq!(
        request(&f.app, "GET", &path, &cookie, None, None, None)
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );
}
#[tokio::test]
async fn node_http_confirmation_and_possession_use_fresh_authority() {
    let f = fixture().await;
    let alice = browser(&f, "alice").await;
    let key = SigningKey::from_bytes(&[44; 32]);
    let ticket:crate::PairTicket=serde_json::from_value(json(request(&f.app,"POST","/v1/enrollments","",None,None,Some(serde_json::json!({"public_key":key.verifying_key().to_bytes(),"capabilities":["model.infer"]}))).await).await).unwrap();
    let claim: crate::PairClaim = serde_json::from_value(
        json(
            request(
                &f.app,
                "POST",
                "/v1/enrollments/claim",
                &alice.cookie,
                Some("http://127.0.0.1:3000"),
                Some(&alice.csrf),
                Some(serde_json::json!({"user_code":ticket.user_code})),
            )
            .await,
        )
        .await,
    )
    .unwrap();
    let path = format!("/v1/enrollments/{}/confirm", ticket.enrollment_id);
    let signature = key
        .sign(&claim.signing_bytes().unwrap())
        .to_bytes()
        .to_vec();
    let body = serde_json::json!({"signature":signature});
    let missing = request(&f.app, "POST", &path, "", None, None, Some(body.clone())).await;
    assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);
    let response = f
        .app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(&path)
                .header(
                    header::AUTHORIZATION,
                    format!("Bearer {}", ticket.poll_secret),
                )
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let cert: crate::NodeCertificate = serde_json::from_value(json(response).await).unwrap();
    let challenge = f
        .authority
        .lock()
        .unwrap()
        .challenge(&cert.node_id, now().unwrap())
        .unwrap();
    let signature = key
        .sign(&challenge.signing_bytes().unwrap())
        .to_bytes()
        .to_vec();
    let proof = serde_json::json!({"challenge":challenge,"signature":signature});
    assert_eq!(
        request(
            &f.app,
            "POST",
            "/v1/nodes/authenticate",
            "",
            None,
            None,
            Some(proof.clone())
        )
        .await
        .status(),
        StatusCode::OK
    );
    assert_eq!(
        request(
            &f.app,
            "POST",
            "/v1/nodes/authenticate",
            "",
            None,
            None,
            Some(proof)
        )
        .await
        .status(),
        StatusCode::UNAUTHORIZED
    );
    verify_logout(&f, &alice).await;
}
async fn verify_logout(f: &Fixture, alice: &Browser) {
    assert_eq!(
        request(
            &f.app,
            "POST",
            "/v1/logout",
            &alice.cookie,
            Some("http://127.0.0.1:3000"),
            Some(&alice.csrf),
            None
        )
        .await
        .status(),
        StatusCode::NO_CONTENT
    );
    assert_eq!(
        request(
            &f.app,
            "GET",
            "/v1/account",
            &alice.cookie,
            None,
            None,
            None
        )
        .await
        .status(),
        StatusCode::UNAUTHORIZED
    );
}

mod relay;

mod streaming;

mod reconnect_process;

mod durable;

mod reconnect_fairness;

mod direct_recovery_guard;

mod governed_tools;
