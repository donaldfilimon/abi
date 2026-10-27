use super::*;

const ORIGIN: &str = "http://127.0.0.1:3000";
const TOKEN: &str = "operator-test-token";

async fn get(app: &Router, path: &str, bearer: Option<&str>) -> Response {
    let mut builder = Request::builder().method("GET").uri(path);
    if let Some(token) = bearer {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {token}"));
    }
    app.clone()
        .oneshot(builder.body(Body::empty()).unwrap())
        .await
        .unwrap()
}
async fn body(response: Response) -> String {
    String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap()
}
fn digest(token: &str) -> [u8; 32] {
    Sha256::digest(token.as_bytes()).into()
}

#[tokio::test]
async fn healthz_reports_ok_without_content() {
    let f = fixture().await;
    let response = get(&f.app, "/healthz", None).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body(response).await, r#"{"status":"ok"}"#);
}

#[tokio::test]
async fn operator_metrics_is_absent_when_unconfigured() {
    let f = fixture().await;
    let response = get(&f.app, "/v1/operator/metrics", Some(TOKEN)).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn operator_metrics_refuses_missing_and_wrong_token() {
    let f = fixture_with_operator(ORIGIN.into(), Some(digest(TOKEN))).await;
    for bearer in [None, Some("wrong-token"), Some("")] {
        let response = get(&f.app, "/v1/operator/metrics", bearer).await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{bearer:?}");
        let text = body(response).await;
        assert!(!text.chars().any(|c| c.is_ascii_digit()), "{text}");
    }
}

#[tokio::test]
async fn operator_metrics_counts_denied_requests() {
    let f = fixture_with_operator(ORIGIN.into(), Some(digest(TOKEN))).await;
    let alice = browser(&f, "alice").await;
    assert_eq!(
        get(&f.app, "/v1/account", None).await.status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        get(&f.app, "/v1/account", None).await.status(),
        StatusCode::UNAUTHORIZED
    );
    let response = get(&f.app, "/v1/operator/metrics", Some(TOKEN)).await;
    assert_eq!(response.status(), StatusCode::OK);
    let text = body(response).await;
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(value["auth_denied_total"].as_u64().unwrap() >= 2);
    assert_eq!(value["metrics"]["accounts"], 1);
    assert_eq!(value["metrics"]["audit_window_secs"], 86_400);
    let account = f
        .authority
        .lock()
        .unwrap()
        .db
        .query_row::<(String, String), _, _>("SELECT id,tenant FROM accounts", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    for secret in [
        account.0.as_str(),
        account.1.as_str(),
        "alice",
        alice.csrf.as_str(),
    ] {
        assert!(!text.contains(secret), "leaked {secret}: {text}");
    }
}
