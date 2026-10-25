//! Actual outbound TCP relay acceptance with two verified sessions and user runtimes.
use super::*;
use crate::{
    InferenceJob, JobView, NodeCertificate, PairRequest, RelayClient, RelayKey, RelayRuntime,
    prepare_job,
};
use abi_worker::{FileControlStore, FileControlStoreLimits, JobResult};
pub(super) async fn enroll(
    f: &Fixture,
    browser: &Browser,
    seed: u8,
) -> (NodeCertificate, SigningKey, RelayKey) {
    let view = json(
        request(
            &f.app,
            "GET",
            "/v1/account",
            &browser.cookie,
            None,
            None,
            None,
        )
        .await,
    )
    .await;
    let account: crate::Account = serde_json::from_value(view["account"].clone()).unwrap();
    let key = SigningKey::from_bytes(&[seed; 32]);
    let enc = RelayKey::from_seed(&[seed + 1; 32]).unwrap();
    let mut authority = f.authority.lock().unwrap();
    let time = now().unwrap();
    let ticket = authority
        .begin_pairing(
            &PairRequest {
                public_key: key.verifying_key().to_bytes(),
                encryption_public_key: Some(enc.public_key()),
                capabilities: vec!["model.infer".into()],
            },
            time,
        )
        .unwrap();
    let claim = authority
        .claim_pairing(&account, &ticket.user_code, time)
        .unwrap();
    let certificate = authority
        .confirm_pairing(
            &ticket.enrollment_id,
            &ticket.poll_secret,
            &key.sign(&claim.signing_bytes().unwrap()).to_bytes(),
            time,
        )
        .unwrap();
    (certificate, key, enc)
}
struct Count(std::sync::atomic::AtomicUsize);
impl abi_agent_runtime::ModelProvider for Count {
    fn id(&self) -> &'static str {
        "tcp-acceptance-echo"
    }
    fn run(
        &self,
        r: &abi_agent_runtime::ModelRequest,
        c: &mut abi_agent_runtime::RunContext<'_>,
    ) -> std::result::Result<(), abi_agent_runtime::RuntimeError> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        abi_agent_runtime::EchoProvider::new().run(r, c)
    }
}
pub(super) fn runtime(
    f: &Fixture,
    cert: &NodeCertificate,
    key: &SigningKey,
    encryption: RelayKey,
    label: &str,
) -> RelayRuntime<FileControlStore> {
    let store =
        FileControlStore::create(&f.dir.path().join(label), FileControlStoreLimits::default())
            .unwrap();
    RelayRuntime::new(
        cert.clone(),
        key.clone(),
        encryption,
        f.authority.lock().unwrap().public_key(),
        "local-echo-v1",
        store,
        now().unwrap(),
    )
    .unwrap()
}
#[tokio::test]
async fn two_tenants_complete_sealed_jobs_over_real_outbound_http() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let f = fixture_with_origin(origin.clone()).await;
    let alice = browser(&f, "alice").await;
    let bob = browser(&f, "bob").await;
    let (a, ka, ea) = enroll(&f, &alice, 70).await;
    let (b, kb, eb) = enroll(&f, &bob, 80).await;
    let mut ra = runtime(&f, &a, &ka, ea, "worker-a.log");
    let mut rb = runtime(&f, &b, &kb, eb, "worker-b.log");
    let ca = RelayClient::new(a.clone(), ka, true).unwrap();
    let cb = RelayClient::new(b.clone(), kb, true).unwrap();
    let service = f.app.clone();
    let server = tokio::spawn(async move {
        axum::serve(listener, service).await.unwrap();
    });
    let http = oauth2::reqwest::Client::new();
    let authority = f.authority.lock().unwrap().public_key();
    let reply_a = RelayKey::generate().unwrap();
    let reply_b = RelayKey::generate().unwrap();
    let sa = prepare_job(
        &a,
        &authority,
        &reply_a,
        &InferenceJob {
            prompt: format!("tenant-a-sensitive-{}", "x".repeat(5000)),
            tokens: 512,
            model: "local-echo-v1".into(),
        },
        now().unwrap(),
    )
    .unwrap();
    let sb = prepare_job(
        &b,
        &authority,
        &reply_b,
        &InferenceJob {
            prompt: "tenant-b-sensitive".into(),
            tokens: 512,
            model: "local-echo-v1".into(),
        },
        now().unwrap(),
    )
    .unwrap();
    assert!(serde_json::to_vec(&sa).unwrap().len() > 16384);
    for (browser, submission) in [(&alice, &sa), (&bob, &sb)] {
        let response = http
            .post(format!("{origin}/v1/jobs"))
            .header(header::COOKIE, &browser.cookie)
            .header(header::ORIGIN, &origin)
            .header("x-csrf-token", &browser.csrf)
            .header(header::CONTENT_TYPE, "application/json")
            .body(serde_json::to_vec(submission).unwrap())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), 200);
    }
    let cross = http
        .post(format!("{origin}/v1/jobs"))
        .header(header::COOKIE, &bob.cookie)
        .header(header::ORIGIN, &origin)
        .header("x-csrf-token", &bob.csrf)
        .header(header::CONTENT_TYPE, "application/json")
        .body(serde_json::to_vec(&sa).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(cross.status().as_u16(), 401);
    let provider_a = Count(std::sync::atomic::AtomicUsize::new(0));
    let provider_b = Count(std::sync::atomic::AtomicUsize::new(0));
    assert_eq!(
        cb.run_once(&mut rb, &provider_b)
            .await
            .unwrap()
            .unwrap()
            .state,
        "completed"
    );
    assert_eq!(provider_a.0.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(
        ca.run_once(&mut ra, &provider_a)
            .await
            .unwrap()
            .unwrap()
            .state,
        "completed"
    );
    assert!(ca.run_once(&mut ra, &provider_a).await.unwrap().is_none());
    assert_eq!(provider_a.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(provider_b.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    check_results(
        &f,
        &http,
        &origin,
        [(&alice, &sa, &reply_a), (&bob, &sb, &reply_b)],
    )
    .await;
    server.abort();
}
async fn check_results(
    f: &Fixture,
    http: &oauth2::reqwest::Client,
    origin: &str,
    cases: [(&Browser, &crate::Submission, &RelayKey); 2],
) {
    let [(alice, sa, reply_a), (bob, sb, reply_b)] = cases;
    let view_a = view_http(http, origin, alice, &sa.binding.job_id).await;
    let view_b = view_http(http, origin, bob, &sb.binding.job_id).await;
    let pa = reply_a
        .open(
            &sa.binding.aad("result").unwrap(),
            view_a.result.as_ref().unwrap(),
        )
        .unwrap();
    let pb = reply_b
        .open(
            &sb.binding.aad("result").unwrap(),
            view_b.result.as_ref().unwrap(),
        )
        .unwrap();
    let result_a: JobResult = serde_json::from_slice(&pa).unwrap();
    let result_b: JobResult = serde_json::from_slice(&pb).unwrap();
    assert!(result_a.verify_digest() && result_b.verify_digest());
    assert_eq!(result_a.status, abi_worker::JobStatus::Succeeded);
    assert!(String::from_utf8_lossy(&result_a.chunks[0].bytes).contains("tenant-a-sensitive"));
    assert!(String::from_utf8_lossy(&result_b.chunks[0].bytes).contains("tenant-b-sensitive"));
    assert!(
        reply_b
            .open(
                &sa.binding.aad("result").unwrap(),
                view_a.result.as_ref().unwrap()
            )
            .is_err()
    );
    let cross = http
        .get(format!("{origin}/v1/jobs/{}", sa.binding.job_id))
        .header(header::COOKIE, &bob.cookie)
        .send()
        .await
        .unwrap();
    assert_eq!(cross.status().as_u16(), 401);
    let db = std::fs::read(f.dir.path().join("control.sqlite")).unwrap();
    for canary in [
        b"tenant-a-sensitive".as_slice(),
        b"tenant-b-sensitive".as_slice(),
        &sa.payload.ciphertext[..16],
    ] {
        assert!(!db.windows(canary.len()).any(|b| b == canary));
    }
}
pub(super) async fn view_http(
    http: &oauth2::reqwest::Client,
    origin: &str,
    browser: &Browser,
    id: &str,
) -> JobView {
    let response = http
        .get(format!("{origin}/v1/jobs/{id}"))
        .header(header::COOKIE, &browser.cookie)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    serde_json::from_slice(&response.bytes().await.unwrap()).unwrap()
}
