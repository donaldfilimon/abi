//! Real TCP: frames are visible before completion, and cancellation reaches an idle provider.
use super::relay::{enroll, runtime, view_http};
use super::*;
use crate::{InferenceJob, RelayClient, RelayKey, prepare_job};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
struct Gated {
    calls: AtomicUsize,
    release: AtomicBool,
    stopped: AtomicBool,
}
impl Gated {
    fn new(release: bool) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            release: AtomicBool::new(release),
            stopped: AtomicBool::new(false),
        }
    }
}
impl abi_agent_runtime::ModelProvider for Gated {
    fn id(&self) -> &'static str {
        "streaming-test"
    }
    fn run(
        &self,
        r: &abi_agent_runtime::ModelRequest,
        c: &mut abi_agent_runtime::RunContext<'_>,
    ) -> std::result::Result<(), abi_agent_runtime::RuntimeError> {
        use abi_agent_runtime::ModelEvent;
        self.calls.fetch_add(1, Ordering::SeqCst);
        if c.emit(&ModelEvent::started(r.model())).is_stop() {
            return Ok(());
        }
        if c.emit(&ModelEvent::text("private-first-frame")).is_stop() {
            return Ok(());
        }
        while !self.release.load(Ordering::SeqCst) {
            if c.checkpoint().is_stop() {
                self.stopped.store(true, Ordering::SeqCst);
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        c.emit(&ModelEvent::text("private-second-frame"));
        Ok(())
    }
}
#[tokio::test]
async fn encrypted_frames_arrive_live_and_owner_cancellation_stops_only_its_worker() {
    tcp_case(false).await;
}
#[tokio::test]
async fn cancellation_racing_frame_delivery_is_acknowledged_after_provider_stops() {
    tcp_case(true).await;
}
#[allow(clippy::too_many_lines)] // One bounded end-to-end fixture keeps both tenants and race assertions together.
async fn tcp_case(cancel_on_frame: bool) {
    tokio::time::timeout(std::time::Duration::from_secs(15), async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let f = fixture_with_origin(origin.clone()).await;
        let alice = browser(&f, "stream-alice").await;
        let bob = browser(&f, "stream-bob").await;
        let (a, ka, ea) = enroll(&f, &alice, 90).await;
        let (b, kb, eb) = enroll(&f, &bob, 100).await;
        let ra = runtime(&f, &a, &ka, ea, "stream-a.log");
        let rb = runtime(&f, &b, &kb, eb, "stream-b.log");
        let ca = RelayClient::new(a.clone(), ka, true).unwrap();
        let cb = RelayClient::new(b.clone(), kb, true).unwrap();
        let authority = f.authority.lock().unwrap().public_key();
        let reply_a = RelayKey::generate().unwrap();
        let reply_b = RelayKey::generate().unwrap();
        let input = InferenceJob {
            prompt: "private-stream-prompt".into(),
            tokens: 100,
            model: "local-echo-v1".into(),
        };
        let sa = prepare_job(&a, &authority, &reply_a, &input, now().unwrap()).unwrap();
        let sb = prepare_job(&b, &authority, &reply_b, &input, now().unwrap()).unwrap();
        let service = f.app.clone();
        let cancel_app = f.app.clone();
        let cancel_cookie = alice.cookie.clone();
        let cancel_csrf = alice.csrf.clone();
        let cancel_origin = origin.clone();
        let cancel_id = sa.binding.job_id.clone();
        let fired=Arc::new(AtomicBool::new(false));
        let service=service.layer(axum::middleware::from_fn(move |req: axum::extract::Request, next: axum::middleware::Next| {
            let app=cancel_app.clone();let cookie=cancel_cookie.clone();let csrf=cancel_csrf.clone();let origin=cancel_origin.clone();let id=cancel_id.clone();let fired=fired.clone();
            async move {
                if req.uri().path()!="/v1/relay/node" {return next.run(req).await;}
                let (parts,body)=req.into_parts();let bytes=axum::body::to_bytes(body,384*1024).await.unwrap();
                if cancel_on_frame && serde_json::from_slice::<crate::NodeMessage>(&bytes).is_ok_and(|m|matches!(m.action,crate::NodeAction::Stream{job_id,..} if job_id==id)) && !fired.swap(true,Ordering::SeqCst) {
                    assert_eq!(request(&app,"POST",&format!("/v1/jobs/{id}/cancel"),&cookie,Some(&origin),Some(&csrf),None).await.status(),StatusCode::OK);
                }
                next.run(axum::extract::Request::from_parts(parts,axum::body::Body::from(bytes))).await
            }
        }));
        let server=tokio::spawn(async move{axum::serve(listener,service).await.unwrap();});
        let http = oauth2::reqwest::Client::new();
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
        let pa = Arc::new(Gated::new(false));
        let pb = Arc::new(Gated::new(true));
        let worker_a = tokio::spawn({
            let pa = pa.clone();
            async move {
                let (runtime, result) = ca.run_streaming_once(ra, pa).await;
                (ca, runtime, result)
            }
        });
        let worker_b = tokio::spawn({
            let pb = pb.clone();
            async move { cb.run_streaming_once(rb, pb).await }
        });
        if cancel_on_frame {
            let (_,_,result)=worker_a.await.unwrap();
            assert_eq!(result.unwrap().unwrap().state,"cancelled");
            let (_,result)=worker_b.await.unwrap();assert_eq!(result.unwrap().unwrap().state,"completed");
            assert_eq!(pa.calls.load(Ordering::SeqCst),1);
            assert!(view_http(&http,&origin,&alice,&sa.binding.job_id).await.frames.is_empty());
            server.abort();return;
        }
        let before = loop {
            let v = view_http(&http, &origin, &alice, &sa.binding.job_id).await;
            if !v.frames.is_empty() {
                break v;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        };
        assert_eq!(before.receipt.state, "dispatched");
        assert!(before.result.is_none());
        assert_eq!(
            before.frames[0]
                .open(&sa.binding, &a.claim.public_key, &reply_a)
                .unwrap(),
            b"private-first-frame"
        );
        assert!(
            before.frames[0]
                .open(&sa.binding, &a.claim.public_key, &reply_b)
                .is_err()
        );
        let path = format!("{origin}/v1/jobs/{}/cancel", sa.binding.job_id);
        let cross = http
            .post(&path)
            .header(header::COOKIE, &bob.cookie)
            .header(header::ORIGIN, &origin)
            .header("x-csrf-token", &bob.csrf)
            .send()
            .await
            .unwrap();
        assert_eq!(cross.status().as_u16(), 401);
        let cancel = http
            .post(&path)
            .header(header::COOKIE, &alice.cookie)
            .header(header::ORIGIN, &origin)
            .header("x-csrf-token", &alice.csrf)
            .send()
            .await
            .unwrap();
        assert_eq!(cancel.status().as_u16(), 200);
        let (ca, mut ra, result) = worker_a.await.unwrap();
        assert_eq!(result.unwrap().unwrap().state, "cancelled");
        let (_, result) = worker_b.await.unwrap();
        assert_eq!(result.unwrap().unwrap().state, "completed");
        assert!(pa.stopped.load(Ordering::SeqCst));
        assert_eq!(pa.calls.load(Ordering::SeqCst), 1);
        assert_eq!(pb.calls.load(Ordering::SeqCst), 1);
        let after = view_http(&http, &origin, &alice, &sa.binding.job_id).await;
        assert_eq!(after.frames.len(), 1);
        assert_eq!(after.receipt.state, "cancelled");
        let other = view_http(&http, &origin, &bob, &sb.binding.job_id).await;
        assert_eq!(other.frames.len(), 2);
        assert_eq!(
            other.frames[1]
                .open(&sb.binding, &b.claim.public_key, &reply_b)
                .unwrap(),
            b"private-second-frame"
        );
        assert!(ca.run_once(&mut ra, pa.as_ref()).await.unwrap().is_none());
        assert_eq!(pa.calls.load(Ordering::SeqCst), 1);
        let db = std::fs::read(f.dir.path().join("control.sqlite")).unwrap();
        for secret in [
            b"private-first-frame".as_slice(),
            b"private-stream-prompt".as_slice(),
            &before.frames[0].payload.ciphertext[..16],
        ] {
            assert!(!db.windows(secret.len()).any(|w| w == secret));
        }
        server.abort();
    })
    .await
    .expect("bounded cooperative cancellation acceptance timed out");
}
