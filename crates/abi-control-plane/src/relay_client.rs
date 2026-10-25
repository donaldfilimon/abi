//! User-node outbound client. No automatic resend of ambiguous dispatch or execution.
use crate::{
    Error, InferenceJob, JobBinding, JobReceipt, NodeAction, NodeCertificate, NodeMessage,
    NodeReply, RelayKey, RelayRuntime, Result, Submission, seal, service::now,
};
use abi_agent_runtime::ModelProvider;
use abi_worker::JobControlStore;
use ed25519_dalek::{Signer as _, SigningKey};
use oauth2::reqwest;
/// Outbound-only authenticated worker relay connection.
pub struct RelayClient {
    pub(crate) recovery_cursor: std::sync::atomic::AtomicUsize,
    pub(crate) recovered: std::sync::Mutex<std::collections::BTreeMap<String, (String, String)>>,
    http: reqwest::Client,
    pub(crate) certificate: NodeCertificate,
    pub(crate) key: SigningKey,
}
impl RelayClient {
    /// Use the enrolled origin; HTTP is allowed only when explicitly enabled on literal loopback.
    pub fn new(
        certificate: NodeCertificate,
        key: SigningKey,
        allow_loopback_http: bool,
    ) -> Result<Self> {
        let url = oauth2::url::Url::parse(&certificate.audience).map_err(|_| Error::Invalid)?;
        if url.origin().ascii_serialization() != certificate.audience
            || !(url.scheme() == "https"
                || (allow_loopback_http
                    && url.scheme() == "http"
                    && matches!(url.host_str(), Some("127.0.0.1" | "[::1]"))))
            || key.verifying_key().to_bytes() != certificate.claim.public_key
        {
            return Err(Error::Denied);
        }
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .timeout(std::time::Duration::from_secs(15))
            .connect_timeout(std::time::Duration::from_secs(3))
            .build()
            .map_err(|_| Error::Invalid)?;
        Ok(Self {
            recovery_cursor: std::sync::atomic::AtomicUsize::new(0),
            recovered: std::sync::Mutex::new(std::collections::BTreeMap::new()),
            http,
            certificate,
            key,
        })
    }
    /// Send one action with a fresh bound signature. Errors are never retried automatically.
    pub async fn exchange(&self, action: NodeAction, time: i64) -> Result<NodeReply> {
        let mut message = NodeMessage::sign(&self.certificate, &self.key, action, time)?;
        // Frequent streaming inspections must fit the existing 64-live-proof ceiling.
        message.proof.expires_at = time + 5;
        message.signature = self
            .key
            .sign(&message.proof.signing_bytes()?)
            .to_bytes()
            .to_vec();
        let body = serde_json::to_vec(&message).map_err(|_| Error::Invalid)?;
        if body.len() > 384 * 1024 {
            return Err(Error::Invalid);
        }
        let mut response = self
            .http
            .post(format!("{}/v1/relay/node", self.certificate.audience))
            .header("content-type", "application/json")
            .body(body)
            .send()
            .await
            .map_err(|_| Error::Unavailable)?;
        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(Error::Full);
        }
        if response.status().is_server_error() {
            return Err(Error::Unavailable);
        }
        if !response.status().is_success() {
            return Err(Error::Denied);
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|_| Error::Unavailable)? {
            if bytes.len() + chunk.len() > 384 * 1024 {
                return Err(Error::Invalid);
            }
            bytes.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&bytes).map_err(|_| Error::Denied)
    }
    /// Poll, obtain a fresh authority permit, execute locally once, and encrypt the result.
    /// A failure after dispatch leaves a recorded unresolved outcome; no repeat is authorized.
    pub async fn run_once<S: JobControlStore>(
        &self,
        runtime: &mut RelayRuntime<S>,
        provider: &dyn ModelProvider,
    ) -> Result<Option<JobReceipt>> {
        let reply = self.exchange(NodeAction::Poll, now()?).await?;
        let Some(dispatch) = reply.dispatch else {
            return Ok(None);
        };
        let id = dispatch.submission.binding.job_id.clone();
        let preflight = self
            .exchange(NodeAction::Inspect { job_id: id.clone() }, now()?)
            .await?;
        let Some(permit) = preflight.permit else {
            return Ok(preflight.receipt);
        };
        let action = match runtime.execute(&dispatch, &permit, provider, || {
            now().unwrap_or(i64::MAX / 1000)
        }) {
            Ok(Some(payload)) => NodeAction::Complete {
                job_id: id,
                payload,
            },
            Ok(None) | Err(_) => NodeAction::Fail { job_id: id },
        };
        Ok(self.exchange(action, now()?).await?.receipt)
    }
}
/// Prepare a client-encrypted job; account and tenant are taken from the enrolled certificate.
/// The caller keeps `reply_key` private on the submitting client.
pub fn prepare_job(
    certificate: &NodeCertificate,
    authority_key: &[u8; 32],
    reply_key: &RelayKey,
    input: &InferenceJob,
    time: i64,
) -> Result<Submission> {
    certificate.verify(authority_key, &certificate.audience, time)?;
    if input.prompt.len() > 8192 || !(1..=1024).contains(&input.tokens) {
        return Err(Error::Invalid);
    }
    prepare_bound(certificate, reply_key, input, "model.infer", time)
}
/// Encrypt a governed proposal for a tool-enrolled node. Submit only through v2.
pub fn prepare_tool_job(
    certificate: &NodeCertificate,
    authority_key: &[u8; 32],
    reply_key: &RelayKey,
    input: &crate::ToolJob,
    time: i64,
) -> Result<Submission> {
    certificate.verify(authority_key, &certificate.audience, time)?;
    input.validate()?;
    prepare_bound(certificate, reply_key, input, "tool.execute", time)
}
fn prepare_bound(
    certificate: &NodeCertificate,
    reply_key: &RelayKey,
    input: &impl serde::Serialize,
    capability: &str,
    time: i64,
) -> Result<Submission> {
    if !certificate
        .claim
        .capabilities
        .iter()
        .any(|c| c == capability)
    {
        return Err(Error::Denied);
    }
    let binding = JobBinding {
        protocol: "abi.relay-job.v1".into(),
        audience: certificate.audience.clone(),
        job_id: uuid::Uuid::new_v4().to_string(),
        account_id: certificate.claim.account_id.clone(),
        tenant_id: certificate.claim.tenant_id.clone(),
        node_id: certificate.node_id.clone(),
        capability: capability.into(),
        submitted_at: time,
        expires_at: (time + 120).min(certificate.expires_at),
        max_result_bytes: 65536,
        reply_key: reply_key.public_key(),
    };
    let key = certificate
        .claim
        .encryption_public_key
        .ok_or(Error::Denied)?;
    let plaintext = serde_json::to_vec(input).map_err(|_| Error::Invalid)?;
    let payload = seal(&key, &binding.aad("request")?, &plaintext)?;
    Ok(Submission { binding, payload })
}
