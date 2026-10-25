//! Explicit local enrollment client. No automatic approval or workload dispatch.
use abi_control_plane::{NodeCertificate, NodeChallenge, PairClaim, PairRequest, PairTicket};
use ed25519_dalek::{Signer as _, SigningKey};
use oauth2::reqwest;
use serde::de::DeserializeOwned;
use std::{
    io::Write as _,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn enrollment_capability(args: &[String]) -> Result<&str> {
    if !(2..=3).contains(&args.len()) {
        return Err("usage: enroll SERVER_ORIGIN NEW_PRIVATE_DIRECTORY [--tools]".into());
    }
    match args.get(2).map(String::as_str) {
        None => Ok("model.infer"),
        Some("--tools") => Ok("tool.execute"),
        _ => Err("unknown enrollment capability".into()),
    }
}
#[tokio::main]
async fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let capability = enrollment_capability(&args)?;
    let (server, directory) = (&args[0], &args[1]);
    let client = http_client(server)?;
    let authority: [u8; 32] =
        response(client.get(format!("{server}/v1/authority")).send().await?).await?;
    let key = create_key(Path::new(directory))?;
    let mut seed = [0; 32];
    seed[..16].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
    seed[16..].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
    write_private(
        &Path::new(directory).join("encryption-key.json"),
        &serde_json::to_vec(&seed)?,
    )?;
    let encryption = abi_control_plane::RelayKey::from_seed(&seed)?;
    let request = PairRequest {
        public_key: key.verifying_key().to_bytes(),
        encryption_public_key: Some(encryption.public_key()),
        capabilities: vec![capability.into()],
    };
    let ticket: PairTicket = response(
        client
            .post(format!("{server}/v1/enrollments"))
            .header("content-type", "application/json")
            .body(serde_json::to_vec(&request)?)
            .send()
            .await?,
    )
    .await?;
    println!(
        "Open {server} and pair with code:\n{}\nLocal public key: {:?}",
        ticket.user_code, request.public_key
    );
    let claim = await_claim(&client, server, &ticket).await?;
    if claim.encryption_public_key != request.encryption_public_key
        || claim.public_key != request.public_key
        || claim.capabilities != request.capabilities
        || claim.enrollment_id != ticket.enrollment_id
        || claim.expires_at != ticket.expires_at
        || claim.protocol != "abi.local-enrollment.v1"
    {
        return Err("enrollment preview differs from the local request".into());
    }
    approve_locally(&claim)?;
    let signature = key.sign(&claim.signing_bytes()?).to_bytes().to_vec();
    let cert: NodeCertificate = response(
        client
            .post(format!(
                "{server}/v1/enrollments/{}/confirm",
                ticket.enrollment_id
            ))
            .bearer_auth(&ticket.poll_secret)
            .header("content-type", "application/json")
            .body(serde_json::to_vec(
                &serde_json::json!({"signature":signature}),
            )?)
            .send()
            .await?,
    )
    .await?;
    cert.verify(&authority, server, now()?)?;
    if cert.claim != claim || cert.node_id != ticket.enrollment_id {
        return Err("issued credential differs from local approval".into());
    }
    write_private(
        &Path::new(directory).join("authority-key.json"),
        &serde_json::to_vec(&authority)?,
    )?;
    write_private(
        &Path::new(directory).join("node-certificate.json"),
        &serde_json::to_vec_pretty(&cert)?,
    )?;
    let challenge = NodeChallenge {
        protocol: "abi.node-proof.v1".into(),
        audience: server.clone(),
        node_id: cert.node_id.clone(),
        nonce: format!("{}{}", uuid::Uuid::new_v4(), uuid::Uuid::new_v4()),
        expires_at: now()? + 60,
    };
    let signature = key.sign(&challenge.signing_bytes()?).to_bytes().to_vec();
    let authenticated: NodeCertificate = response(
        client
            .post(format!("{server}/v1/nodes/authenticate"))
            .header("content-type", "application/json")
            .body(serde_json::to_vec(
                &serde_json::json!({"challenge":challenge,"signature":signature}),
            )?)
            .send()
            .await?,
    )
    .await?;
    if authenticated != cert {
        return Err("authentication result differs from issued credential".into());
    }
    println!(
        "Node {} enrolled and possession verified. No worker job was dispatched.",
        cert.node_id
    );
    Ok(())
}
fn http_client(server: &str) -> Result<reqwest::Client> {
    let url = oauth2::url::Url::parse(server)?;
    if url.origin().ascii_serialization() != server
        || !(url.scheme() == "https"
            || (url.scheme() == "http" && matches!(url.host_str(), Some("127.0.0.1" | "[::1]"))))
    {
        return Err("use an HTTPS origin or literal loopback development origin".into());
    }
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    Ok(client)
}
async fn await_claim(
    client: &reqwest::Client,
    server: &str,
    ticket: &PairTicket,
) -> Result<PairClaim> {
    loop {
        if now()? >= ticket.expires_at {
            return Err("pairing expired; the private key remains in the new directory".into());
        }
        let claim: Option<PairClaim> = response(
            client
                .get(format!("{server}/v1/enrollments/{}", ticket.enrollment_id))
                .bearer_auth(&ticket.poll_secret)
                .send()
                .await?,
        )
        .await?;
        if let Some(claim) = claim {
            return Ok(claim);
        }
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }
}
fn approve_locally(claim: &PairClaim) -> Result<()> {
    println!(
        "Exact local approval preview:\n{}\nType the account ID to approve, or anything else to refuse:",
        serde_json::to_string_pretty(&claim)?
    );
    std::io::stdout().flush()?;
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    if answer.trim() != claim.account_id {
        return Err("local enrollment refused".into());
    }
    Ok(())
}
async fn response<T: DeserializeOwned>(mut response: reqwest::Response) -> Result<T> {
    if !response.status().is_success() {
        return Err(format!("authority refused request ({})", response.status()).into());
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if bytes.len() + chunk.len() > 65536 {
            return Err("authority response too large".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(serde_json::from_slice(&bytes)?)
}
fn now() -> Result<i64> {
    Ok(i64::try_from(
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
    )?)
}
#[cfg(unix)]
fn create_key(directory: &Path) -> Result<SigningKey> {
    use std::os::unix::fs::DirBuilderExt as _;
    std::fs::DirBuilder::new().mode(0o700).create(directory)?;
    let mut bytes = [0; 32];
    bytes[..16].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
    bytes[16..].copy_from_slice(uuid::Uuid::new_v4().as_bytes());
    write_private(
        &directory.join("node-key.json"),
        &serde_json::to_vec(&bytes)?,
    )?;
    Ok(SigningKey::from_bytes(&bytes))
}
#[cfg(unix)]
fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::os::unix::fs::OpenOptionsExt as _;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    std::fs::File::open(path.parent().ok_or("missing parent")?)?.sync_all()?;
    Ok(())
}
#[cfg(not(unix))]
fn create_key(_directory: &Path) -> Result<SigningKey> {
    Err("platform credential ACL implementation unavailable".into())
}
#[cfg(not(unix))]
fn write_private(_path: &Path, _bytes: &[u8]) -> Result<()> {
    Err("platform credential ACL implementation unavailable".into())
}
