//! Outbound local-echo research worker. No shell, tool or model-download dispatch.
#[allow(dead_code)]
mod support;
use abi_control_plane::{
    DurableRelayRuntime, Error, FileRelayOutbox, NodeCertificate, OutboxLimits, RelayClient,
    RelayKey, RelayRuntime, reconnect_delay,
};
use abi_worker::{FileControlStore, FileControlStoreLimits};
use std::path::Path;
use support::{Result, now, read_private};
#[tokio::main]
async fn main() {
    if run().await.is_err() {
        eprintln!(
            "Worker refused startup; check enrollment, expiry and private history. No state was reset."
        );
        std::process::exit(1);
    }
}
async fn run() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty()
        || args.len() > 4
        || args.iter().skip(1).any(|a| {
            !matches!(
                a.as_str(),
                "--initialize" | "--initialize-outbox" | "--loopback-development"
            )
        })
    {
        return Err(
            "usage: relay_worker NODE_DIRECTORY [--initialize | --initialize-outbox] [--loopback-development]".into(),
        );
    }
    if args.iter().any(|a| a == "--initialize") && args.iter().any(|a| a == "--initialize-outbox") {
        return Err("Initialization modes are mutually exclusive.".into());
    }
    let directory = Path::new(&args[0]);
    let cert: NodeCertificate = read_private(&directory.join("node-certificate.json"))?;
    let key =
        ed25519_dalek::SigningKey::from_bytes(&read_private(&directory.join("node-key.json"))?);
    let encryption = RelayKey::from_seed(&read_private(&directory.join("encryption-key.json"))?)?;
    let authority = read_private(&directory.join("authority-key.json"))?;
    let history = directory.join("worker-control.log");
    let limits = FileControlStoreLimits::default();
    let store = if args.iter().any(|a| a == "--initialize") {
        FileControlStore::create(&history, limits)?
    } else {
        FileControlStore::open(&history, limits)?
    };
    let runtime = RelayRuntime::new(
        cert.clone(),
        key.clone(),
        encryption,
        authority,
        "local-echo-v1",
        store,
        now()?,
    )?;
    let outbox_path = directory.join("relay-outbox.sqlite");
    let outbox = if args
        .iter()
        .any(|a| matches!(a.as_str(), "--initialize" | "--initialize-outbox"))
    {
        FileRelayOutbox::create(&outbox_path, OutboxLimits::default())?
    } else {
        FileRelayOutbox::open(&outbox_path, OutboxLimits::default())?
    };
    let mut runtime = DurableRelayRuntime::new(runtime, outbox)?;
    let mut attempt = 0u32;
    let client = RelayClient::new(
        cert,
        key,
        args.iter().any(|a| a == "--loopback-development"),
    )?;
    let provider = std::sync::Arc::new(abi_agent_runtime::EchoProvider::new());
    println!("Outbound local-echo-v1 worker ready; this probe is not a trained assistant model.");
    loop {
        tokio::select! {
            _=tokio::signal::ctrl_c()=>break,
            result=client.run_durable_once(runtime,provider.clone())=>{runtime=result.0;match result.1 {
                Ok(Some(receipt))=>{attempt=0;println!("job={} state={}",receipt.binding.job_id,receipt.state);},
                Ok(None)=>{attempt=0;},
                Err(Error::Unavailable|Error::Full)=>{attempt=attempt.saturating_add(1);eprintln!("Relay unavailable or full; retained encrypted output awaits reconnect.");},
                Err(_)=>return Err("Worker refused recovery; retain both journals for inspection.".into()),
            }}
        }
        tokio::time::sleep(if attempt == 0 {
            std::time::Duration::from_secs(2)
        } else {
            reconnect_delay(attempt - 1)
        })
        .await;
    }
    Ok(())
}
