//! Explicit initialization and loopback service entry point.
use abi_control_plane::{Authority, OidcConfig};
use serde::Deserialize;
use std::{net::SocketAddr, path::PathBuf};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    listen: SocketAddr,
    database: PathBuf,
    oidc: OidcConfig,
    #[serde(default)]
    relay_limits: abi_control_plane::RelayLimits,
    /// Hex SHA-256 of the operator bearer token; absent disables `/v1/operator/metrics`.
    #[serde(default)]
    operator_token_sha256: Option<String>,
}
/// Metrics window for the offline `metrics` command: one day of audit events.
const METRICS_WINDOW_SECS: i64 = 86_400;
#[tokio::main]
async fn main() {
    if run().await.is_err() {
        eprintln!(
            "control plane refused startup; check private state, configuration and OIDC provider"
        );
        std::process::exit(1);
    }
}
async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [command, path, audience] if command == "init" => {
            Authority::open(std::path::Path::new(path), true, audience)?;
            println!("Initialized private account authority");
        }
        [command, path] if command == "serve" => {
            let config: Config = serde_json::from_slice(&read_config(path)?)?;
            if !config.listen.ip().is_loopback() {
                return Err("listener must be loopback behind a TLS reverse proxy".into());
            }
            let operator = config
                .operator_token_sha256
                .as_deref()
                .map(parse_digest)
                .transpose()?;
            let app = abi_control_plane::router_with_operator(
                &config.database,
                config.oidc,
                config.relay_limits,
                operator,
            )
            .await?;
            let listener = tokio::net::TcpListener::bind(config.listen).await?;
            println!("ABI account authority listening on {}", config.listen);
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await?;
        }
        [command, path, audience] if command == "metrics" => {
            let authority = Authority::open(std::path::Path::new(path), false, audience)?;
            let now = i64::try_from(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs(),
            )?;
            let metrics = authority.operator_metrics(now, METRICS_WINDOW_SECS)?;
            println!("{}", serde_json::to_string_pretty(&metrics)?);
        }
        _ => {
            return Err(
                "usage: abi-control-plane init DATABASE ORIGIN | serve CONFIG.json | metrics DATABASE ORIGIN"
                    .into(),
            );
        }
    }
    Ok(())
}
fn parse_digest(hex: &str) -> Result<[u8; 32], Box<dyn std::error::Error>> {
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("operator_token_sha256 must be 64 hex digits".into());
    }
    let mut out = [0_u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)?;
    }
    Ok(out)
}
#[cfg(unix)]
fn read_config(path: &str) -> std::io::Result<Vec<u8>> {
    use std::{
        io::Read as _,
        os::unix::fs::{MetadataExt as _, OpenOptionsExt as _},
    };
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.mode() & 0o077 != 0
        || metadata.uid() != nix::unistd::Uid::effective().as_raw()
        || metadata.nlink() != 1
        || metadata.len() > 65536
    {
        return Err(std::io::Error::other(
            "configuration must be a private bounded file",
        ));
    }
    let mut bytes = Vec::new();
    file.take(65537).read_to_end(&mut bytes)?;
    if bytes.len() > 65536 {
        return Err(std::io::Error::other("configuration exceeds size limit"));
    }
    Ok(bytes)
}
#[cfg(not(unix))]
fn read_config(_path: &str) -> std::io::Result<Vec<u8>> {
    Err(std::io::Error::other("platform ACL validation unavailable"))
}
