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
}
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
            let app = abi_control_plane::router_with_limits(
                &config.database,
                config.oidc,
                config.relay_limits,
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
        _ => {
            return Err("usage: abi-control-plane init DATABASE ORIGIN | serve CONFIG.json".into());
        }
    }
    Ok(())
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
