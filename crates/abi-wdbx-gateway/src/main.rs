//! `abi-wdbx-gateway` server binary.

use abi_wdbx_gateway::{GatewayConfig, PreparedGateway, TlsFiles};
use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(about = "Authenticated bounded WDBX gRPC and WebSocket gateway")]
struct Arguments {
    #[arg(long, default_value = "127.0.0.1:50051")]
    grpc: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:50052")]
    events: SocketAddr,
    #[arg(long)]
    store: PathBuf,
    #[arg(long)]
    token_file: PathBuf,
    #[arg(long)]
    tls_cert: Option<PathBuf>,
    #[arg(long)]
    tls_key: Option<PathBuf>,
    #[arg(long)]
    client_ca: Option<PathBuf>,
    #[arg(long = "origin", default_values_t = [
        "http://127.0.0.1".to_owned(),
        "http://localhost".to_owned(),
    ])]
    origins: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    let mut config = GatewayConfig::loopback(arguments.store, arguments.token_file);
    config.grpc_addr = arguments.grpc;
    config.events_addr = arguments.events;
    config.allowed_origins = arguments.origins;
    config.tls = TlsFiles {
        certificate: arguments.tls_cert,
        private_key: arguments.tls_key,
        client_ca: arguments.client_ca,
    };
    let gateway = PreparedGateway::prepare(config).await?;
    let addresses = gateway
        .serve(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    eprintln!(
        "abi-wdbx-gateway stopped (grpc={}, events={})",
        addresses.grpc, addresses.events
    );
    Ok(())
}
