//! Two-listener gateway preparation and serving.

use crate::events::websocket_router;
use crate::proto::wdbx_gateway_server::WdbxGatewayServer;
use crate::{
    BearerToken, EventHub, GatewayConfig, GatewayError, GatewayService, StoreExecutor,
    grpc_tls_config, websocket_tls_config,
};
use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio_stream::wrappers::TcpListenerStream;

enum FirstExit<G, E> {
    Shutdown,
    Grpc(G),
    Events(E),
}

/// Actual bound addresses, including OS-selected ports for port-zero configs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundAddresses {
    /// gRPC address.
    pub grpc: SocketAddr,
    /// WebSocket event address.
    pub events: SocketAddr,
}

/// Fully validated state. Construction performs every credential, TLS, and
/// store check before serving is allowed to bind a socket.
pub struct PreparedGateway {
    config: GatewayConfig,
    token: Arc<BearerToken>,
    executor: StoreExecutor,
    events: Arc<EventHub>,
    grpc_tls: Option<tonic::transport::ServerTlsConfig>,
    websocket_tls: Option<axum_server::tls_rustls::RustlsConfig>,
}

impl std::fmt::Debug for PreparedGateway {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedGateway")
            .field("grpc_addr", &self.config.grpc_addr)
            .field("events_addr", &self.config.events_addr)
            .field("tls", &self.config.tls.is_complete())
            .finish_non_exhaustive()
    }
}

impl PreparedGateway {
    /// Validate and load all state before either listener can bind.
    pub async fn prepare(config: GatewayConfig) -> Result<Self, GatewayError> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        config.validate_pre_bind()?;
        let token = Arc::new(BearerToken::load(&config.token_file)?);
        let grpc_tls = grpc_tls_config(&config.tls)?;
        let websocket_tls = websocket_tls_config(&config.tls).await?;
        let executor = StoreExecutor::open(&config.store_path, config.limits.blocking_jobs)?;
        let events = Arc::new(EventHub::new(&config.limits));
        Ok(Self {
            config,
            token,
            executor,
            events,
            grpc_tls,
            websocket_tls,
        })
    }

    /// Serve both transports until shutdown or the first transport failure.
    pub async fn serve<F>(self, shutdown: F) -> Result<BoundAddresses, GatewayError>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let (grpc_listener, events_listener, addresses) = bind_listeners(&self.config).await?;

        let service = GatewayService::new(
            Arc::clone(&self.token),
            self.config.limits.clone(),
            self.executor,
            Arc::clone(&self.events),
        );
        let grpc_service = WdbxGatewayServer::new(service)
            .max_decoding_message_size(self.config.limits.message_bytes)
            .max_encoding_message_size(self.config.limits.message_bytes);
        let (stop_sender, stop_receiver) = tokio::sync::watch::channel(false);
        let mut grpc_builder = tonic::transport::Server::builder();
        if let Some(tls) = self.grpc_tls {
            grpc_builder = grpc_builder
                .tls_config(tls)
                .map_err(|error| GatewayError::Transport(format!("gRPC TLS failed: {error}")))?;
        }
        let mut grpc_stop = stop_receiver.clone();
        let grpc_task = tokio::spawn(async move {
            grpc_builder
                .add_service(grpc_service)
                .serve_with_incoming_shutdown(TcpListenerStream::new(grpc_listener), async move {
                    while !*grpc_stop.borrow() {
                        if grpc_stop.changed().await.is_err() {
                            break;
                        }
                    }
                })
                .await
        });

        let application = websocket_router(
            self.token,
            self.config.allowed_origins,
            self.events,
            self.config.limits.message_bytes,
            self.config.limits.requests_per_second,
        );
        let event_handle = axum_server::Handle::new();
        let event_shutdown = event_handle.clone();
        let websocket_tls = self.websocket_tls;
        let event_task = tokio::spawn(async move {
            if let Some(tls) = websocket_tls {
                axum_server::from_tcp_rustls(events_listener, tls)?
                    .handle(event_handle)
                    .serve(application.into_make_service())
                    .await
            } else {
                axum_server::from_tcp(events_listener)?
                    .handle(event_handle)
                    .serve(application.into_make_service())
                    .await
            }
        });

        tokio::pin!(shutdown);
        let mut grpc_task = grpc_task;
        let mut event_task = event_task;
        let first = tokio::select! {
            () = &mut shutdown => FirstExit::Shutdown,
            result = &mut grpc_task => FirstExit::Grpc(result),
            result = &mut event_task => FirstExit::Events(result),
        };
        let _ = stop_sender.send(true);
        event_shutdown.graceful_shutdown(Some(std::time::Duration::from_secs(5)));
        match first {
            FirstExit::Shutdown => {
                let (grpc_result, event_result) = tokio::join!(
                    finish_task(grpc_task, "gRPC"),
                    finish_task(event_task, "event")
                );
                grpc_result?;
                event_result?;
            }
            FirstExit::Grpc(result) => {
                let grpc_result = flatten_task_result(result, "gRPC");
                let event_result = finish_task(event_task, "event").await;
                grpc_result?;
                event_result?;
            }
            FirstExit::Events(result) => {
                let event_result = flatten_task_result(result, "event");
                let grpc_result = finish_task(grpc_task, "gRPC").await;
                event_result?;
                grpc_result?;
            }
        }
        Ok(addresses)
    }
}

async fn bind_listeners(
    config: &GatewayConfig,
) -> Result<
    (
        tokio::net::TcpListener,
        std::net::TcpListener,
        BoundAddresses,
    ),
    GatewayError,
> {
    let grpc = tokio::net::TcpListener::bind(config.grpc_addr)
        .await
        .map_err(|error| GatewayError::Transport(format!("gRPC bind failed: {error}")))?;
    let events = std::net::TcpListener::bind(config.events_addr)
        .map_err(|error| GatewayError::Transport(format!("event bind failed: {error}")))?;
    events
        .set_nonblocking(true)
        .map_err(|error| GatewayError::Transport(format!("event listener failed: {error}")))?;
    let addresses = BoundAddresses {
        grpc: grpc
            .local_addr()
            .map_err(|error| GatewayError::Transport(error.to_string()))?,
        events: events
            .local_addr()
            .map_err(|error| GatewayError::Transport(error.to_string()))?,
    };
    Ok((grpc, events, addresses))
}

async fn finish_task<E>(
    mut task: tokio::task::JoinHandle<Result<(), E>>,
    label: &str,
) -> Result<(), GatewayError>
where
    E: std::fmt::Display,
{
    if let Ok(result) = tokio::time::timeout(std::time::Duration::from_secs(6), &mut task).await {
        flatten_task_result(result, label)
    } else {
        task.abort();
        let _ = task.await;
        Err(GatewayError::Transport(format!(
            "{label} transport did not stop within the shutdown deadline"
        )))
    }
}

fn flatten_task_result<E>(
    result: Result<Result<(), E>, tokio::task::JoinError>,
    label: &str,
) -> Result<(), GatewayError>
where
    E: std::fmt::Display,
{
    result
        .map_err(|error| GatewayError::Transport(format!("{label} task failed: {error}")))?
        .map_err(|error| GatewayError::Transport(format!("{label} transport failed: {error}")))
}
