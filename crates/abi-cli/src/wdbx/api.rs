//! WDBX `api` subcommand: loopback REST server with bearer-token auth.
//!
//! Split from the flat `wdbx` CLI module; dispatch lives in `super::run`.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::app::Outcome;
use abi_wdbx::{DurableStore, RestConfig, RestServer, StorePaths};

pub(crate) const API_HELP: &str = "usage: abi wdbx api serve [port]\n\nServe the loopback WDBX REST API.\n\nEnv:\n  ABI_WDBX_REST_TOKEN     Optional bearer token for request auth.\n  ABI_WDBX_TLS_CERT       Path to PEM certificate (TLS config / proxy deployment).\n  ABI_WDBX_TLS_KEY        Path to PEM private key (TLS config / proxy deployment).\n\nTLS: native termination is not linked; deploy behind nginx/Caddy/haproxy.\n";

fn open_default_store() -> Result<DurableStore, String> {
    if let Ok(path) = std::env::var("ABI_WDBX_PATH") {
        if path == ":memory:" {
            return Err(
                "ABI_WDBX_PATH=:memory: is not valid for durable REST/cluster serving".into(),
            );
        }
        return DurableStore::open(StorePaths::new(path)).map_err(|e| e.to_string());
    }
    if matches!(
        std::env::var("ABI_WDBX_PERSIST").as_deref(),
        Ok("0" | "false" | "no" | "off")
    ) {
        return Err("WDBX persistence disabled (ABI_WDBX_PERSIST=0)".into());
    }
    // Shared with `util::open_store_result` so the lib-test build's refusal to
    // touch the operator's live `~/.abi/` store covers this path too.
    let home = crate::util::default_store_home().ok_or_else(|| "HOME is unset".to_string())?;
    DurableStore::open(StorePaths::new(format!("{home}/.abi/wdbx"))).map_err(|e| e.to_string())
}

fn api_serve(port_raw: Option<&str>) -> Outcome {
    let port: u16 = match port_raw {
        None => 8081,
        Some(raw) => match raw.parse() {
            Ok(p) => p,
            Err(_) => return super::usage(),
        },
    };
    let store = match open_default_store() {
        Ok(store) => store,
        Err(detail) => return super::error("api serve failed", detail),
    };
    // Disclose TLS env presence without claiming native TLS termination.
    if let Some(tls) = abi_wdbx::TlsConfig::from_env() {
        let _ = tls;
        eprintln!(
            "note: ABI_WDBX_TLS_CERT/KEY present — native TLS termination is not linked; deploy behind nginx/Caddy/haproxy"
        );
    }
    let config = RestConfig::from_env();
    let auth = if config.bearer_token.is_some() {
        "auth=bearer"
    } else {
        "auth=off"
    };
    let mut server = match RestServer::bind(port, store, config) {
        Ok(server) => server,
        Err(detail) => return super::error("api serve failed", detail),
    };
    let bound = match server.local_port() {
        Ok(p) => p,
        Err(detail) => return super::error("api serve failed", detail),
    };
    eprintln!(
        "wdbx REST serving on 127.0.0.1:{bound} ({auth}); routes: POST /insert /query /verify, GET /health /stats; loopback only"
    );

    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let stop_flag = std::sync::Arc::clone(&stop);
    let _ = ctrlc::set_handler(move || {
        stop_flag.store(true, Ordering::SeqCst);
    });

    while !stop.load(Ordering::SeqCst) {
        if let Err(err) = server.serve_one() {
            if stop.load(Ordering::SeqCst) {
                break;
            }
            eprintln!("wdbx REST serve error: {err}");
        }
    }
    Outcome::stderr("wdbx REST stopped\n".into(), 0)
}

pub(crate) fn run_api(args: &[String]) -> Outcome {
    match args {
        [operation] if operation == "serve" => api_serve(None),
        [operation, port] if operation == "serve" => api_serve(Some(port)),
        _ => super::usage(),
    }
}
