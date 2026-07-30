//! Local OpenAI-compatible inference bridge.
//!
//! Ported from `src/connectors/local_bridge.zig`. Dispatches completions to a
//! user-run local server (llama-server, ollama, mlx-server, …) over loopback
//! HTTP. ABI does **not** embed an inference engine.
//!
//! Endpoints (env overrides):
//! - [`LLAMA_CPP_DEFAULT_ENDPOINT`] / `ABI_LLAMA_CPP_ENDPOINT`
//! - [`MLX_DEFAULT_ENDPOINT`] / `ABI_MLX_ENDPOINT`

use crate::connector::{ConnectorConfig, ConnectorError, Response, Result, TransportMode};
use crate::payload::build_openai_body;
use crate::transport::{Method, Transport, execute};
use abi_foundation::env::{self, LLAMA_CPP_ENDPOINT, MLX_ENDPOINT};
use serde_json::Value;

/// Default llama.cpp / ollama-style endpoint.
pub const LLAMA_CPP_DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8080";
/// Default MLX server endpoint.
pub const MLX_DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8081";

const LOCAL_PREFIXES: &[&str] = &[
    "llama-cpp/",
    "llama/",
    "ollama/",
    "ollama-",
    "lmstudio/",
    "vllm/",
    "mlx/",
    "mlx-",
];

/// Whether `model` should dispatch to the local bridge (not the persona router).
#[must_use]
pub fn is_local_bridge_model(model: &str) -> bool {
    LOCAL_PREFIXES.iter().any(|p| model.starts_with(p))
}

/// Resolve the bridge base URL for `model`.
///
/// `override_endpoint` wins when set. Otherwise `mlx/` / `mlx-` use the MLX
/// endpoint (env or default); other local models use the llama.cpp endpoint.
#[must_use]
pub fn endpoint_for(model: &str, override_endpoint: Option<&str>) -> String {
    if let Some(ep) = override_endpoint.filter(|s| !s.is_empty()) {
        return ep.to_string();
    }
    let is_mlx = model.starts_with("mlx/") || model.starts_with("mlx-");
    if is_mlx {
        env::get(MLX_ENDPOINT).unwrap_or_else(|| MLX_DEFAULT_ENDPOINT.to_string())
    } else {
        env::get(LLAMA_CPP_ENDPOINT).unwrap_or_else(|| LLAMA_CPP_DEFAULT_ENDPOINT.to_string())
    }
}

fn live_config(endpoint: &str, timeout_ms: u32) -> ConnectorConfig {
    ConnectorConfig {
        api_key: "local-bridge".into(), // some servers require a non-empty key header
        base_url: endpoint.trim_end_matches('/').to_string(),
        timeout_ms,
        transport: TransportMode::Live,
    }
}

/// `GET /health` on `endpoint`. Returns `false` when unreachable (not an error).
pub fn health_check(transport: &dyn Transport, endpoint: &str) -> bool {
    let config = live_config(endpoint, 3_000);
    let Ok(request) =
        crate::transport::build_request(&config, Method::Get, "/health", None, None, Vec::new())
    else {
        return false;
    };
    match execute(transport, &request) {
        Ok(response) => response.status == 200,
        Err(_) => false,
    }
}

/// Non-streaming OpenAI-compatible completion against the local bridge.
pub fn complete_live(
    transport: &dyn Transport,
    model: &str,
    input: &str,
    endpoint_override: Option<&str>,
) -> Result<Response> {
    let endpoint = endpoint_for(model, endpoint_override);
    let config = live_config(&endpoint, 30_000);
    let messages = build_user_messages(input);
    let body = build_openai_body(model, &messages, false)?;
    let request = crate::transport::build_request(
        &config,
        Method::Post,
        "/v1/chat/completions",
        Some("application/json"),
        Some(body),
        Vec::new(),
    )?;
    execute(transport, &request)
}

/// Streaming OpenAI-compatible completion; returns the raw SSE body.
pub fn complete_live_streaming(
    transport: &dyn Transport,
    model: &str,
    input: &str,
    endpoint_override: Option<&str>,
) -> Result<Response> {
    let endpoint = endpoint_for(model, endpoint_override);
    let config = live_config(&endpoint, 60_000);
    let messages = build_user_messages(input);
    let body = build_openai_body(model, &messages, true)?;
    let request = crate::transport::build_request(
        &config,
        Method::Post,
        "/v1/chat/completions",
        Some("application/json"),
        Some(body),
        Vec::new(),
    )?;
    execute(transport, &request)
}

fn build_user_messages(input: &str) -> String {
    let mut out = String::from(r#"[{"role":"user","content":"#);
    out.push_str(&abi_foundation::json::escape_json_string(input));
    out.push_str("}]");
    out
}

/// Extract `choices[0].message.content` from an `OpenAI` chat completion body.
pub fn extract_completion(body: &str) -> Result<String> {
    let parsed: Value = serde_json::from_str(body).map_err(|_| ConnectorError::InvalidResponse)?;
    let content = parsed
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
        .ok_or(ConnectorError::InvalidResponse)?;
    Ok(content.to_string())
}

/// Convenience: health-check via the default live HTTP transport.
#[cfg(feature = "live")]
#[must_use]
pub fn health_check_default(endpoint: &str) -> bool {
    health_check(&crate::transport::HttpTransport::new(), endpoint)
}

/// Convenience: complete via the default live HTTP transport.
#[cfg(feature = "live")]
pub fn complete_live_default(
    model: &str,
    input: &str,
    endpoint_override: Option<&str>,
) -> Result<Response> {
    complete_live(
        &crate::transport::HttpTransport::new(),
        model,
        input,
        endpoint_override,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::RecordingTransport;

    #[test]
    fn is_local_bridge_model_matches_known_prefixes() {
        assert!(is_local_bridge_model("llama-cpp/mistral"));
        assert!(is_local_bridge_model("llama/llama3"));
        assert!(is_local_bridge_model("ollama/qwen2"));
        assert!(is_local_bridge_model("ollama-qwen2"));
        assert!(is_local_bridge_model("lmstudio/phi3"));
        assert!(is_local_bridge_model("vllm/mixtral"));
        assert!(is_local_bridge_model("mlx/gemma"));
        assert!(is_local_bridge_model("mlx-gemma"));
        assert!(!is_local_bridge_model("claude-fable-5"));
        assert!(!is_local_bridge_model("abi-local"));
        assert!(!is_local_bridge_model("gpt-4"));
        assert!(!is_local_bridge_model("fable-5"));
    }

    #[test]
    fn endpoint_for_routes_mlx_and_override() {
        assert_eq!(
            endpoint_for("llama-cpp/mistral", None),
            LLAMA_CPP_DEFAULT_ENDPOINT
        );
        assert_eq!(endpoint_for("mlx/gemma", None), MLX_DEFAULT_ENDPOINT);
        assert_eq!(endpoint_for("mlx-gemma", None), MLX_DEFAULT_ENDPOINT);
        assert_eq!(
            endpoint_for("ollama/qwen2", None),
            LLAMA_CPP_DEFAULT_ENDPOINT
        );
        assert_eq!(
            endpoint_for("mlx/gemma", Some("http://localhost:9999")),
            "http://localhost:9999"
        );
    }

    #[test]
    fn extract_completion_parses_openai_response() {
        let body = r#"{"id":"chatcmpl-1","choices":[{"index":0,"message":{"role":"assistant","content":"Hello world"},"finish_reason":"stop"}]}"#;
        assert_eq!(extract_completion(body).unwrap(), "Hello world");
        assert!(extract_completion(r#"{"choices":[]}"#).is_err());
        assert!(extract_completion("not json").is_err());
    }

    #[test]
    fn complete_live_builds_openai_request_against_override() {
        let transport = RecordingTransport::new().with_response(Ok(Response::ok(
            r#"{"choices":[{"message":{"content":"pong"}}]}"#.to_string(),
        )));
        let response = complete_live(&transport, "llama/phi3", "ping", Some("http://127.0.0.1:1"))
            .expect("synthetic transport");
        assert_eq!(response.status, 200);
        let requests = transport.requests();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].url.contains("127.0.0.1:1"));
        assert!(requests[0].url.ends_with("/v1/chat/completions"));
        let body = requests[0].body.as_deref().unwrap_or("");
        assert!(body.contains("\"model\":\"llama/phi3\""));
        assert!(body.contains("ping"));
    }

    #[test]
    fn health_check_false_on_transport_error() {
        use crate::transport::UnavailableTransport;
        assert!(!health_check(&UnavailableTransport, "http://127.0.0.1:1"));
    }
}
