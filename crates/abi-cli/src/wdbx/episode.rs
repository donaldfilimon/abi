//! WDBX `episode` subcommand: the first caller of the gateway's canonical
//! episode gate (`ProposeEpisodeWrite` / `VerifyEpisode`, WDBX v3).
//!
//! The CLI is a thin gRPC client. It never opens the store itself: the write
//! travels as the JSON the operator supplied, the gateway decodes and gates
//! it, and rejections come back as gRPC statuses whose message is the store's
//! stable reason label (for example `episode_replay`).

use std::path::{Path, PathBuf};
use std::time::Duration;

use abi_wdbx_gateway::proto::wdbx_gateway_client::WdbxGatewayClient;
use abi_wdbx_gateway::proto::{EpisodeReceipt, ProposeEpisodeWriteRequest, VerifyEpisodeRequest};
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};
use tonic::{Request, Status};

use crate::app::Outcome;
use crate::usage::is_help_token;

pub(crate) const EPISODE_HELP: &str = "usage: abi wdbx episode propose <write.json> [--preview] [options]\n       abi wdbx episode verify <guild_ref> <digest-hex> [options]\n\nCall the gateway's canonical episode gate (WDBX v3). `propose` sends one\nEpisodeWrite as JSON (unknown fields are rejected by the gateway) and prints\nthe commitment; `--preview` computes the commitment without appending.\n`verify` asks whether a commitment exists in the guild's ledger (the gateway\nanswers from the first 2048 receipts of that guild in ledger order).\n\nOptions\n  --endpoint <URL>       Gateway gRPC endpoint (default http://127.0.0.1:50051;\n                         env ABI_WDBX_GATEWAY_ENDPOINT). Plain http is accepted\n                         for loopback only; other hosts need https + --ca-cert\n  --token-file <PATH>    Bearer token file, the same file the gateway was given\n                         (env ABI_WDBX_GATEWAY_TOKEN_FILE; required)\n  --ca-cert <PEM>        Trust this CA for an https endpoint (server TLS only;\n                         an mTLS client identity is not wired yet)\n  --json                 Print the result as one JSON object\n\nExit status: 0 on append, preview, or found; 1 on a gateway rejection (stderr\ncarries the gRPC code and the store's reason label) or when verify finds\nnothing; 2 on usage errors.\n";

const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:50051";
const ENDPOINT_ENV: &str = "ABI_WDBX_GATEWAY_ENDPOINT";
const TOKEN_FILE_ENV: &str = "ABI_WDBX_GATEWAY_TOKEN_FILE";
const MAX_WRITE_BYTES: u64 = 64 * 1024;
const MAX_TOKEN_BYTES: u64 = 64 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

struct Options {
    endpoint: String,
    token_file: Option<PathBuf>,
    ca_cert: Option<PathBuf>,
    json: bool,
    preview: bool,
    positional: Vec<String>,
}

fn usage(exit_code: u8) -> Outcome {
    Outcome::stderr(EPISODE_HELP.to_owned(), exit_code)
}

fn failure(context: &str, detail: impl std::fmt::Display) -> Outcome {
    Outcome::stderr(format!("episode {context}: {detail}\n"), 1)
}

fn parse(args: &[String]) -> Result<Options, Outcome> {
    let mut options = Options {
        endpoint: std::env::var(ENDPOINT_ENV).unwrap_or_else(|_| DEFAULT_ENDPOINT.to_owned()),
        token_file: std::env::var_os(TOKEN_FILE_ENV).map(PathBuf::from),
        ca_cert: None,
        json: false,
        preview: false,
        positional: Vec::new(),
    };
    let mut iter = args.iter();
    while let Some(argument) = iter.next() {
        match argument.as_str() {
            "--json" => options.json = true,
            "--preview" => options.preview = true,
            "--endpoint" => options
                .endpoint
                .clone_from(iter.next().ok_or_else(|| usage(2))?),
            "--token-file" => {
                options.token_file = Some(PathBuf::from(iter.next().ok_or_else(|| usage(2))?));
            }
            "--ca-cert" => {
                options.ca_cert = Some(PathBuf::from(iter.next().ok_or_else(|| usage(2))?));
            }
            other if is_help_token(other) => return Err(usage(0)),
            other if other.starts_with('-') => return Err(usage(2)),
            other => options.positional.push(other.to_owned()),
        }
    }
    Ok(options)
}

/// Dispatch arguments following `wdbx episode`.
pub(crate) fn run_episode(args: &[String]) -> Outcome {
    let Some(action) = args.first() else {
        return usage(2);
    };
    if is_help_token(action) {
        return usage(0);
    }
    let options = match parse(&args[1..]) {
        Ok(options) => options,
        Err(outcome) => return outcome,
    };
    match action.as_str() {
        "propose" => propose(&options),
        "verify" => verify(&options),
        _ => usage(2),
    }
}

fn propose(options: &Options) -> Outcome {
    let [write_path] = options.positional.as_slice() else {
        return usage(2);
    };
    let episode_write_json = match read_bounded(Path::new(write_path), MAX_WRITE_BYTES) {
        Ok(bytes) => bytes,
        Err(detail) => return failure("propose", format!("write file {write_path}: {detail}")),
    };
    let preview_only = options.preview;
    let result = call(options, move |mut client, token| async move {
        client
            .propose_episode_write(authenticated(
                ProposeEpisodeWriteRequest {
                    episode_write_json,
                    preview_only,
                },
                &token,
            )?)
            .await
            .map(tonic::Response::into_inner)
    });
    match result {
        Ok(response) => {
            let mut fields = vec![
                ("decision", response.decision.clone()),
                ("episode_digest", hex(&response.episode_digest)),
            ];
            if let Some(receipt) = &response.receipt {
                fields.extend(receipt_fields(receipt));
            }
            render(options.json, &fields)
        }
        Err(detail) => failure("propose", detail),
    }
}

fn verify(options: &Options) -> Outcome {
    let [guild_ref, digest_hex] = options.positional.as_slice() else {
        return usage(2);
    };
    let Some(episode_digest) = unhex(digest_hex) else {
        return failure("verify", "digest must be 64 hexadecimal characters");
    };
    let guild_ref = guild_ref.clone();
    let printed_guild = guild_ref.clone();
    let result = call(options, move |mut client, token| async move {
        client
            .verify_episode(authenticated(
                VerifyEpisodeRequest {
                    guild_ref,
                    episode_digest,
                },
                &token,
            )?)
            .await
            .map(tonic::Response::into_inner)
    });
    match result {
        Ok(response) => {
            let mut fields = vec![("found", response.found.to_string())];
            if let Some(receipt) = &response.receipt {
                fields.extend(receipt_fields(receipt));
            }
            let mut outcome = render(options.json, &fields);
            if !response.found {
                outcome.exit_code = 1;
                outcome.stderr = format!(
                    "episode verify: commitment {digest_hex} not found in the first 2048 receipts of guild {printed_guild}\n"
                );
            }
            outcome
        }
        Err(detail) => failure("verify", detail),
    }
}

/// Connect, run one RPC on a current-thread runtime, and flatten every failure
/// into one operator-facing line.
fn call<F, Fut, T>(options: &Options, rpc: F) -> Result<T, String>
where
    F: FnOnce(WdbxGatewayClient<Channel>, String) -> Fut,
    Fut: std::future::Future<Output = Result<T, Status>>,
{
    check_endpoint_transport(&options.endpoint, options.ca_cert.as_deref())?;
    let token = load_token(options.token_file.as_deref())?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| format!("runtime: {error}"))?;
    runtime.block_on(async {
        let client = connect(options).await?;
        rpc(client, token)
            .await
            .map_err(|status| format!("{:?}: {}", status.code(), status.message()))
    })
}

/// Mirror the gateway's own listener rule: a bearer token only travels in
/// cleartext to loopback. Any other host needs `https` and a CA to trust.
fn check_endpoint_transport(endpoint: &str, ca_cert: Option<&Path>) -> Result<(), String> {
    let (scheme, rest) = endpoint
        .split_once("://")
        .ok_or_else(|| format!("endpoint {endpoint}: missing scheme"))?;
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = authority.strip_prefix('[').map_or_else(
        || {
            authority
                .rsplit_once(':')
                .map_or(authority, |(host, _)| host)
        },
        |bracketed| bracketed.split(']').next().unwrap_or(""),
    );
    let loopback = matches!(host, "127.0.0.1" | "::1" | "localhost");
    match scheme {
        "http" if loopback => Ok(()),
        "http" => Err(format!(
            "endpoint {endpoint}: non-loopback endpoints require https and --ca-cert (a bearer token must not travel in cleartext)"
        )),
        "https" if loopback || ca_cert.is_some() => Ok(()),
        "https" => Err(format!(
            "endpoint {endpoint}: --ca-cert is required for a non-loopback https endpoint"
        )),
        other => Err(format!("endpoint {endpoint}: unsupported scheme {other}")),
    }
}

async fn connect(options: &Options) -> Result<WdbxGatewayClient<Channel>, String> {
    let mut endpoint = Endpoint::from_shared(options.endpoint.clone())
        .map_err(|error| format!("endpoint {}: {error}", options.endpoint))?
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT);
    if let Some(ca_cert) = &options.ca_cert {
        let pem = std::fs::read(ca_cert)
            .map_err(|error| format!("CA certificate {}: {error}", ca_cert.display()))?;
        endpoint = endpoint
            .tls_config(ClientTlsConfig::new().ca_certificate(Certificate::from_pem(pem)))
            .map_err(|error| format!("TLS configuration: {error}"))?;
    }
    let channel = endpoint
        .connect()
        .await
        .map_err(|error| format!("connect {}: {error}", options.endpoint))?;
    Ok(WdbxGatewayClient::new(channel))
}

fn authenticated<T>(message: T, token: &str) -> Result<Request<T>, Status> {
    let mut request = Request::new(message);
    let value = format!("Bearer {token}")
        .parse()
        .map_err(|_| Status::invalid_argument("bearer token must be visible ASCII"))?;
    request.metadata_mut().insert("authorization", value);
    Ok(request)
}

fn load_token(path: Option<&Path>) -> Result<String, String> {
    let path = path.ok_or_else(|| {
        format!("bearer token file is required (--token-file or {TOKEN_FILE_ENV})")
    })?;
    let bytes = read_bounded(path, MAX_TOKEN_BYTES)
        .map_err(|detail| format!("token file {}: {detail}", path.display()))?;
    let token = String::from_utf8(bytes)
        .map_err(|_| format!("token file {}: must be UTF-8", path.display()))?;
    let token = token.trim_end().to_owned();
    if token.is_empty() {
        return Err(format!("token file {}: is empty", path.display()));
    }
    Ok(token)
}

fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>, String> {
    use std::io::Read as _;

    let file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    file.take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.is_empty() {
        return Err("is empty".into());
    }
    if u64::try_from(bytes.len()).is_ok_and(|length| length > maximum) {
        return Err(format!("exceeds {maximum} bytes"));
    }
    Ok(bytes)
}

fn receipt_fields(receipt: &EpisodeReceipt) -> Vec<(&'static str, String)> {
    vec![
        ("sequence", receipt.sequence.to_string()),
        ("request_id", receipt.request_id.clone()),
        ("operation_id", receipt.operation_id.clone()),
        ("guild_ref", receipt.guild_ref.clone()),
        ("event_kind", receipt.event_kind.clone()),
        ("policy_version", receipt.policy_version.clone()),
        ("evidence_level", receipt.evidence_level.clone()),
        (
            "previous_digest",
            if receipt.previous_digest.is_empty() {
                "none".to_owned()
            } else {
                hex(&receipt.previous_digest)
            },
        ),
        (
            "terminal_status",
            if receipt.terminal_status.is_empty() {
                "none".to_owned()
            } else {
                receipt.terminal_status.clone()
            },
        ),
        ("redacted", receipt.redacted.to_string()),
    ]
}

fn render(json: bool, fields: &[(&'static str, String)]) -> Outcome {
    let stdout = if json {
        let object: serde_json::Map<String, serde_json::Value> = fields
            .iter()
            .map(|(key, value)| ((*key).to_owned(), serde_json::Value::String(value.clone())))
            .collect();
        format!("{}\n", serde_json::Value::Object(object))
    } else {
        let mut line = fields
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(" ");
        line.push('\n');
        line
    };
    Outcome {
        stdout,
        stderr: String::new(),
        exit_code: 0,
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    if text.len() != 64 || !text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&text[index..index + 2], 16).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trips_and_rejects_bad_lengths() {
        let digest = [0xabu8; 32];
        let text = hex(&digest);
        assert_eq!(text.len(), 64);
        assert_eq!(unhex(&text).unwrap(), digest.to_vec());
        assert!(unhex("abc").is_none());
        assert!(unhex(&"zz".repeat(32)).is_none());
    }

    #[test]
    fn help_and_usage_paths_do_not_touch_the_network() {
        assert_eq!(run_episode(&[]).exit_code, 2);
        assert_eq!(run_episode(&["--help".to_owned()]).exit_code, 0);
        assert_eq!(run_episode(&["nonsense".to_owned()]).exit_code, 2);
        let absent = std::env::temp_dir()
            .join(format!("abi-episode-no-token-{}", std::process::id()))
            .to_string_lossy()
            .into_owned();
        let absent_token = run_episode(&[
            "verify".to_owned(),
            "guild".to_owned(),
            "00".repeat(32),
            "--token-file".to_owned(),
            absent.clone(),
        ]);
        assert_eq!(absent_token.exit_code, 1);
        assert!(
            absent_token.stderr.contains("token file"),
            "{}",
            absent_token.stderr
        );
        let cleartext = run_episode(&[
            "verify".to_owned(),
            "guild".to_owned(),
            "00".repeat(32),
            "--endpoint".to_owned(),
            "http://10.0.0.1:1".to_owned(),
            "--token-file".to_owned(),
            absent,
        ]);
        assert_eq!(cleartext.exit_code, 1);
        assert!(
            cleartext.stderr.contains("require https and --ca-cert"),
            "{}",
            cleartext.stderr
        );
        let bad_digest = run_episode(&["verify".to_owned(), "guild".to_owned(), "nope".to_owned()]);
        assert_eq!(bad_digest.exit_code, 1);
        assert!(bad_digest.stderr.contains("64 hexadecimal"));
    }

    #[test]
    fn transport_rule_mirrors_the_gateway_listener() {
        assert!(check_endpoint_transport("http://127.0.0.1:50051", None).is_ok());
        assert!(check_endpoint_transport("http://localhost:50051", None).is_ok());
        assert!(check_endpoint_transport("http://[::1]:50051", None).is_ok());
        assert!(check_endpoint_transport("http://gateway.internal:50051", None).is_err());
        assert!(check_endpoint_transport("https://gateway.internal:50051", None).is_err());
        assert!(
            check_endpoint_transport("https://gateway.internal:50051", Some(Path::new("ca.pem")))
                .is_ok()
        );
        assert!(check_endpoint_transport("gateway.internal:50051", None).is_err());
        assert!(check_endpoint_transport("ftp://127.0.0.1:1", None).is_err());
    }
}
