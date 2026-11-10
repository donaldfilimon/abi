//! `abi-model-serve`: load one exact, license-accepted local model and serve it
//! on loopback through the `OpenAI` chat-completions protocol.

use abi_agent_runtime::RunBudget;
use abi_model_runtime::{DevicePreference, LoadConfig, LocalModelProvider};
use abi_model_serve::{DEFAULT_PORT, ServeConfig, Server};
use abi_models::{AcceptanceLedger, ModelRegistry, StorageRoot};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Bearer token for `/v1/*`, read through the ABI environment registry.
const TOKEN_VAR: &str = "ABI_MODEL_SERVE_TOKEN";

const USAGE: &str = "usage: abi-model-serve --model <id> --principal <principal> --manifests <dir> --ledger <file> --device <cpu|metal|cuda> [--port <port>] [--max-output-tokens <n>]

Serve one exact local model on 127.0.0.1 through the OpenAI chat-completions
protocol (GET /health, GET /v1/models, POST /v1/chat/completions).

  --model       Manifest id to load. No other model is ever substituted.
  --principal   Principal whose license acceptance the ledger must hold.
  --manifests   Directory of model manifest *.json files.
  --ledger      License-acceptance ledger file.
  --device      Exact device. There is no automatic selection or CPU fallback.
  --port        Loopback port (default 8090).
  --max-output-tokens
                Hard generation ceiling per request (default 256).

Env:
  ABI_MODELS_DIR          Model storage root (default ~/.abi/models).
  ABI_MODEL_SERVE_TOKEN   Optional bearer token required on /v1/*.
";

struct Arguments {
    model: String,
    principal: String,
    manifests: PathBuf,
    ledger: PathBuf,
    device: DevicePreference,
    port: u16,
    max_output_tokens: Option<u32>,
}

fn parse(arguments: &[String]) -> Result<Arguments, String> {
    let mut model = None;
    let mut principal = None;
    let mut manifests = None;
    let mut ledger = None;
    let mut device = None;
    let mut port = DEFAULT_PORT;
    let mut max_output_tokens = None;
    let mut rest = arguments.iter();
    while let Some(flag) = rest.next() {
        let value = rest
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--model" => model = Some(value.clone()),
            "--principal" => principal = Some(value.clone()),
            "--manifests" => manifests = Some(PathBuf::from(value)),
            "--ledger" => ledger = Some(PathBuf::from(value)),
            "--device" => {
                device = Some(match value.as_str() {
                    "cpu" => DevicePreference::Cpu,
                    "metal" => DevicePreference::Metal,
                    "cuda" => DevicePreference::Cuda,
                    other => return Err(format!("unknown device '{other}'")),
                });
            }
            "--port" => {
                port = value
                    .parse()
                    .map_err(|_| format!("invalid port '{value}'"))?;
            }
            "--max-output-tokens" => {
                max_output_tokens = Some(
                    value
                        .parse::<u32>()
                        .ok()
                        .filter(|tokens| *tokens > 0)
                        .ok_or_else(|| format!("invalid token ceiling '{value}'"))?,
                );
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(Arguments {
        model: model.ok_or("--model is required")?,
        principal: principal.ok_or("--principal is required")?,
        manifests: manifests.ok_or("--manifests is required")?,
        ledger: ledger.ok_or("--ledger is required")?,
        device: device.ok_or("--device is required")?,
        port,
        max_output_tokens,
    })
}

fn run(arguments: &Arguments) -> Result<(), String> {
    let registry = ModelRegistry::from_dir(&arguments.manifests).map_err(|e| e.to_string())?;
    let ledger = AcceptanceLedger::load(&arguments.ledger).map_err(|e| e.to_string())?;
    let storage = StorageRoot::resolve().map_err(|e| e.to_string())?;
    let mut load = LoadConfig::new(arguments.device);
    if let Some(tokens) = arguments.max_output_tokens {
        load = load.with_max_generated_tokens(tokens);
    }
    let provider = LocalModelProvider::load(
        &registry,
        &ledger,
        &storage,
        &arguments.model,
        &arguments.principal,
        load,
    )
    .map_err(|e| e.to_string())?;
    let report = provider.load_report();
    eprintln!(
        "loaded model={} revision={} device={} tensors={} parameters={} artifact_bytes={}",
        report.model_id,
        report.revision,
        report.initialized_path.as_str(),
        report.tensor_count,
        report.parameter_count,
        report.artifact_bytes
    );

    let mut config = ServeConfig::new(&arguments.model).with_budget(RunBudget::unlimited());
    let authenticated = match abi_foundation::env::get(TOKEN_VAR) {
        Some(token) => {
            config = config.with_bearer_token(token);
            "bearer"
        }
        None => "off",
    };
    let server =
        Server::bind(arguments.port, Arc::new(provider), config).map_err(|e| e.to_string())?;
    let address = server.local_addr().map_err(|e| e.to_string())?;
    eprintln!("serving http://{address}/v1/chat/completions auth={authenticated}");

    let stop = Arc::new(AtomicBool::new(false));
    let handler_stop = Arc::clone(&stop);
    ctrlc::set_handler(move || handler_stop.store(true, Ordering::SeqCst))
        .map_err(|e| e.to_string())?;
    server.serve(&stop).map_err(|e| e.to_string())
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments
        .iter()
        .any(|flag| flag == "--help" || flag == "-h")
    {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let parsed = match parse(&arguments) {
        Ok(parsed) => parsed,
        Err(detail) => {
            eprint!("error: {detail}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&parsed) {
        Ok(()) => ExitCode::SUCCESS,
        Err(detail) => {
            eprintln!("error: {detail}");
            ExitCode::FAILURE
        }
    }
}
