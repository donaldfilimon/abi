//! `abi complete` — local completion, optional SEA `--learn`, neural demo,
//! soul-network blend, and explicit live Anthropic transport.
//!
//! Ported from the local and live paths of `src/cli/handlers/complete_handlers.zig`.
//! `--live` is Anthropic-only for HTTP providers. Apple `FoundationModels`
//! (`apple-fm` + `--confirm`) uses the Swift `@c` shim when linked on arm64
//! macOS; otherwise it reports honest unavailability.
//! `--soul <file.json>` loads a `SoulLayout`, bootstraps a `[3, 8, 3]` point net,
//! and blends with keyword routing via `--soul-alpha` (default 0.5).

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;

use abi_ai::{PointNeuralNetwork, SoulLayout, complete, models, route_with_soul};
use abi_connectors::{
    Client, ConnectorConfig, DefaultTransport, Provider, fm_available, fm_bridge_linked,
    fm_complete_live, fm_config, is_local_bridge_model, local_bridge_complete,
    local_bridge_complete_stream, local_bridge_endpoint, local_bridge_extract, local_bridge_health,
    parse_stream,
};
use abi_foundation::credentials::{self, CredentialField};
use abi_sea::{LearnLoopConfig, run_learn_loop};
use abi_wdbx::RecordId;

use crate::app::Outcome;
use crate::util;

const USAGE: &str = "usage: abi complete [--live] [--stream] [--learn] [--neural] [--soul <file>] [--soul-alpha <0..1>] [--model <id>] [--confirm] [--] <input>";

/// Fields for the one-line-per-field completion metadata block.
struct MetaReport<'a> {
    requested_model: &'a str,
    profile: &'a str,
    audit_passed: bool,
    escore: f32,
    audit_vetoed: bool,
    persisted: bool,
    learn: Option<(usize, bool)>,
    kv: usize,
    vectors: usize,
    blocks: usize,
    query_id: Option<RecordId>,
    response_id: Option<RecordId>,
    block_hex: Option<&'a str>,
    wdbx_status: Option<&'a str>,
    output: &'a str,
}

fn render_local(report: &MetaReport<'_>) -> String {
    let mut out = String::new();
    if let Some((evidence_count, adapted)) = report.learn {
        let _ = writeln!(
            out,
            "requested_model={} provider=local transport=in-process generation_engine=persona-template policy_scope=lexical-signal profile={} audit_passed={} audit_escore={:.3} audit_vetoed={} persisted={} learn=true evidence_count={evidence_count} adapted={adapted}",
            report.requested_model,
            report.profile,
            report.audit_passed,
            report.escore,
            report.audit_vetoed,
            report.persisted,
        );
    } else {
        let _ = writeln!(
            out,
            "requested_model={} provider=local transport=in-process generation_engine=persona-template policy_scope=lexical-signal profile={} audit_passed={} audit_escore={:.3} audit_vetoed={} persisted={}",
            report.requested_model,
            report.profile,
            report.audit_passed,
            report.escore,
            report.audit_vetoed,
            report.persisted,
        );
    }
    let _ = writeln!(
        out,
        "wdbx kv_entries={} vectors={} blocks={}",
        report.kv, report.vectors, report.blocks
    );
    if let Some(qid) = report.query_id {
        let _ = writeln!(out, "query_vector_id={qid}");
        let _ = writeln!(out, "metadata_key=completion:{qid}");
    }
    if let Some(rid) = report.response_id {
        let _ = writeln!(out, "response_vector_id={rid}");
    }
    if let Some(hex) = report.block_hex {
        let _ = writeln!(out, "block_id={hex}");
    }
    if let Some(status) = report.wdbx_status {
        let _ = writeln!(out, "wdbx_status={status}");
    }
    let _ = writeln!(out, "{}", report.output);
    out
}

fn run_local(input: &str, model: &str) -> Outcome {
    let Ok(result) = complete(input, model) else {
        return Outcome::stderr("error: completion input must not be empty\n".to_owned(), 1);
    };

    let mut store = util::open_store();
    let (persisted, qid, rid, hex, kv, vectors, blocks, status) =
        if let Some(store) = store.as_mut() {
            let before = store.stats();
            let query = abi_ai::text_embedding(input);
            let response = abi_ai::text_embedding(&result.output);
            let query_id = store.put_vector(&query).ok();
            let response_id = store.put_vector(&response).ok();
            let persisted_record = match (query_id, response_id) {
                (Some(q), Some(r)) => {
                    let metadata = abi_ai::completion::metadata_json(input, &result, q, r);
                    let key = abi_ai::completion::metadata_key(q);
                    store.put(&key, &metadata).ok().and_then(|_| {
                        store
                            .add_block(
                                result.selected_profile.label(),
                                q,
                                r,
                                &metadata,
                                abi_foundation::time::unix_ms(),
                            )
                            .ok()
                            .map(|block| (q, r, block.hash))
                    })
                }
                _ => None,
            };
            let after = store.stats();
            let persisted = persisted_record.is_some();
            let (qid, rid, hex) = persisted_record.map_or((None, None, None), |(q, r, hash)| {
                (Some(q), Some(r), Some(hash))
            });
            (
                persisted,
                qid,
                rid,
                hex,
                after.kv_entries.saturating_sub(before.kv_entries),
                after.vectors.saturating_sub(before.vectors),
                after.blocks.saturating_sub(before.blocks),
                if persisted {
                    None
                } else {
                    Some("wdbx write failed")
                },
            )
        } else {
            (
                false,
                None,
                None,
                None,
                0,
                0,
                0,
                Some("no persistent WDBX path configured"),
            )
        };

    let text = render_local(&MetaReport {
        requested_model: &result.model,
        profile: result.selected_profile.label(),
        audit_passed: result.audit.passed,
        escore: result.audit.escore,
        audit_vetoed: result.audit.vetoed,
        persisted,
        learn: None,
        kv,
        vectors,
        blocks,
        query_id: qid,
        response_id: rid,
        block_hex: hex.as_deref(),
        wdbx_status: status,
        output: &result.output,
    });
    Outcome {
        stdout: text,
        stderr: String::new(),
        exit_code: 0,
    }
}

fn run_learn(input: &str, model: &str) -> Outcome {
    let Some(mut store) = util::open_store() else {
        return run_local(input, model);
    };
    let before = store.stats();
    let Ok(learned) = run_learn_loop(
        &mut store,
        input,
        model,
        LearnLoopConfig::default(),
        abi_foundation::time::unix_ms(),
    ) else {
        return Outcome::stderr("error: completion input must not be empty\n".to_owned(), 1);
    };
    let after = store.stats();
    let result = &learned.completion;
    let persisted = learned.persisted.is_some();
    let text = render_local(&MetaReport {
        requested_model: &result.model,
        profile: result.selected_profile.label(),
        audit_passed: result.audit.passed,
        escore: result.audit.escore,
        audit_vetoed: result.audit.vetoed,
        persisted,
        learn: Some((learned.evidence_count, learned.adapted)),
        kv: after.kv_entries.saturating_sub(before.kv_entries),
        vectors: after.vectors.saturating_sub(before.vectors),
        blocks: after.blocks.saturating_sub(before.blocks),
        query_id: learned.persisted.as_ref().map(|p| p.query_vector_id),
        response_id: learned.persisted.as_ref().map(|p| p.response_vector_id),
        block_hex: learned
            .persisted
            .as_ref()
            .map(|p| p.block_hash_hex.as_str()),
        wdbx_status: if persisted {
            None
        } else {
            Some("wdbx write failed")
        },
        output: &result.output,
    });
    Outcome {
        stdout: text,
        stderr: String::new(),
        exit_code: 0,
    }
}

fn run_neural(input: &str) -> Outcome {
    // In-process char-LM demo via abi-nn — not a production LLM.
    let corpus = format!("{input} {input} ");
    let model = match abi_nn::train_model(
        corpus.as_bytes(),
        abi_nn::TrainConfig {
            epochs: 80,
            lr: 0.3,
            seed: 7,
            ..abi_nn::TrainConfig::default()
        },
    ) {
        Ok(model) => model,
        Err(err) => {
            return Outcome::stderr(format!("error: neural train failed: {err}\n"), 1);
        }
    };
    let seed = input.as_bytes().first().copied().unwrap_or(b'h');
    let sampled = abi_nn::sample(&model, seed, 48);
    let text = format!(
        "[model=nn-char-lm | neural=true | stream=false | note=in-process character-level demo model — not a production LLM]\n{}\nnn sample: {}\n",
        abi_nn::format_report(&model.report),
        String::from_utf8_lossy(&sampled),
    );
    Outcome {
        stdout: text,
        stderr: String::new(),
        exit_code: 0,
    }
}

/// On-device Apple `FoundationModels` path (`--live --model apple-fm --confirm`).
///
/// Links the Swift `@c` shim (`libabi_fm_shim.dylib`) on arm64 macOS when the
/// `foundationmodels` feature is enabled. Returns an honest unavailability error
/// when the bridge is not linked or the on-device model is not ready — never a
/// fabricated reply.
fn run_fm_complete(input: &str, model: &str, confirmed: bool) -> Outcome {
    if !confirmed {
        return Outcome::stderr(
            "error: on-device apple-fm requires --confirm (e.g. `abi complete --live --model apple-fm --confirm <input>`)\n".into(),
            2,
        );
    }
    if input.trim().is_empty() {
        return Outcome::stderr("error: completion input must not be empty\n".into(), 1);
    }
    if !fm_bridge_linked() || !fm_available() {
        return Outcome::stderr(
            format!(
                "error: on-device FoundationModels unavailable for model={model}: not built with FoundationModels FFI, not running on arm64 macOS, or the on-device runtime is not reachable on this host\n"
            ),
            1,
        );
    }
    match fm_complete_live(input, fm_config()) {
        Ok(resp) => Outcome {
            stdout: format!(
                "{}\n[model={model} | provider=foundationmodels | transport=on-device | status={}]\n",
                resp.body.trim_end(),
                resp.status
            ),
            stderr: String::new(),
            exit_code: 0,
        },
        Err(err) => Outcome::stderr(
            format!("error: on-device FoundationModels request failed: {err}\n"),
            1,
        ),
    }
}

/// Stage 2: live Anthropic path behind `--live` (Zig parity).
///
/// Only anthropic-provider models are supported; the API key is read from the
/// credential store and the request crosses the explicit live transport boundary.
fn run_live_complete(input: &str, model: &str, stream: bool) -> Outcome {
    if models::provider_of(model) != models::Provider::Anthropic {
        return Outcome::stderr(
            "error: --live currently supports anthropic models only (e.g. --model fable-5)\n"
                .into(),
            2,
        );
    }

    let creds = match credentials::load() {
        Ok(c) => c,
        Err(err) => {
            return Outcome::stderr(
                format!(
                    "error: failed to load credentials ({err}); run `abi auth signin anthropic`\n"
                ),
                2,
            );
        }
    };
    let Some(secret) = creds.get(CredentialField::ANTHROPIC_API_KEY) else {
        return Outcome::stderr(
            "error: no anthropic credentials configured; run `abi auth signin anthropic`\n".into(),
            2,
        );
    };

    let config = ConnectorConfig::new(secret.expose(), "https://api.anthropic.com").live();
    let client = Client::new(Provider::Anthropic, config);
    let transport = DefaultTransport::new();

    if stream {
        let resp = match client.complete_stream_live(&transport, model, input) {
            Ok(r) => r,
            Err(err) => {
                return Outcome::stderr(format!("error: anthropic live stream failed: {err}\n"), 1);
            }
        };
        let mut stdout = std::io::stdout();
        let mut captured = String::new();
        let full = match parse_stream(&resp.body, |chunk| {
            if !chunk.delta.is_empty() {
                let _ = write!(stdout, "{}", chunk.delta);
                let _ = stdout.flush();
                captured.push_str(&chunk.delta);
            }
            Ok(())
        }) {
            Ok(full) => full,
            Err(err) => {
                return Outcome::stderr(
                    format!("error: anthropic live stream parse failed: {err}\n"),
                    1,
                );
            }
        };
        let _ = writeln!(stdout);
        let footer = format!("model={model} provider=anthropic transport=live stream=sse\n");
        let _ = write!(stdout, "{footer}");
        let _ = stdout.flush();
        let mut out = if full.is_empty() { captured } else { full };
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&footer);
        return Outcome {
            stdout: out,
            stderr: String::new(),
            exit_code: 0,
        };
    }

    let resp = match client.complete_live(&transport, model, input) {
        Ok(r) => r,
        Err(err) => {
            return Outcome::stderr(format!("error: anthropic live request failed: {err}\n"), 1);
        }
    };
    let ok = (200..300).contains(&resp.status);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "model={model} provider=anthropic transport=live status={}",
        resp.status
    );
    let _ = writeln!(out, "{}", resp.body);
    Outcome {
        stdout: out,
        stderr: String::new(),
        exit_code: u8::from(!ok),
    }
}

fn run_local_bridge(input: &str, model: &str, stream: bool) -> Outcome {
    let transport = DefaultTransport::new();
    let endpoint = local_bridge_endpoint(model, None);
    if !local_bridge_health(&transport, &endpoint) {
        eprintln!(
            "warning: local inference server not reachable at {endpoint}; falling back to in-process router"
        );
        return run_local(input, model);
    }
    if stream {
        let resp = match local_bridge_complete_stream(&transport, model, input, None) {
            Ok(r) => r,
            Err(err) => {
                eprintln!(
                    "warning: local bridge stream failed ({err}); falling back to in-process router"
                );
                return run_local(input, model);
            }
        };
        let mut stdout = std::io::stdout();
        let mut full = String::new();
        if let Err(err) = parse_stream(&resp.body, |chunk| {
            if !chunk.delta.is_empty() {
                let _ = write!(stdout, "{}", chunk.delta);
                let _ = stdout.flush();
                full.push_str(&chunk.delta);
            }
            Ok(())
        }) {
            eprintln!(
                "warning: local bridge stream parse failed ({err}); falling back to in-process router"
            );
            return run_local(input, model);
        }
        let _ = writeln!(stdout);
        let footer = format!("[model={model} | bridge={endpoint} | local=true | stream=sse]\n");
        let _ = write!(stdout, "{footer}");
        if !full.ends_with('\n') {
            full.push('\n');
        }
        full.push_str(&footer);
        return Outcome {
            stdout: full,
            stderr: String::new(),
            exit_code: 0,
        };
    }
    let resp = match local_bridge_complete(&transport, model, input, None) {
        Ok(r) => r,
        Err(err) => {
            // Health can be a false positive (e.g. unrelated /health on :8080).
            eprintln!(
                "warning: local bridge request failed ({err}); falling back to in-process router"
            );
            return run_local(input, model);
        }
    };
    let text = match local_bridge_extract(&resp.body) {
        Ok(t) => t,
        Err(err) => {
            eprintln!(
                "warning: local bridge response unusable ({err}); falling back to in-process router"
            );
            return run_local(input, model);
        }
    };
    Outcome {
        stdout: format!("{text}\n[model={model} | bridge={endpoint} | local=true]\n"),
        stderr: String::new(),
        exit_code: 0,
    }
}

fn run_soul(input: &str, soul_path: &str, blend_alpha: f32) -> Outcome {
    if input.trim().is_empty() {
        return Outcome::stderr("error: completion input must not be empty\n".into(), 1);
    }
    let path = Path::new(soul_path);
    let json = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) => {
            return Outcome::stderr(
                format!("error: failed to read soul file '{soul_path}': {err}\n"),
                1,
            );
        }
    };
    if json.len() > 64 * 1024 {
        return Outcome::stderr("error: soul file exceeds 64 KiB limit\n".into(), 2);
    }
    let layout = match SoulLayout::from_json(&json) {
        Ok(layout) => layout,
        Err(err) => {
            return Outcome::stderr(format!("error: soul layout: {err}\n"), 2);
        }
    };
    let mut net = match PointNeuralNetwork::init(&[3, 8, 3], 0.01) {
        Ok(net) => net,
        Err(err) => {
            return Outcome::stderr(format!("error: soul network init failed: {err}\n"), 1);
        }
    };
    let (loss, report) = match layout.bootstrap(&mut net) {
        Ok(v) => v,
        Err(err) => {
            return Outcome::stderr(format!("error: soul bootstrap failed: {err}\n"), 1);
        }
    };
    let body = route_with_soul(input, Some(&net), blend_alpha);
    let text = format!(
        "model=soul-net profile_route=blended soul_alpha={blend_alpha:.3} bootstrap_loss={loss:.6} pruned={} records={}\n{body}\n",
        report.pruned_count,
        layout.records.len(),
    );
    Outcome {
        stdout: text,
        stderr: String::new(),
        exit_code: 0,
    }
}

/// Parsed `abi complete` flags after the command token.
#[allow(clippy::struct_excessive_bools)]
struct CompleteRequest {
    learn: bool,
    neural: bool,
    live: bool,
    stream: bool,
    confirm: bool,
    soul_path: Option<String>,
    soul_alpha: Option<f32>,
    model: String,
    input: String,
}

#[allow(clippy::too_many_lines)]
fn parse_complete_args(args: &[String]) -> Result<CompleteRequest, Outcome> {
    let mut learn = false;
    let mut neural = false;
    let mut live = false;
    let mut stream = false;
    let mut confirm = false;
    let mut soul_path: Option<String> = None;
    let mut soul_alpha: Option<f32> = None;
    let mut model = models::DEFAULT_MODEL.to_string();
    let mut input_parts: Vec<&str> = Vec::new();
    let mut saw_separator = false;
    let mut i = 0;
    while i < args.len() {
        let token = args[i].as_str();
        if !saw_separator {
            match token {
                "--" => {
                    saw_separator = true;
                    i += 1;
                    continue;
                }
                "--learn" => {
                    learn = true;
                    i += 1;
                    continue;
                }
                "--neural" => {
                    neural = true;
                    i += 1;
                    continue;
                }
                "--live" => {
                    live = true;
                    i += 1;
                    continue;
                }
                "--stream" => {
                    stream = true;
                    i += 1;
                    continue;
                }
                "--confirm" => {
                    confirm = true;
                    i += 1;
                    continue;
                }
                "--soul" => {
                    i += 1;
                    let Some(value) = args.get(i) else {
                        return Err(Outcome::stderr(
                            "error: --soul requires a JSON file path\n".into(),
                            2,
                        ));
                    };
                    soul_path = Some(value.clone());
                    i += 1;
                    continue;
                }
                "--soul-alpha" => {
                    i += 1;
                    let Some(value) = args.get(i) else {
                        return Err(Outcome::stderr(
                            "error: --soul-alpha requires a number in [0.0, 1.0]\n".into(),
                            2,
                        ));
                    };
                    let Ok(alpha) = value.parse::<f32>() else {
                        return Err(Outcome::stderr(
                            "error: --soul-alpha must be a number in [0.0, 1.0]\n".into(),
                            2,
                        ));
                    };
                    if !(0.0..=1.0).contains(&alpha) {
                        return Err(Outcome::stderr(
                            "error: --soul-alpha must be a number in [0.0, 1.0]\n".into(),
                            2,
                        ));
                    }
                    soul_alpha = Some(alpha);
                    i += 1;
                    continue;
                }
                "--model" => {
                    i += 1;
                    let Some(value) = args.get(i) else {
                        return Err(Outcome::stderr(format!("error: {USAGE}\n"), 2));
                    };
                    model = models::canonical(value).to_string();
                    i += 1;
                    continue;
                }
                flag if flag.starts_with('-') => {
                    return Err(Outcome::stderr(format!("error: {USAGE}\n"), 2));
                }
                _ => {}
            }
        }
        input_parts.push(token);
        i += 1;
    }

    if input_parts.is_empty() {
        return Err(Outcome::stderr(format!("error: {USAGE}\n"), 2));
    }
    if soul_alpha.is_some() && soul_path.is_none() {
        return Err(Outcome::stderr(
            "error: --soul-alpha requires --soul <file>\n".into(),
            2,
        ));
    }
    if soul_path.is_some() && (neural || live || learn) {
        return Err(Outcome::stderr(
            "error: --soul cannot combine with --neural, --live, or --learn\n".into(),
            2,
        ));
    }
    if neural && (learn || live) {
        return Err(Outcome::stderr(
            "error: --neural cannot combine with --live or --learn\n".into(),
            2,
        ));
    }
    if neural && model != models::DEFAULT_MODEL {
        return Err(Outcome::stderr(
            "error: --neural is mutually exclusive with --model\n".into(),
            2,
        ));
    }
    Ok(CompleteRequest {
        learn,
        neural,
        live,
        stream,
        confirm,
        soul_path,
        soul_alpha,
        model,
        input: input_parts.join(" "),
    })
}

/// Dispatch `abi complete …` (args after the command token).
pub(crate) fn run(args: &[String]) -> Outcome {
    let req = match parse_complete_args(args) {
        Ok(req) => req,
        Err(outcome) => return outcome,
    };
    if let Some(path) = req.soul_path {
        return run_soul(&req.input, &path, req.soul_alpha.unwrap_or(0.5));
    }
    if req.neural {
        return run_neural(&req.input);
    }
    if req.live {
        if models::provider_of(&req.model) == models::Provider::Fm {
            return run_fm_complete(&req.input, &req.model, req.confirm);
        }
        return run_live_complete(&req.input, &req.model, req.stream);
    }
    // Local OpenAI-compatible bridge (llama/ollama/mlx/…) — explicit model id.
    if is_local_bridge_model(&req.model) {
        return run_local_bridge(&req.input, &req.model, req.stream);
    }
    if req.stream {
        return Outcome::stderr(
            "error: --stream requires --live (Anthropic SSE) or a local-bridge --model id\n".into(),
            2,
        );
    }
    if req.learn {
        run_learn(&req.input, &req.model)
    } else {
        run_local(&req.input, &req.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_complete_prints_persona_output_without_a_store() {
        let result = complete("hello world", "claude-fable-5").unwrap();
        let text = render_local(&MetaReport {
            requested_model: &result.model,
            profile: result.selected_profile.label(),
            audit_passed: result.audit.passed,
            escore: result.audit.escore,
            audit_vetoed: result.audit.vetoed,
            persisted: false,
            learn: None,
            kv: 0,
            vectors: 0,
            blocks: 0,
            query_id: None,
            response_id: None,
            block_hex: None,
            wdbx_status: Some("no persistent WDBX path configured"),
            output: &result.output,
        });
        assert!(text.contains("requested_model=claude-fable-5"));
        assert!(text.contains("provider=local transport=in-process"));
        assert!(text.contains("generation_engine=persona-template"));
        assert!(text.contains("policy_scope=lexical-signal"));
        assert!(text.contains("profile=abbey"));
        assert!(text.contains("persisted=false"));
        assert!(text.contains("Abbey: hello world"));
    }

    #[test]
    fn missing_input_is_a_usage_error() {
        let outcome = run(&[]);
        assert_eq!(outcome.exit_code, 2);
        assert!(outcome.stderr.contains("usage: abi complete"));
    }

    #[test]
    fn live_without_credentials_is_honest() {
        // Serialize against auth tests that also mutate CREDENTIALS_PATH_ENV.
        let _lock = abi_foundation::env::lock_for_test();
        let isolated = std::env::temp_dir().join(format!(
            "abi-complete-live-creds-{}-{}.json",
            std::process::id(),
            abi_foundation::time::unix_ms()
        ));
        std::fs::write(&isolated, "{}").expect("write empty creds");
        abi_foundation::env::set_override(
            abi_foundation::credentials::CREDENTIALS_PATH_ENV,
            &isolated.to_string_lossy(),
        );
        abi_foundation::env::set_override(abi_foundation::credentials::BACKEND_ENV, "file");
        let outcome = run(&["--live".to_owned(), "hello".to_owned()]);
        abi_foundation::env::clear_override(abi_foundation::credentials::CREDENTIALS_PATH_ENV);
        abi_foundation::env::clear_override(abi_foundation::credentials::BACKEND_ENV);
        let _ = std::fs::remove_file(&isolated);
        assert_eq!(outcome.exit_code, 2, "{}", outcome.stderr);
        assert!(
            outcome.stderr.contains("anthropic") || outcome.stderr.contains("credentials"),
            "{}",
            outcome.stderr
        );
    }

    #[test]
    fn live_rejects_non_anthropic_models() {
        let outcome = run(&[
            "--live".to_owned(),
            "--model".to_owned(),
            "gpt-5".to_owned(),
            "hello".to_owned(),
        ]);
        assert_eq!(outcome.exit_code, 2);
        assert!(outcome.stderr.contains("anthropic models only"));
    }

    #[test]
    fn apple_fm_requires_confirm() {
        let outcome = run(&[
            "--live".to_owned(),
            "--model".to_owned(),
            "apple-fm".to_owned(),
            "hello".to_owned(),
        ]);
        assert_eq!(outcome.exit_code, 2);
        assert!(outcome.stderr.contains("--confirm"));
    }

    #[test]
    fn apple_fm_with_confirm_is_honest_or_live() {
        let outcome = run(&[
            "--live".to_owned(),
            "--model".to_owned(),
            "apple-fm".to_owned(),
            "--confirm".to_owned(),
            "hello".to_owned(),
        ]);
        // On hosts without the bridge / model: exit 1 with unavailable.
        // On arm64 macOS with Apple Intelligence ready: exit 0 with on-device text.
        if outcome.exit_code == 0 {
            assert!(outcome.stdout.contains("provider=foundationmodels"));
            assert!(outcome.stdout.contains("transport=on-device"));
        } else {
            assert_eq!(outcome.exit_code, 1);
            assert!(
                outcome.stderr.contains("FoundationModels unavailable")
                    || outcome.stderr.contains("FoundationModels request failed")
            );
        }
    }

    #[test]
    fn neural_flag_runs_char_lm_demo() {
        let outcome = run(&["--neural".to_owned(), "hello".to_owned()]);
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(outcome.stdout.contains("nn-char-lm"));
        assert!(outcome.stdout.contains("neural=true"));
    }

    #[test]
    fn stream_without_live_is_usage() {
        let outcome = run(&["--stream".to_owned(), "hello".to_owned()]);
        assert_eq!(outcome.exit_code, 2);
        assert!(outcome.stderr.contains("--live"));
    }

    #[test]
    fn soul_alpha_without_soul_is_usage() {
        let outcome = run(&[
            "--soul-alpha".to_owned(),
            "0.3".to_owned(),
            "hello".to_owned(),
        ]);
        assert_eq!(outcome.exit_code, 2);
        assert!(outcome.stderr.contains("--soul"));
    }

    #[test]
    fn soul_alpha_out_of_range_is_usage() {
        let outcome = run(&[
            "--soul".to_owned(),
            "missing.json".to_owned(),
            "--soul-alpha".to_owned(),
            "1.5".to_owned(),
            "hello".to_owned(),
        ]);
        assert_eq!(outcome.exit_code, 2);
        assert!(outcome.stderr.contains("[0.0, 1.0]"));
    }

    #[test]
    fn soul_complete_bootstraps_and_routes() {
        let dir = std::env::temp_dir().join(format!(
            "abi-soul-{}-{}",
            std::process::id(),
            abi_foundation::time::unix_ms()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("soul.json");
        std::fs::write(
            &path,
            r#"[
              {"label":"empathic teaching abbey"},
              {"label":"direct expert aviva concise"},
              {"label":"orchestration governance abi"}
            ]"#,
        )
        .unwrap();
        let outcome = run(&[
            "--soul".to_owned(),
            path.to_string_lossy().into_owned(),
            "--soul-alpha".to_owned(),
            "0.6".to_owned(),
            "analyze the logical structure".to_owned(),
        ]);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(outcome.stdout.contains("model=soul-net"));
        assert!(outcome.stdout.contains("soul_alpha=0.600"));
        assert!(
            outcome.stdout.contains("Abbey:")
                || outcome.stdout.contains("Aviva")
                || outcome.stdout.contains("ABI"),
            "{}",
            outcome.stdout
        );
    }
}
