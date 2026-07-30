//! `abi complete` — local completion, optional SEA `--learn`, neural demo,
//! and explicit live Anthropic transport.
//!
//! Ported from the local and live paths of `src/cli/handlers/complete_handlers.zig`.
//! `--live` is Anthropic-only (matching Zig). Apple `FoundationModels`
//! (`apple-fm` + `--confirm`) reports honest unavailability (no Swift FFI).
//! Soul routing remains honestly not-yet-ported.

use std::fmt::Write as _;
use std::io::Write as _;

use abi_ai::{complete, models};
use abi_connectors::{Client, ConnectorConfig, DefaultTransport, Provider, parse_stream};
use abi_foundation::credentials::{self, CredentialField};
use abi_sea::{LearnLoopConfig, run_learn_loop};
use abi_wdbx::{DurableStore, StorePaths};

use crate::app::Outcome;

const USAGE: &str = "usage: abi complete [--live] [--stream] [--learn] [--neural] [--model <id>] [--confirm] [--] <input>";

/// Fields for the one-line-per-field completion metadata block.
struct MetaReport<'a> {
    model: &'a str,
    profile: &'a str,
    audit_passed: bool,
    escore: f32,
    audit_vetoed: bool,
    persisted: bool,
    learn: Option<(usize, bool)>,
    kv: usize,
    vectors: usize,
    blocks: usize,
    query_id: Option<u64>,
    response_id: Option<u64>,
    block_hex: Option<&'a str>,
    wdbx_status: Option<&'a str>,
    output: &'a str,
}

/// Resolve the durable store the same way MCP does, or `None` for in-memory.
fn open_store() -> Option<DurableStore> {
    if let Ok(path) = std::env::var("ABI_WDBX_PATH") {
        if path == ":memory:" {
            return None;
        }
        return DurableStore::open(StorePaths::new(path)).ok();
    }
    if matches!(
        std::env::var("ABI_WDBX_PERSIST").as_deref(),
        Ok("0" | "false" | "no" | "off")
    ) {
        return None;
    }
    let home = std::env::var("HOME").ok()?;
    DurableStore::open(StorePaths::new(format!("{home}/.abi/wdbx"))).ok()
}

fn render_local(report: &MetaReport<'_>) -> String {
    let mut out = String::new();
    if let Some((evidence_count, adapted)) = report.learn {
        let _ = writeln!(
            out,
            "model={} profile={} audit_passed={} audit_escore={:.3} audit_vetoed={} persisted={} learn=true evidence_count={evidence_count} adapted={adapted}",
            report.model,
            report.profile,
            report.audit_passed,
            report.escore,
            report.audit_vetoed,
            report.persisted,
        );
    } else {
        let _ = writeln!(
            out,
            "model={} profile={} audit_passed={} audit_escore={:.3} audit_vetoed={} persisted={}",
            report.model,
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

    let mut store = open_store();
    let (persisted, qid, rid, hex, kv, vectors, blocks, status) =
        if let Some(store) = store.as_mut() {
            let before = store.stats();
            let query = abi_ai::text_embedding(input);
            let response = abi_ai::text_embedding(&result.output);
            let query_id = store.put_vector(&query).ok();
            let response_id = store.put_vector(&response).ok();
            let (qid, rid, hex) = match (query_id, response_id) {
                (Some(q), Some(r)) => {
                    let metadata = abi_ai::completion::metadata_json(input, &result, q, r);
                    let key = abi_ai::completion::metadata_key(q);
                    let _ = store.put(&key, &metadata);
                    let block = store
                        .add_block(
                            result.selected_profile.label(),
                            q,
                            r,
                            &metadata,
                            abi_foundation::time::unix_ms(),
                        )
                        .ok();
                    (Some(q), Some(r), block.map(|b| b.hash.to_hex()))
                }
                _ => (None, None, None),
            };
            let after = store.stats();
            let persisted = qid.is_some() && rid.is_some() && hex.is_some();
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
        model: &result.model,
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
    let Some(mut store) = open_store() else {
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
        model: &result.model,
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
/// Honest: the Swift FFI shim is not linked in the Rust port, so this always
/// reports unavailability rather than inventing on-device inference.
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
    Outcome::stderr(
        format!(
            "error: on-device FoundationModels unavailable for model={model}: not built with FoundationModels FFI, or the on-device runtime is not reachable on this host\n"
        ),
        1,
    )
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

/// Dispatch `abi complete …` (args after the command token).
#[allow(clippy::too_many_lines)]
pub(crate) fn run(args: &[String]) -> Outcome {
    let mut learn = false;
    let mut neural = false;
    let mut live = false;
    let mut stream = false;
    let mut confirm = false;
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
                "--model" => {
                    i += 1;
                    let Some(value) = args.get(i) else {
                        return Outcome::stderr(format!("error: {USAGE}\n"), 2);
                    };
                    model = models::canonical(value).to_string();
                    i += 1;
                    continue;
                }
                "--soul" | "--soul-alpha" => {
                    return Outcome::stderr(
                        format!(
                            "error: Rust handler for complete flag `{token}` is not yet ported\n"
                        ),
                        1,
                    );
                }
                flag if flag.starts_with('-') => {
                    return Outcome::stderr(format!("error: {USAGE}\n"), 2);
                }
                _ => {}
            }
        }
        input_parts.push(token);
        i += 1;
    }

    if input_parts.is_empty() {
        return Outcome::stderr(format!("error: {USAGE}\n"), 2);
    }
    if neural && (learn || live) {
        return Outcome::stderr(
            "error: --neural cannot combine with --live or --learn\n".into(),
            2,
        );
    }
    if neural && model != models::DEFAULT_MODEL {
        return Outcome::stderr(
            "error: --neural is mutually exclusive with --model\n".into(),
            2,
        );
    }
    if stream && !live && !neural {
        // Zig streams local via scheduler; we only expose stream with live SSE
        // for now (honest scope). Local streaming persona chunks are incremental
        // but not requested here — use local non-stream complete.
        return Outcome::stderr(
            "error: --stream currently requires --live (Anthropic SSE)\n".into(),
            2,
        );
    }
    let input = input_parts.join(" ");
    if neural {
        return run_neural(&input);
    }
    if live {
        if models::provider_of(&model) == models::Provider::Fm {
            return run_fm_complete(&input, &model, confirm);
        }
        return run_live_complete(&input, &model, stream);
    }
    if learn {
        run_learn(&input, &model)
    } else {
        run_local(&input, &model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_complete_prints_persona_output_without_a_store() {
        let result = complete("hello world", "claude-fable-5").unwrap();
        let text = render_local(&MetaReport {
            model: &result.model,
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
        assert!(text.contains("model=claude-fable-5"));
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
    fn apple_fm_with_confirm_is_honestly_unavailable() {
        let outcome = run(&[
            "--live".to_owned(),
            "--model".to_owned(),
            "apple-fm".to_owned(),
            "--confirm".to_owned(),
            "hello".to_owned(),
        ]);
        assert_eq!(outcome.exit_code, 1);
        assert!(outcome.stderr.contains("FoundationModels unavailable"));
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
}
