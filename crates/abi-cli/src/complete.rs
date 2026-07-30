//! `abi complete` — local completion, optional SEA `--learn`.
//!
//! Ported from the local path of `src/cli/handlers/complete_handlers.zig`.
//! Live/remote (`--live`), `FoundationModels` (`apple-fm` + `--confirm`), soul
//! routing, and `--neural` remain honestly not-yet-ported.

use std::fmt::Write as _;

use abi_ai::{complete, models};
use abi_sea::{LearnLoopConfig, run_learn_loop};
use abi_wdbx::{DurableStore, StorePaths};

use crate::app::Outcome;

const USAGE: &str = "usage: abi complete [--learn] [--model <id>] [--] <input>";

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

/// Dispatch `abi complete …` (args after the command token).
pub(crate) fn run(args: &[String]) -> Outcome {
    let mut learn = false;
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
                "--model" => {
                    i += 1;
                    let Some(value) = args.get(i) else {
                        return Outcome::stderr(format!("error: {USAGE}\n"), 2);
                    };
                    model = models::canonical(value).to_string();
                    i += 1;
                    continue;
                }
                "--live" | "--stream" | "--neural" | "--confirm" | "--soul" | "--soul-alpha" => {
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
    let input = input_parts.join(" ");
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
    fn live_flag_is_honestly_not_yet_ported() {
        let outcome = run(&["--live".to_owned(), "hello".to_owned()]);
        assert_eq!(outcome.exit_code, 1);
        assert!(outcome.stderr.contains("not yet ported"));
    }
}
