//! `abi improve` — audit-gated preview/apply wrapper over the SEA learn loop.
//!
//! The default is a dry run: the learn loop runs with `persist` and
//! `adapt_router` off, so the store is read (evidence recall) but never
//! written. `--apply` re-runs the pass with the default config, and only when
//! the dry-run completion passed the constitution audit without a veto, so a
//! rejected completion never becomes evidence or router weight.

use std::fmt::Write as _;

use abi_sea::{LearnLoopConfig, LearnLoopResult, run_learn_loop};
use abi_wdbx::VersionedStore;

use crate::app::Outcome;
use crate::util;

const USAGE: &str = "usage: abi improve [--apply] [--model <id>] <input>";
const DEFAULT_MODEL: &str = "abi-local";

#[derive(Debug, PartialEq, Eq)]
struct Request {
    apply: bool,
    model: String,
    input: String,
}

fn parse(args: &[String]) -> Result<Request, String> {
    let mut apply = false;
    let mut model = DEFAULT_MODEL.to_owned();
    let mut words = Vec::new();
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--apply" => apply = true,
            "--model" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "error: --model requires a value".to_owned())?;
                model.clone_from(value);
            }
            "--" => {
                words.extend(iter.by_ref().cloned());
            }
            other if other.starts_with("--") => {
                return Err(format!("error: unknown option '{other}'"));
            }
            other => words.push(other.to_owned()),
        }
    }
    let input = words.join(" ");
    if input.trim().is_empty() {
        return Err(format!("error: {USAGE}"));
    }
    Ok(Request {
        apply,
        model,
        input,
    })
}

fn dry_run_config() -> LearnLoopConfig {
    LearnLoopConfig {
        persist: false,
        adapt_router: false,
        ..LearnLoopConfig::default()
    }
}

/// Whether a dry-run result may be applied: audit passed and nothing vetoed.
fn audit_allows_apply(result: &LearnLoopResult) -> bool {
    result.completion.audit.passed && !result.completion.audit.vetoed
}

/// Run the preview, then (if requested and allowed) the applying pass.
fn improve(store: &mut VersionedStore, request: &Request, now_ms: i64) -> Result<String, String> {
    let preview = run_learn_loop(
        store,
        &request.input,
        &request.model,
        dry_run_config(),
        now_ms,
    )
    .map_err(|_| "error: improve input must not be empty".to_owned())?;
    let allowed = audit_allows_apply(&preview);
    let mut out = String::new();
    let _ = writeln!(
        out,
        "improve mode={} task={:?} profile={} evidence_count={} audit_passed={} audit_vetoed={} apply_allowed={allowed}",
        if request.apply { "apply" } else { "dry-run" },
        preview.query_task,
        preview.completion.selected_profile.label(),
        preview.evidence_count,
        preview.completion.audit.passed,
        preview.completion.audit.vetoed,
    );
    if !request.apply {
        let _ = writeln!(
            out,
            "dry-run: store unchanged; `abi improve --apply` would persist the completion and update router weights"
        );
        return Ok(out);
    }
    if !allowed {
        let _ = writeln!(
            out,
            "skipped: the completion did not pass the audit, so nothing was persisted"
        );
        return Ok(out);
    }
    let applied = run_learn_loop(
        store,
        &request.input,
        &request.model,
        LearnLoopConfig::default(),
        now_ms,
    )
    .map_err(|_| "error: improve input must not be empty".to_owned())?;
    let _ = writeln!(
        out,
        "applied: persisted={} adapted={}",
        applied.persisted.is_some(),
        applied.adapted
    );
    Ok(out)
}

/// Dispatch `abi improve …` (args after the command token).
pub(crate) fn run(args: &[String]) -> Outcome {
    let request = match parse(args) {
        Ok(request) => request,
        Err(message) => return Outcome::stderr(format!("{message}\n"), 2),
    };
    let Some(mut store) = util::open_store() else {
        return Outcome::stderr(
            "error: abi improve needs a persistent WDBX store (set ABI_WDBX_PATH)\n".to_owned(),
            1,
        );
    };
    match improve(&mut store, &request, abi_foundation::time::unix_ms()) {
        Ok(stdout) => Outcome {
            stdout,
            stderr: String::new(),
            exit_code: 0,
        },
        Err(message) => Outcome::stderr(format!("{message}\n"), 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use abi_wdbx::StorePaths;
    use std::path::PathBuf;

    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn open(tag: &str) -> (Scratch, VersionedStore) {
        let path = std::env::temp_dir().join(format!("abi_improve_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        let store = VersionedStore::open(StorePaths::new(&path)).unwrap();
        (Scratch(path), store)
    }

    fn strings(words: &[&str]) -> Vec<String> {
        words.iter().map(|w| (*w).to_owned()).collect()
    }

    fn request(apply: bool) -> Request {
        Request {
            apply,
            model: DEFAULT_MODEL.to_owned(),
            input: "summarize the prior repair decision".to_owned(),
        }
    }

    #[test]
    fn parse_defaults_to_dry_run() {
        let parsed = parse(&strings(&["plan", "next", "repair"])).unwrap();
        assert!(!parsed.apply);
        assert_eq!(parsed.model, DEFAULT_MODEL);
        assert_eq!(parsed.input, "plan next repair");
    }

    #[test]
    fn parse_reads_apply_model_and_literal_dash() {
        let parsed = parse(&strings(&["--apply", "--model", "m1", "--", "--x"])).unwrap();
        assert!(parsed.apply);
        assert_eq!(parsed.model, "m1");
        assert_eq!(parsed.input, "--x");
    }

    #[test]
    fn parse_rejects_empty_input_and_unknown_options() {
        assert!(parse(&[]).unwrap_err().contains(USAGE));
        assert!(parse(&strings(&["--apply"])).is_err());
        assert!(
            parse(&strings(&["--bogus", "x"]))
                .unwrap_err()
                .contains("--bogus")
        );
        assert!(parse(&strings(&["--model"])).is_err());
    }

    #[test]
    fn dry_run_leaves_the_store_unchanged() {
        let (_dir, mut store) = open("dry");
        let before = store.stats();
        let out = improve(&mut store, &request(false), 1_700_000_000_000).unwrap();
        assert!(out.contains("mode=dry-run"), "{out}");
        assert!(out.contains("store unchanged"), "{out}");
        let after = store.stats();
        assert_eq!(before.kv_entries, after.kv_entries);
        assert_eq!(before.vectors, after.vectors);
        assert_eq!(before.blocks, after.blocks);
    }

    #[test]
    fn apply_writes_only_when_the_audit_allows_it() {
        let (_dir, mut store) = open("apply");
        let before = store.stats();
        let out = improve(&mut store, &request(true), 1_700_000_000_000).unwrap();
        let after = store.stats();
        if out.contains("apply_allowed=true") {
            assert!(out.contains("applied: persisted=true"), "{out}");
            assert!(after.blocks > before.blocks);
        } else {
            assert!(out.contains("skipped:"), "{out}");
            assert_eq!(before.blocks, after.blocks);
            assert_eq!(before.vectors, after.vectors);
        }
    }
}
