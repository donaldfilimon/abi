//! `abi nn` — miniature character-level demo trainer.
//!
//! Ported from `src/cli/handlers/nn.zig`. Demo-grade only. Optional JSON
//! checkpoint via `--out` / `--checkpoint` for train→sample durability.

use std::fmt::Write as _;
use std::path::Path;

use abi_nn::{TrainConfig, format_report, load_checkpoint, sample, save_checkpoint, train_model};

use crate::app::Outcome;

const USAGE: &str = "\
abi nn <command> ...   (miniature character-level demo trainer)

  train \"<text>\" [--out <path.json>]                 Train on inline text; optional checkpoint
  train --jsonl <path> [--field <name>] [--out <p>]   Train on JSONL; optional checkpoint
  sample --text \"<corpus>\" --seed <char> --n <k>    Train then sample
  sample --checkpoint <path.json> --seed <char> --n <k>  Sample from a saved checkpoint

This is a demonstration char-level trainer, not a production/LLM/distributed trainer.
";

const TRAIN_HELP: &str = "\
usage: abi nn train \"<text>\" [--out <path.json>]
       abi nn train --jsonl <path> [--field <name>] [--out <path.json>]

Train the miniature local character model from inline text or a JSONL text field.
With --out, write a demo JSON checkpoint (not a production model format).
";

const SAMPLE_HELP: &str = "\
usage: abi nn sample --text \"<corpus>\" --seed <char> --n <k>
       abi nn sample --checkpoint <path.json> --seed <char> --n <k>

Train on <corpus> or load a checkpoint, then greedily emit k characters from the seed byte.
";

/// Dispatch `abi nn …` (args after the `nn` command token).
pub(crate) fn run(args: &[String]) -> Outcome {
    if args.is_empty() {
        return Outcome::stderr(USAGE.to_owned(), 2);
    }
    let sub = args[0].as_str();
    if matches!(sub, "--help" | "-h" | "help") {
        return Outcome::stderr(USAGE.to_owned(), 0);
    }
    match sub {
        "train" => train_cmd(&args[1..]),
        "sample" => sample_cmd(&args[1..]),
        _ => Outcome::stderr(USAGE.to_owned(), 2),
    }
}

fn train_cmd(args: &[String]) -> Outcome {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h" | "help") {
        return Outcome::stderr(TRAIN_HELP.to_owned(), 0);
    }

    let mut jsonl_path: Option<&str> = None;
    let mut field = "text";
    let mut inline_text: Option<&str> = None;
    let mut out_path: Option<&str> = None;

    let mut i = 0;
    while i < args.len() {
        let tok = args[i].as_str();
        match tok {
            "--jsonl" => {
                i += 1;
                let Some(path) = args.get(i) else {
                    return Outcome::stderr(USAGE.to_owned(), 2);
                };
                jsonl_path = Some(path.as_str());
            }
            "--field" => {
                i += 1;
                let Some(name) = args.get(i) else {
                    return Outcome::stderr(USAGE.to_owned(), 2);
                };
                field = name.as_str();
            }
            "--out" => {
                i += 1;
                let Some(path) = args.get(i) else {
                    return Outcome::stderr(USAGE.to_owned(), 2);
                };
                out_path = Some(path.as_str());
            }
            _ => {
                if inline_text.is_some() {
                    return Outcome::stderr(USAGE.to_owned(), 2);
                }
                inline_text = Some(tok);
            }
        }
        i += 1;
    }

    if jsonl_path.is_some() && inline_text.is_some() {
        return Outcome::stderr(USAGE.to_owned(), 2);
    }

    let model = match train_model_from_args(jsonl_path, field, inline_text) {
        Ok(m) => m,
        Err(outcome) => return outcome,
    };

    let mut stdout = format!("{}\n", format_report(&model.report));
    if let Some(path) = out_path {
        if let Err(err) = save_checkpoint(Path::new(path), &model) {
            return Outcome::stderr(format!("error: failed to write checkpoint: {err}\n"), 1);
        }
        let _ = writeln!(stdout, "nn checkpoint: wrote {path}");
    }

    Outcome {
        stdout,
        stderr: String::new(),
        exit_code: 0,
    }
}

fn train_model_from_args(
    jsonl_path: Option<&str>,
    field: &str,
    inline_text: Option<&str>,
) -> Result<abi_nn::Model, Outcome> {
    if let Some(path) = jsonl_path {
        let raw = std::fs::read_to_string(path).map_err(|err| {
            Outcome::stderr(format!("error: nn train --jsonl failed: {err}\n"), 1)
        })?;
        let bytes = abi_nn::extract_corpus_from_jsonl(&raw, field).map_err(|err| {
            Outcome::stderr(format!("error: nn train --jsonl failed: {err}\n"), 1)
        })?;
        train_model(&bytes, TrainConfig::default())
            .map_err(|err| Outcome::stderr(format!("error: nn train --jsonl failed: {err}\n"), 1))
    } else if let Some(text) = inline_text {
        train_model(text.as_bytes(), TrainConfig::default())
            .map_err(|err| Outcome::stderr(format!("error: nn train failed: {err}\n"), 1))
    } else {
        Err(Outcome::stderr(USAGE.to_owned(), 2))
    }
}

fn sample_cmd(args: &[String]) -> Outcome {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h" | "help") {
        return Outcome::stderr(SAMPLE_HELP.to_owned(), 0);
    }

    let mut text: Option<&str> = None;
    let mut checkpoint: Option<&str> = None;
    let mut seed: Option<u8> = None;
    let mut n: usize = 16;

    let mut i = 0;
    while i < args.len() {
        let tok = args[i].as_str();
        match tok {
            "--text" => {
                i += 1;
                let Some(value) = args.get(i) else {
                    return Outcome::stderr(USAGE.to_owned(), 2);
                };
                text = Some(value.as_str());
            }
            "--checkpoint" => {
                i += 1;
                let Some(value) = args.get(i) else {
                    return Outcome::stderr(USAGE.to_owned(), 2);
                };
                checkpoint = Some(value.as_str());
            }
            "--seed" => {
                i += 1;
                let Some(value) = args.get(i) else {
                    return Outcome::stderr(USAGE.to_owned(), 2);
                };
                if value.is_empty() {
                    return Outcome::stderr(USAGE.to_owned(), 2);
                }
                seed = Some(value.as_bytes()[0]);
            }
            "--n" => {
                i += 1;
                let Some(value) = args.get(i) else {
                    return Outcome::stderr(USAGE.to_owned(), 2);
                };
                let Ok(parsed) = value.parse::<usize>() else {
                    return Outcome::stderr(USAGE.to_owned(), 2);
                };
                n = parsed;
            }
            _ => return Outcome::stderr(USAGE.to_owned(), 2),
        }
        i += 1;
    }

    let Some(seed_char) = seed else {
        return Outcome::stderr(USAGE.to_owned(), 2);
    };
    if text.is_some() && checkpoint.is_some() {
        return Outcome::stderr(
            "error: sample accepts either --text or --checkpoint, not both\n".into(),
            2,
        );
    }

    let model = if let Some(path) = checkpoint {
        match load_checkpoint(Path::new(path)) {
            Ok(m) => m,
            Err(err) => {
                return Outcome::stderr(format!("error: failed to load checkpoint: {err}\n"), 1);
            }
        }
    } else if let Some(corpus) = text {
        match train_model(corpus.as_bytes(), TrainConfig::default()) {
            Ok(model) => model,
            Err(err) => {
                return Outcome::stderr(format!("error: nn sample training failed: {err}\n"), 1);
            }
        }
    } else {
        return Outcome::stderr(USAGE.to_owned(), 2);
    };

    let out = sample(&model, seed_char, n);
    let sampled = String::from_utf8_lossy(&out);
    Outcome {
        stdout: format!("{}\nnn sample: {sampled}\n", format_report(&model.report)),
        stderr: String::new(),
        exit_code: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_exits_zero() {
        let outcome = run(&["--help".to_owned()]);
        assert_eq!(outcome.exit_code, 0);
        assert!(outcome.stderr.contains("miniature character-level"));
    }

    #[test]
    fn train_on_inline_text_improves_loss() {
        let outcome = run(&["train".to_owned(), "hello world hello world ".to_owned()]);
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(outcome.stdout.contains("nn train:"));
        assert!(outcome.stdout.contains("improved=true"));
    }

    #[test]
    fn sample_emits_characters() {
        let outcome = run(&[
            "sample".to_owned(),
            "--text".to_owned(),
            "hello world hello world ".to_owned(),
            "--seed".to_owned(),
            "h".to_owned(),
            "--n".to_owned(),
            "8".to_owned(),
        ]);
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(outcome.stdout.contains("nn sample:"));
        assert!(outcome.stdout.contains("improved=true"));
    }

    #[test]
    fn checkpoint_round_trip_via_cli() {
        let dir = abi_foundation::temp_path::temp_file_path("abi-nn-cli-checkpoint", "scratch");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.json");
        let train = run(&[
            "train".to_owned(),
            "hello world hello world ".to_owned(),
            "--out".to_owned(),
            path.to_string_lossy().into_owned(),
        ]);
        assert_eq!(train.exit_code, 0, "{}", train.stderr);
        assert!(train.stdout.contains("nn checkpoint: wrote"));
        let sample = run(&[
            "sample".to_owned(),
            "--checkpoint".to_owned(),
            path.to_string_lossy().into_owned(),
            "--seed".to_owned(),
            "h".to_owned(),
            "--n".to_owned(),
            "6".to_owned(),
        ]);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(sample.exit_code, 0, "{}", sample.stderr);
        assert!(sample.stdout.contains("nn sample:"));
    }

    #[test]
    fn missing_subcommand_is_usage() {
        let outcome = run(&[]);
        assert_eq!(outcome.exit_code, 2);
    }

    #[test]
    fn train_jsonl_round_trip() {
        let dir = abi_foundation::temp_path::temp_file_path("abi-nn-cli-jsonl", "scratch");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.jsonl");
        std::fs::write(
            &path,
            r#"{"text":"hello world hello world "}
{"text":"hello world hello world "}
"#,
        )
        .unwrap();
        let outcome = run(&[
            "train".to_owned(),
            "--jsonl".to_owned(),
            path.to_string_lossy().into_owned(),
        ]);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(outcome.stdout.contains("improved=true"));
    }
}
