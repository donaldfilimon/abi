//! `abi nn` — miniature character-level demo trainer.
//!
//! Ported from `src/cli/handlers/nn.zig`. Demo-grade only.

use abi_nn::{TrainConfig, format_report, sample, train_model, train_on_jsonl, train_on_text};

use crate::app::Outcome;

const USAGE: &str = "\
abi nn <command> ...   (miniature character-level demo trainer)

  train \"<text>\"                                  Train on an inline text corpus
  train --jsonl <path> [--field <name>]           Train on a JSONL dataset (default field \"text\")
  sample --text \"<corpus>\" --seed <char> --n <k>  Train on <corpus>, then greedily emit k chars

This is a demonstration char-level trainer, not a production/LLM/distributed trainer.
";

const TRAIN_HELP: &str = "\
usage: abi nn train \"<text>\" | train --jsonl <path> [--field <name>]

Train the miniature local character model from inline text or a JSONL text field.
";

const SAMPLE_HELP: &str = "\
usage: abi nn sample --text \"<corpus>\" --seed <char> --n <k>

Train on <corpus>, then greedily emit k characters from the seed byte.
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

    let report = if let Some(path) = jsonl_path {
        match train_on_jsonl(path, field, TrainConfig::default()) {
            Ok(report) => report,
            Err(err) => {
                return Outcome::stderr(format!("error: nn train --jsonl failed: {err}\n"), 1);
            }
        }
    } else if let Some(text) = inline_text {
        match train_on_text(text.as_bytes(), TrainConfig::default()) {
            Ok(report) => report,
            Err(err) => {
                return Outcome::stderr(format!("error: nn train failed: {err}\n"), 1);
            }
        }
    } else {
        return Outcome::stderr(USAGE.to_owned(), 2);
    };

    Outcome {
        stdout: format!("{}\n", format_report(&report)),
        stderr: String::new(),
        exit_code: 0,
    }
}

fn sample_cmd(args: &[String]) -> Outcome {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h" | "help") {
        return Outcome::stderr(SAMPLE_HELP.to_owned(), 0);
    }

    let mut text: Option<&str> = None;
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

    let Some(corpus) = text else {
        return Outcome::stderr(USAGE.to_owned(), 2);
    };
    let Some(seed_char) = seed else {
        return Outcome::stderr(USAGE.to_owned(), 2);
    };

    let model = match train_model(corpus.as_bytes(), TrainConfig::default()) {
        Ok(model) => model,
        Err(err) => {
            return Outcome::stderr(format!("error: nn sample training failed: {err}\n"), 1);
        }
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
    fn missing_subcommand_is_usage() {
        let outcome = run(&[]);
        assert_eq!(outcome.exit_code, 2);
    }

    #[test]
    fn train_jsonl_round_trip() {
        let dir = std::env::temp_dir().join(format!("abi_nn_cli_test_{}", std::process::id()));
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
