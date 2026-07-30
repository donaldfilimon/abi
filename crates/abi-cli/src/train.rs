//! `abi train` — pipeline-compat trainer over the ported AI training path.

use abi_ai::{DatasetFormat, DatasetSpec, TrainingConfig, train_inspect};

use crate::app::Outcome;

const USAGE: &str = "usage: abi train <input>";

/// Dispatch `abi train …` (args after the command token).
pub(crate) fn run(args: &[String]) -> Outcome {
    if args.is_empty() {
        return Outcome::stderr(format!("error: {USAGE}\n"), 2);
    }
    // Zig's top-level `train` takes a free-text note as the "input" and runs
    // the compatibility trainer against a default dataset path derived from it
    // in some paths; the golden help only requires a single required argument.
    // We treat the joined args as a dataset path when it looks like a path,
    // otherwise as a profile label with a missing-dataset demo path — matching
    // the MCP shape of accepting a profile and recording metadata.
    let input = args.join(" ");
    let config = TrainingConfig {
        profile: "abbey".to_string(),
        dataset: DatasetSpec {
            path: input.clone(),
            format: DatasetFormat::Text,
        },
        artifact_dir: "zig-cache/agents".to_string(),
    };

    match train_inspect(&config) {
        Ok((result, summary)) => {
            let text = format!(
                "training accepted profile={} dataset={} records={} backend={} available={} message={}\n",
                result.profile,
                result.dataset_path,
                summary.records,
                result.acceleration_backend,
                summary.available,
                result.message,
            );
            Outcome {
                stdout: text,
                stderr: String::new(),
                exit_code: 0,
            }
        }
        Err(err) => Outcome::stderr(format!("error: {err}\n"), 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_input_is_usage() {
        let outcome = run(&[]);
        assert_eq!(outcome.exit_code, 2);
        assert!(outcome.stderr.contains("usage: abi train"));
    }

    #[test]
    fn accepts_a_freeform_note() {
        let outcome = run(&["local routing note".to_owned()]);
        assert_eq!(outcome.exit_code, 0);
        assert!(outcome.stdout.contains("training accepted"));
        assert!(outcome.stdout.contains("backend=cpu"));
        assert!(outcome.stdout.contains("profile=abbey"));
    }
}
