//! ABI Rust command-line entrypoint.

use std::io::{self, Write};
use std::process::ExitCode;

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args_os()
        .skip(1)
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect();
    let outcome = abi_cli::app::run(&arguments);

    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    if stdout.write_all(outcome.stdout.as_bytes()).is_err()
        || stderr.write_all(outcome.stderr.as_bytes()).is_err()
    {
        return ExitCode::FAILURE;
    }

    ExitCode::from(outcome.exit_code)
}
