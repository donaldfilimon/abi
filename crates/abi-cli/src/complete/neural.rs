//! In-process character-language-model demo for `abi complete --neural`.

use crate::app::Outcome;

pub(super) fn run_neural(input: &str) -> Outcome {
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
