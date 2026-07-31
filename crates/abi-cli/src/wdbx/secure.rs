//! WDBX `secure` subcommand: reference compression + homomorphic-encryption demos (not security-audited).
//!
//! Split from the flat `wdbx` CLI module; dispatch lives in `super::run`.

use crate::app::Outcome;
use std::fmt::Write;

pub(crate) const SECURE_HELP: &str = "usage: abi wdbx secure demo\n\nDemonstrate local compression plus reference homomorphic aggregation; not security-audited FHE.\n";

const fn entropy_mode_name(mode: abi_wdbx::entropy::EntropyMode) -> &'static str {
    match mode {
        abi_wdbx::entropy::EntropyMode::Stored => "stored",
        abi_wdbx::entropy::EntropyMode::Huffman => "huffman",
    }
}

const fn ans_mode_name(mode: abi_wdbx::AnsMode) -> &'static str {
    match mode {
        abi_wdbx::AnsMode::Stored => "stored",
        abi_wdbx::AnsMode::Rans0 => "rans0",
        abi_wdbx::AnsMode::Rans1 => "rans1",
    }
}

fn append_entropy_demo(report: &mut String) -> Result<(), String> {
    let entropy_source = b"WDBX-entropy-demo-aaaaaaaaaa-bbbbbbbbbb-cccccccccc-HELLO";
    let huffman = abi_wdbx::entropy_encode(entropy_source);
    let huffman_round_trip =
        abi_wdbx::entropy_decode(&huffman).map_err(|detail| detail.to_string())?;
    writeln!(
        report,
        "entropy Huffman: mode={} {}B -> serialized {}B ratio={:.2}x roundtrip={}",
        entropy_mode_name(huffman.mode),
        entropy_source.len(),
        huffman.serialized_len(),
        huffman.compression_ratio(),
        huffman_round_trip == entropy_source
    )
    .expect("writing to a String cannot fail");

    let rans0 = abi_wdbx::ans_encode(entropy_source).map_err(|detail| detail.to_string())?;
    let rans0_round_trip = abi_wdbx::ans_decode(&rans0).map_err(|detail| detail.to_string())?;
    writeln!(
        report,
        "entropy rANS0: mode={} {}B -> serialized {}B ratio={:.2}x roundtrip={}",
        ans_mode_name(rans0.mode),
        entropy_source.len(),
        rans0.serialized_len(),
        rans0.compression_ratio(),
        rans0_round_trip == entropy_source
    )
    .expect("writing to a String cannot fail");

    let order_one_source = b"the the the cat sat on the mat the the cat sat";
    let rans1 =
        abi_wdbx::ans_encode_order1(order_one_source).map_err(|detail| detail.to_string())?;
    let rans1_round_trip = abi_wdbx::ans_decode(&rans1).map_err(|detail| detail.to_string())?;
    writeln!(
        report,
        "entropy rANS1: mode={} {}B -> serialized {}B ratio={:.2}x roundtrip={} (demo; not SOTA)",
        ans_mode_name(rans1.mode),
        order_one_source.len(),
        rans1.serialized_len(),
        rans1.compression_ratio(),
        rans1_round_trip == order_one_source
    )
    .expect("writing to a String cannot fail");
    Ok(())
}

fn append_autoencoder_demo(report: &mut String) -> Result<(), String> {
    let mut autoencoder =
        abi_wdbx::Autoencoder::new(8, 4, 0xC0DE_C0DE).map_err(|detail| detail.to_string())?;
    let sample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    autoencoder
        .train_step(&sample, 0.05)
        .map_err(|detail| detail.to_string())?;
    let mut latent = [0.0; 4];
    let mut reconstruction = [0.0; 8];
    autoencoder
        .encode_into(&sample, &mut latent)
        .map_err(|detail| detail.to_string())?;
    autoencoder
        .decode_into(&latent, &mut reconstruction)
        .map_err(|detail| detail.to_string())?;
    writeln!(
        report,
        "neural_compress: autoencoder 8->4->8 recon[0]={:.4} (reference demo, not SOTA)",
        reconstruction[0]
    )
    .expect("writing to a String cannot fail");
    Ok(())
}

fn append_additive_he_demo(report: &mut String) -> Result<(), String> {
    let key = abi_wdbx::HeKey::new(0xAB_CDEF);
    let mut accumulated = key.encrypt(0, 0).map_err(|detail| detail.to_string())?;
    let mut plain_sum = 0;
    for value in 1_u64..=5 {
        let cipher = key
            .encrypt(value * 100, 1000 + value)
            .map_err(|detail| detail.to_string())?;
        accumulated = abi_wdbx::he_add(&accumulated, &cipher);
        plain_sum += value * 100;
    }
    let decrypted = key
        .decrypt(&accumulated)
        .map_err(|detail| detail.to_string())?;
    writeln!(
        report,
        "additive HE: sum of 5 encrypted values decrypts to {decrypted} (expected {plain_sum}, match={})",
        decrypted == plain_sum
    )
    .expect("writing to a String cannot fail");
    Ok(())
}

fn append_dghv_demo(report: &mut String) {
    let mut random = abi_wdbx::DghvRng::new(0x5EED_F00D_C0FF_EE11);
    let keypair = abi_wdbx::dghv_keygen(&mut random);
    let encrypted_one = abi_wdbx::dghv_encrypt(&keypair, &mut random, true);
    let encrypted_one_again = abi_wdbx::dghv_encrypt(&keypair, &mut random, true);
    let encrypted_zero = abi_wdbx::dghv_encrypt(&keypair, &mut random, false);
    let product = abi_wdbx::dghv_mul(&keypair, &encrypted_one, &encrypted_one_again);
    let evaluated = abi_wdbx::dghv_add(&keypair, &product, &encrypted_zero);
    let evaluated_bit = u8::from(abi_wdbx::dghv_decrypt(&keypair, &evaluated));
    writeln!(
        report,
        "homomorphic eval: enc((1 AND 1) XOR 0) decrypts to {evaluated_bit} (expected 1, match={})",
        evaluated_bit == 1
    )
    .expect("writing to a String cannot fail");
    report.push_str(
        "(DGHV somewhat-homomorphic scheme: real encrypted add+multiply on ciphertexts, reference parameters / bounded depth — not security-audited)\n",
    );
}

fn secure_demo_result() -> Result<String, String> {
    let vector: Vec<f32> = (0_u16..128)
        .map(|index| (f32::from(index) * 0.1).sin())
        .collect();
    let quantized = abi_wdbx::quantize(&vector).map_err(|detail| detail.to_string())?;
    let reconstructed = abi_wdbx::dequantize(&quantized);
    let mut report = format!(
        "compression: {} f32 -> int8 codes, ratio={:.2}x, max_error={:.5}\n",
        vector.len(),
        quantized.compression_ratio(),
        abi_wdbx::max_error(&vector, &reconstructed)
    );
    append_entropy_demo(&mut report)?;
    append_autoencoder_demo(&mut report)?;
    append_additive_he_demo(&mut report)?;
    append_dghv_demo(&mut report);
    report.push_str(
        "north-star status: Partial — int8 + Huffman + rANS/order-1 demos + autoencoder + additive HE + reference DGHV SHE (not audited, not SOTA); production FHE/SOTA codecs remain Proposed\n",
    );
    Ok(report)
}

pub(crate) fn run_secure(args: &[String]) -> Outcome {
    match args {
        [operation] if operation == "demo" => match secure_demo_result() {
            Ok(report) => Outcome::stderr(report, 0),
            Err(detail) => super::error("secure demo failed", detail),
        },
        _ => super::usage(),
    }
}
