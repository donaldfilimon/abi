//! Deterministic vision, audio, video, and voice feature extractors.
//!
//! These are classical, non-learned processors. They do not decode camera
//! containers, run a neural vision model, or transcribe speech. A browser page
//! (or another caller) downsamples pixels and PCM first, then posts the already
//! reduced buffers here.
//!
//! Embeddings are unit vectors of [`EMBED_DIM`] so they can sit next to the
//! existing text-embedding space without inventing a second store geometry.

use crate::embedding::{EMBED_DIM, text_embedding};
use abi_foundation::wyhash;
use serde::Serialize;
use serde_json::{Value, json};

/// Vision / audio / video embedding width. Same as [`EMBED_DIM`].
pub const VISION_DIM: usize = EMBED_DIM;
/// Audio embedding width. Same as [`EMBED_DIM`].
pub const AUDIO_DIM: usize = EMBED_DIM;
/// Video embedding width. Same as [`EMBED_DIM`].
pub const VIDEO_DIM: usize = EMBED_DIM;

/// Largest accepted image/frame side in pixels.
pub const MAX_SIDE: u32 = 64;
/// Largest accepted video frame count.
pub const MAX_FRAMES: usize = 8;
/// Largest accepted PCM sample count.
pub const MAX_SAMPLES: usize = 2048;
/// Lowest accepted audio sample rate.
pub const MIN_SAMPLE_RATE: u32 = 8_000;
/// Highest accepted audio sample rate.
pub const MAX_SAMPLE_RATE: u32 = 48_000;

const ENGINE: &str = "deterministic-local";
const VISION_DISCLOSURE: &str = "Not a neural vision model. Bucket-hash embedding plus classical stats from a downsampled grayscale buffer.";
const AUDIO_DISCLOSURE: &str = "Not a speech recognizer or neural audio encoder. Bucket-hash embedding plus RMS/ZCR/band energy from PCM.";
const VIDEO_DISCLOSURE: &str = "Not a video understanding model. Mean-pooled per-frame vision embeddings plus inter-frame motion stats.";
const VOICE_DISCLOSURE: &str = "Not in-process STT. Transcript text is embedded lexically; optional PCM is analyzed as audio. Browser Web Speech is client-side when used.";
const FUSE_DISCLOSURE: &str = "Mean-pool of provided unit embeddings, then L2-normalized. Not a learned multimodal fusion model.";

/// A claim-honest analysis result.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnalysisReport {
    /// `image`, `video`, `audio`, `voice`, or `fuse`.
    pub kind: &'static str,
    /// Always `deterministic-local`.
    pub engine: &'static str,
    /// What this path is not.
    pub disclosure: &'static str,
    /// Unit embedding of [`EMBED_DIM`] floats.
    pub embedding: Vec<f32>,
    /// Classical measurements for the same buffer.
    pub stats: Value,
}

/// Why a buffer was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaError {
    /// No payload was supplied.
    Empty,
    /// Width or height was zero or above [`MAX_SIDE`].
    InvalidGeometry,
    /// `pixels.len()` did not equal `width * height`.
    PixelCountMismatch,
    /// More than [`MAX_FRAMES`] frames.
    TooManyFrames,
    /// A frame did not match the declared geometry.
    FrameSizeMismatch,
    /// Sample rate outside `[MIN_SAMPLE_RATE, MAX_SAMPLE_RATE]`.
    InvalidSampleRate,
    /// More than [`MAX_SAMPLES`] samples, or zero samples when audio is required.
    InvalidSampleCount,
    /// A PCM sample was NaN or infinite.
    NonFiniteSample,
    /// Fuse was called with no embeddings.
    NothingToFuse,
    /// An embedding did not have [`EMBED_DIM`] finite elements.
    InvalidEmbedding,
}

impl MediaError {
    /// HTTP-ish status for the studio router.
    #[must_use]
    pub const fn status(self) -> u16 {
        400
    }

    /// Stable error token.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Empty => "empty media payload",
            Self::InvalidGeometry => "width and height must be 1..=64",
            Self::PixelCountMismatch => "pixel count must equal width * height",
            Self::TooManyFrames => "video is limited to 8 frames",
            Self::FrameSizeMismatch => "every frame must match width * height",
            Self::InvalidSampleRate => "sample_rate must be 8000..=48000",
            Self::InvalidSampleCount => "audio needs 1..=2048 finite samples",
            Self::NonFiniteSample => "audio samples must be finite",
            Self::NothingToFuse => "fuse needs at least one embedding",
            Self::InvalidEmbedding => "embeddings must be 32 finite floats",
        }
    }
}

/// Analyze one grayscale image.
pub fn analyze_image(width: u32, height: u32, pixels: &[u8]) -> Result<AnalysisReport, MediaError> {
    let (width_usize, height_usize) = checked_geometry(width, height)?;
    if pixels.is_empty() {
        return Err(MediaError::Empty);
    }
    if pixels.len() != width_usize.saturating_mul(height_usize) {
        return Err(MediaError::PixelCountMismatch);
    }
    let embedding = embed_bytes(1, pixels);
    Ok(AnalysisReport {
        kind: "image",
        engine: ENGINE,
        disclosure: VISION_DISCLOSURE,
        embedding: embedding.to_vec(),
        stats: image_stats(width, height, pixels),
    })
}

/// Analyze a short grayscale frame sequence.
pub fn analyze_video(
    width: u32,
    height: u32,
    frames: &[Vec<u8>],
) -> Result<AnalysisReport, MediaError> {
    if frames.is_empty() {
        return Err(MediaError::Empty);
    }
    if frames.len() > MAX_FRAMES {
        return Err(MediaError::TooManyFrames);
    }
    let (width_usize, height_usize) = checked_geometry(width, height)?;
    let expected = width_usize.saturating_mul(height_usize);
    let mut pooled = [0.0_f32; EMBED_DIM];
    let mut previous: Option<&[u8]> = None;
    let mut motion_sum = 0.0_f32;
    let mut motion_pairs = 0_u32;
    let mut scene_cuts = 0_u32;
    for frame in frames {
        if frame.len() != expected {
            return Err(MediaError::FrameSizeMismatch);
        }
        let embedding = embed_bytes(2, frame);
        for (slot, value) in pooled.iter_mut().zip(embedding) {
            *slot += value;
        }
        if let Some(prior) = previous {
            let motion = mean_abs_delta(prior, frame);
            motion_sum += motion;
            motion_pairs += 1;
            if motion > 16.0 {
                scene_cuts += 1;
            }
        }
        previous = Some(frame.as_slice());
    }
    let frame_count = f32_from_count(frames.len());
    for value in &mut pooled {
        *value /= frame_count;
    }
    let embedding = l2_normalize(pooled);
    let mean_motion = if motion_pairs == 0 {
        0.0
    } else {
        motion_sum / f32_from_count(motion_pairs)
    };
    Ok(AnalysisReport {
        kind: "video",
        engine: ENGINE,
        disclosure: VIDEO_DISCLOSURE,
        embedding: embedding.to_vec(),
        stats: json!({
            "width": width,
            "height": height,
            "frames": frames.len(),
            "mean_motion": mean_motion,
            "scene_cuts": scene_cuts,
        }),
    })
}

/// Analyze a short PCM clip in `[-1, 1]`.
pub fn analyze_audio(sample_rate: u32, samples: &[f32]) -> Result<AnalysisReport, MediaError> {
    validate_audio(sample_rate, samples)?;
    let quantized = quantize_pcm(samples);
    let embedding = embed_bytes(3, &quantized);
    Ok(AnalysisReport {
        kind: "audio",
        engine: ENGINE,
        disclosure: AUDIO_DISCLOSURE,
        embedding: embedding.to_vec(),
        stats: audio_stats(sample_rate, samples),
    })
}

/// Analyze a client-side transcript plus optional PCM.
pub fn analyze_voice(
    transcript: &str,
    sample_rate: Option<u32>,
    samples: Option<&[f32]>,
) -> Result<AnalysisReport, MediaError> {
    if transcript.is_empty() && samples.is_none_or(<[f32]>::is_empty) {
        return Err(MediaError::Empty);
    }
    let text = text_embedding(transcript);
    let (embedding, audio_stats_value) = match (sample_rate, samples) {
        (Some(rate), Some(pcm)) if !pcm.is_empty() => {
            validate_audio(rate, pcm)?;
            let audio = embed_bytes(3, &quantize_pcm(pcm));
            (
                l2_normalize(mean_pair(text, audio)),
                Some(audio_stats(rate, pcm)),
            )
        }
        _ => (text, None),
    };
    let words = transcript
        .split_whitespace()
        .filter(|word| !word.is_empty())
        .count();
    Ok(AnalysisReport {
        kind: "voice",
        engine: ENGINE,
        disclosure: VOICE_DISCLOSURE,
        embedding: embedding.to_vec(),
        stats: json!({
            "transcript_chars": transcript.chars().count(),
            "transcript_words": words,
            "has_audio": audio_stats_value.is_some(),
            "client_stt": "WebSpeechRecognition-or-operator-supplied",
            "audio": audio_stats_value,
        }),
    })
}

/// Mean-pool provided unit embeddings and L2-normalize.
pub fn fuse_embeddings(
    vision: Option<&[f32]>,
    audio: Option<&[f32]>,
    video: Option<&[f32]>,
) -> Result<AnalysisReport, MediaError> {
    let mut acc = [0.0_f32; EMBED_DIM];
    let mut parts = Vec::new();
    for (name, values) in [("vision", vision), ("audio", audio), ("video", video)] {
        let Some(values) = values else {
            continue;
        };
        let parsed = parse_embedding(values)?;
        for (slot, value) in acc.iter_mut().zip(parsed) {
            *slot += value;
        }
        parts.push(name);
    }
    if parts.is_empty() {
        return Err(MediaError::NothingToFuse);
    }
    let count = f32_from_count(parts.len());
    for value in &mut acc {
        *value /= count;
    }
    let embedding = l2_normalize(acc);
    Ok(AnalysisReport {
        kind: "fuse",
        engine: ENGINE,
        disclosure: FUSE_DISCLOSURE,
        embedding: embedding.to_vec(),
        stats: json!({ "parts": parts, "count": parts.len() }),
    })
}

fn checked_geometry(width: u32, height: u32) -> Result<(usize, usize), MediaError> {
    if !(1..=MAX_SIDE).contains(&width) || !(1..=MAX_SIDE).contains(&height) {
        return Err(MediaError::InvalidGeometry);
    }
    Ok((
        usize::try_from(width).map_err(|_| MediaError::InvalidGeometry)?,
        usize::try_from(height).map_err(|_| MediaError::InvalidGeometry)?,
    ))
}

fn validate_audio(sample_rate: u32, samples: &[f32]) -> Result<(), MediaError> {
    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&sample_rate) {
        return Err(MediaError::InvalidSampleRate);
    }
    if samples.is_empty() || samples.len() > MAX_SAMPLES {
        return Err(MediaError::InvalidSampleCount);
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(MediaError::NonFiniteSample);
    }
    Ok(())
}

fn parse_embedding(values: &[f32]) -> Result<[f32; EMBED_DIM], MediaError> {
    if values.len() != EMBED_DIM || values.iter().any(|value| !value.is_finite()) {
        return Err(MediaError::InvalidEmbedding);
    }
    let mut out = [0.0_f32; EMBED_DIM];
    out.copy_from_slice(values);
    Ok(out)
}

fn embed_bytes(seed: u64, input: &[u8]) -> [f32; EMBED_DIM] {
    let mut out = [0.0_f32; EMBED_DIM];
    if input.is_empty() {
        out[0] = 1.0;
        return out;
    }
    let mut window = [0_u8; 3];
    for n in 1..=3 {
        let mut i = 0;
        while i + n <= input.len() {
            window[..n].copy_from_slice(&input[i..i + n]);
            let hash = wyhash::hash(
                seed.saturating_add(u64::try_from(n).unwrap_or(3)),
                &window[..n],
            );
            #[allow(clippy::cast_possible_truncation)]
            let bucket = (hash % EMBED_DIM as u64) as usize;
            let weight = [0.0_f32, 1.0, 2.0, 3.0][n];
            out[bucket] += if (hash >> 63) & 1 == 0 {
                weight
            } else {
                -weight
            };
            i += 1;
        }
    }
    l2_normalize(out)
}

fn l2_normalize(mut out: [f32; EMBED_DIM]) -> [f32; EMBED_DIM] {
    let norm: f32 = out.iter().map(|value| value * value).sum();
    if norm == 0.0 {
        out[0] = 1.0;
        return out;
    }
    let scale = norm.sqrt();
    for value in &mut out {
        *value /= scale;
    }
    out
}

fn mean_pair(left: [f32; EMBED_DIM], right: [f32; EMBED_DIM]) -> [f32; EMBED_DIM] {
    let mut out = [0.0_f32; EMBED_DIM];
    for (slot, (a, b)) in out.iter_mut().zip(left.iter().zip(right)) {
        *slot = (a + b) * 0.5;
    }
    out
}

fn quantize_pcm(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .map(|sample| {
            let clamped = sample.clamp(-1.0, 1.0);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                ((clamped + 1.0) * 127.5) as u8
            }
        })
        .collect()
}

fn mean_abs_delta(left: &[u8], right: &[u8]) -> f32 {
    if left.is_empty() {
        return 0.0;
    }
    let mut total = 0_u32;
    for (a, b) in left.iter().zip(right) {
        total += u32::from(a.abs_diff(*b));
    }
    f32_from_count(total) / f32_from_count(left.len())
}

fn image_stats(width: u32, height: u32, pixels: &[u8]) -> Value {
    let count = pixels.len().max(1);
    let sum: u32 = pixels.iter().map(|pixel| u32::from(*pixel)).sum();
    let mean = f32_from_count(sum) / f32_from_count(count);
    let variance = pixels
        .iter()
        .map(|pixel| {
            let delta = f32::from(*pixel) - mean;
            delta * delta
        })
        .sum::<f32>()
        / f32_from_count(count);
    let mut histogram = [0_u32; 8];
    for pixel in pixels {
        histogram[usize::from(*pixel >> 5)] += 1;
    }
    let gradient = horizontal_energy(width, height, pixels);
    json!({
        "width": width,
        "height": height,
        "mean": mean,
        "std": variance.sqrt(),
        "min": pixels.iter().copied().min().unwrap_or(0),
        "max": pixels.iter().copied().max().unwrap_or(0),
        "gradient_energy": gradient,
        "histogram8": histogram,
    })
}

fn horizontal_energy(width: u32, height: u32, pixels: &[u8]) -> f32 {
    let Ok(width_usize) = usize::try_from(width) else {
        return 0.0;
    };
    let Ok(height_usize) = usize::try_from(height) else {
        return 0.0;
    };
    if width_usize < 2 {
        return 0.0;
    }
    let mut total = 0_u32;
    let mut pairs = 0_u32;
    for row in 0..height_usize {
        let start = row * width_usize;
        for col in 0..width_usize.saturating_sub(1) {
            total += u32::from(pixels[start + col].abs_diff(pixels[start + col + 1]));
            pairs += 1;
        }
    }
    if pairs == 0 {
        0.0
    } else {
        f32_from_count(total) / f32_from_count(pairs)
    }
}

fn audio_stats(sample_rate: u32, samples: &[f32]) -> Value {
    let count = f32_from_count(samples.len());
    let rms = (samples.iter().map(|sample| sample * sample).sum::<f32>() / count).sqrt();
    let peak = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0_f32, f32::max);
    let mut crossings = 0_u32;
    for window in samples.windows(2) {
        if window[0].signum() != window[1].signum() && (window[0] != 0.0 || window[1] != 0.0) {
            crossings += 1;
        }
    }
    let zcr = f32_from_count(crossings) / count.max(1.0);
    let mut bands = [0.0_f32; 8];
    let band_len = samples.len().div_ceil(8).max(1);
    for (index, sample) in samples.iter().enumerate() {
        bands[index / band_len] += sample.abs();
    }
    for band in &mut bands {
        *band /= f32_from_count(band_len);
    }
    json!({
        "sample_rate": sample_rate,
        "samples": samples.len(),
        "rms": rms,
        "peak": peak,
        "zero_crossing_rate": zcr,
        "band_energy8": bands,
    })
}

fn f32_from_count(count: impl TryInto<u32>) -> f32 {
    let value = count.try_into().unwrap_or(u32::MAX);
    // Counts here are small (pixels, frames, samples) and already bounded.
    #[allow(clippy::cast_precision_loss)]
    {
        value as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(values: &[f32]) -> f32 {
        values.iter().map(|value| value * value).sum::<f32>().sqrt()
    }

    #[test]
    fn image_analysis_is_deterministic_and_unit_length() {
        let pixels = (0u8..=15).collect::<Vec<_>>();
        let first = analyze_image(4, 4, &pixels).expect("image");
        let second = analyze_image(4, 4, &pixels).expect("image");
        assert_eq!(first, second);
        assert_eq!(first.kind, "image");
        assert_eq!(first.engine, ENGINE);
        assert!(first.disclosure.contains("Not a neural vision model"));
        assert_eq!(first.embedding.len(), VISION_DIM);
        assert!((norm(&first.embedding) - 1.0).abs() < 0.001);
    }

    #[test]
    fn image_rejects_bad_geometry_and_counts() {
        assert_eq!(
            analyze_image(0, 4, &[0, 1, 2, 3]).unwrap_err(),
            MediaError::InvalidGeometry
        );
        assert_eq!(
            analyze_image(4, 4, &[1, 2, 3]).unwrap_err(),
            MediaError::PixelCountMismatch
        );
        assert_eq!(analyze_image(2, 2, &[]).unwrap_err(), MediaError::Empty);
    }

    #[test]
    fn video_reports_motion_between_frames() {
        let dark = vec![0_u8; 16];
        let bright = vec![255_u8; 16];
        let report = analyze_video(4, 4, &[dark, bright]).expect("video");
        assert_eq!(report.kind, "video");
        assert_eq!(report.stats["frames"], 2);
        assert!(report.stats["mean_motion"].as_f64().expect("motion") > 100.0);
        assert_eq!(report.stats["scene_cuts"], 1);
        assert!((norm(&report.embedding) - 1.0).abs() < 0.001);
    }

    #[test]
    fn video_rejects_too_many_or_mismatched_frames() {
        let frame = vec![1_u8; 4];
        let too_many = vec![frame.clone(); 9];
        assert_eq!(
            analyze_video(2, 2, &too_many).unwrap_err(),
            MediaError::TooManyFrames
        );
        assert_eq!(
            analyze_video(2, 2, &[vec![1, 2]]).unwrap_err(),
            MediaError::FrameSizeMismatch
        );
    }

    #[test]
    fn audio_analysis_is_deterministic() {
        let samples: Vec<f32> = (0_u8..32)
            .map(|index| (f32::from(index) / 16.0) - 1.0)
            .collect();
        let first = analyze_audio(16_000, &samples).expect("audio");
        let second = analyze_audio(16_000, &samples).expect("audio");
        assert_eq!(first, second);
        assert_eq!(first.kind, "audio");
        assert!(first.disclosure.contains("Not a speech recognizer"));
        assert!((norm(&first.embedding) - 1.0).abs() < 0.001);
        assert_eq!(first.stats["samples"], 32);
    }

    #[test]
    fn audio_rejects_non_finite_and_bad_rates() {
        assert_eq!(
            analyze_audio(100, &[0.1]).unwrap_err(),
            MediaError::InvalidSampleRate
        );
        assert_eq!(
            analyze_audio(16_000, &[f32::NAN]).unwrap_err(),
            MediaError::NonFiniteSample
        );
        assert_eq!(
            analyze_audio(16_000, &[]).unwrap_err(),
            MediaError::InvalidSampleCount
        );
    }

    #[test]
    fn voice_embeds_transcript_and_optional_audio() {
        let text_only = analyze_voice("hello studio", None, None).expect("voice");
        assert_eq!(text_only.kind, "voice");
        assert_eq!(text_only.stats["has_audio"], false);
        assert_eq!(text_only.stats["transcript_words"], 2);
        let samples = [0.25_f32, -0.5, 0.1, 0.0];
        let with_audio = analyze_voice("hello studio", Some(16_000), Some(&samples)).expect("mix");
        assert_eq!(with_audio.stats["has_audio"], true);
        assert_ne!(text_only.embedding, with_audio.embedding);
    }

    #[test]
    fn fuse_mean_pools_present_parts_only() {
        let vision = analyze_image(2, 2, &[0, 64, 128, 255])
            .expect("vision")
            .embedding;
        let audio = analyze_audio(16_000, &[0.1, 0.2, -0.1, 0.0])
            .expect("audio")
            .embedding;
        let fused = fuse_embeddings(Some(&vision), Some(&audio), None).expect("fuse");
        assert_eq!(fused.kind, "fuse");
        assert_eq!(fused.stats["count"], 2);
        assert_eq!(fused.stats["parts"], json!(["vision", "audio"]));
        assert!((norm(&fused.embedding) - 1.0).abs() < 0.001);
        assert_eq!(
            fuse_embeddings(None, None, None).unwrap_err(),
            MediaError::NothingToFuse
        );
        assert_eq!(
            fuse_embeddings(Some(&[1.0]), None, None).unwrap_err(),
            MediaError::InvalidEmbedding
        );
    }

    #[test]
    fn error_tokens_are_stable() {
        assert_eq!(MediaError::Empty.as_str(), "empty media payload");
        assert_eq!(MediaError::Empty.status(), 400);
    }
}
