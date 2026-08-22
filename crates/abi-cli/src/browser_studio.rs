//! Loopback multimodal browser studio (`abi agent browser --studio`).
//!
//! Serves a single-page capture UI and JSON analyze/fuse routes on
//! `127.0.0.1` only. Analysis is the deterministic [`abi_ai`] multimodal
//! processors — not a neural vision, STT, or TTS model.

use std::io::{self, Write as _};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};

use abi_ai::{
    AnalysisReport, MAX_FRAMES, MAX_SAMPLES, MAX_SIDE, MediaError, analyze_audio, analyze_image,
    analyze_video, analyze_voice, fuse_embeddings,
};
use abi_foundation::http::{
    MAX_REQUEST_SIZE, ReadResult, find_body, has_bearer_token, header_value, read_request,
    reason_phrase, write_all, write_unauthorized,
};
use serde_json::{Value, json};

use crate::app::Outcome;

const HTML: &str = include_str!("browser_studio.html");
const STUDIO_USAGE: &str = "usage: abi agent browser --studio [--port <port>] [--once]\n";
const DEFAULT_PORT: u16 = 8095;
const TOKEN_ENV: &str = "ABI_BROWSER_STUDIO_TOKEN";

/// One studio HTTP response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StudioResponse {
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
}

impl StudioResponse {
    fn html(body: &str) -> Self {
        Self {
            status: 200,
            content_type: "text/html; charset=utf-8",
            body: body.as_bytes().to_vec(),
        }
    }

    fn json(status: u16, value: &Value) -> Self {
        Self {
            status,
            content_type: "application/json",
            body: value.to_string().into_bytes(),
        }
    }

    fn error(status: u16, message: impl Into<String>) -> Self {
        Self::json(status, &json!({"error": message.into()}))
    }

    fn report(report: &AnalysisReport) -> Self {
        Self::json(
            200,
            &json!({
                "kind": report.kind,
                "engine": report.engine,
                "disclosure": report.disclosure,
                "embedding": report.embedding,
                "stats": report.stats,
            }),
        )
    }

    fn from_media(result: Result<AnalysisReport, MediaError>) -> Self {
        match result {
            Ok(report) => Self::report(&report),
            Err(error) => Self::error(error.status(), error.as_str()),
        }
    }
}

/// Capability document for the studio surface.
#[must_use]
pub(crate) fn capabilities() -> Value {
    json!({
        "type": "abi.browser.studio",
        "engine": "deterministic-local",
        "loopback": true,
        "embedded_browser": false,
        "neural_vision": false,
        "neural_stt": false,
        "neural_tts": false,
        "client_stt": "WebSpeechRecognition",
        "client_tts": "speechSynthesis",
        "delegation_hint": "external-mcp-playwright",
        "modalities": ["image", "video", "audio", "voice"],
        "limits": {
            "max_side": MAX_SIDE,
            "max_frames": MAX_FRAMES,
            "max_samples": MAX_SAMPLES,
        },
    })
}

/// Route one already-framed studio request.
#[must_use]
pub(crate) fn route(method: &str, path: &str, body: &[u8]) -> StudioResponse {
    let path = path.split('?').next().unwrap_or(path);
    match (method, path) {
        ("GET", "/" | "/index.html") => StudioResponse::html(HTML),
        ("GET", "/health") => StudioResponse::json(
            200,
            &json!({
                "status": "ok",
                "surface": "browser-studio",
                "engine": "deterministic-local",
            }),
        ),
        ("GET", "/capabilities") => StudioResponse::json(200, &capabilities()),
        ("POST", "/analyze") => route_analyze(body),
        ("POST", "/fuse") => route_fuse(body),
        _ => StudioResponse::error(404, format!("no route for {method} {path}")),
    }
}

fn route_analyze(body: &[u8]) -> StudioResponse {
    let object = match parsed_object(body) {
        Ok(object) => object,
        Err(response) => return response,
    };
    let kind = object.get("kind").and_then(Value::as_str).unwrap_or("");
    match kind {
        "image" => {
            let Some((width, height, pixels)) = image_fields(&object) else {
                return StudioResponse::error(400, MediaError::InvalidGeometry.as_str());
            };
            StudioResponse::from_media(analyze_image(width, height, &pixels))
        }
        "video" => route_video(&object),
        "audio" => {
            let Some((rate, samples)) = audio_fields(&object) else {
                return StudioResponse::error(400, MediaError::InvalidSampleCount.as_str());
            };
            StudioResponse::from_media(analyze_audio(rate, &samples))
        }
        "voice" => {
            let transcript = object
                .get("transcript")
                .and_then(Value::as_str)
                .unwrap_or("");
            let rate = object
                .get("sample_rate")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok());
            let samples = object.get("samples").and_then(parse_samples);
            StudioResponse::from_media(analyze_voice(transcript, rate, samples.as_deref()))
        }
        _ => StudioResponse::error(400, "kind must be image, video, audio, or voice"),
    }
}

fn route_video(object: &Value) -> StudioResponse {
    let width = object
        .get("width")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    let height = object
        .get("height")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    let Some(width) = width else {
        return StudioResponse::error(400, MediaError::InvalidGeometry.as_str());
    };
    let Some(height) = height else {
        return StudioResponse::error(400, MediaError::InvalidGeometry.as_str());
    };
    let Some(frames_value) = object.get("frames").and_then(Value::as_array) else {
        return StudioResponse::error(400, MediaError::Empty.as_str());
    };
    let mut frames = Vec::new();
    for frame in frames_value {
        let Some(pixels) = parse_pixels(frame) else {
            return StudioResponse::error(400, MediaError::FrameSizeMismatch.as_str());
        };
        frames.push(pixels);
    }
    StudioResponse::from_media(analyze_video(width, height, &frames))
}

fn parsed_object(body: &[u8]) -> Result<Value, StudioResponse> {
    if body.is_empty() {
        return Err(StudioResponse::error(400, "JSON object required"));
    }
    match serde_json::from_slice::<Value>(body) {
        Ok(Value::Object(map)) => Ok(Value::Object(map)),
        Ok(_) => Err(StudioResponse::error(400, "JSON object required")),
        Err(_) => Err(StudioResponse::error(400, "invalid JSON")),
    }
}

fn image_fields(object: &Value) -> Option<(u32, u32, Vec<u8>)> {
    let width = u32::try_from(object.get("width")?.as_u64()?).ok()?;
    let height = u32::try_from(object.get("height")?.as_u64()?).ok()?;
    let pixels = parse_pixels(object.get("pixels")?)?;
    Some((width, height, pixels))
}

fn audio_fields(object: &Value) -> Option<(u32, Vec<f32>)> {
    let rate = u32::try_from(object.get("sample_rate")?.as_u64()?).ok()?;
    let samples = parse_samples(object.get("samples")?)?;
    Some((rate, samples))
}

fn parse_pixels(value: &Value) -> Option<Vec<u8>> {
    let items = value.as_array()?;
    let mut pixels = Vec::with_capacity(items.len());
    for item in items {
        let number = item.as_u64()?;
        if number > 255 {
            return None;
        }
        pixels.push(u8::try_from(number).ok()?);
    }
    Some(pixels)
}

fn parse_samples(value: &Value) -> Option<Vec<f32>> {
    let items = value.as_array()?;
    let mut samples = Vec::with_capacity(items.len());
    for item in items {
        let number = item.as_f64()?;
        if !number.is_finite() {
            return None;
        }
        #[allow(clippy::cast_possible_truncation)]
        samples.push(number as f32);
    }
    Some(samples)
}

fn owned_embedding(value: Option<&Value>) -> Option<Vec<f32>> {
    let items = value?.as_array()?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let number = item.as_f64()?;
        if !number.is_finite() {
            return None;
        }
        #[allow(clippy::cast_possible_truncation)]
        out.push(number as f32);
    }
    Some(out)
}

fn route_fuse(body: &[u8]) -> StudioResponse {
    let object = match parsed_object(body) {
        Ok(object) => object,
        Err(response) => return response,
    };
    let vision = owned_embedding(object.get("vision"));
    let audio = owned_embedding(object.get("audio"));
    let video = owned_embedding(object.get("video"));
    StudioResponse::from_media(fuse_embeddings(
        vision.as_deref(),
        audio.as_deref(),
        video.as_deref(),
    ))
}

fn token_from_env() -> Option<String> {
    std::env::var(TOKEN_ENV)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn is_loopback_host(host: &str) -> bool {
    let hostname = host.rsplit_once(':').map_or(host, |(name, _)| name);
    matches!(hostname, "127.0.0.1" | "localhost" | "[::1]")
}

fn is_loopback_origin(origin: &str) -> bool {
    origin.starts_with("http://127.0.0.1")
        || origin.starts_with("http://localhost")
        || origin.starts_with("http://[::1]")
}

fn host_allowed(raw: &str) -> bool {
    header_value(raw, "Host").is_some_and(is_loopback_host)
}

fn origin_allowed(raw: &str) -> bool {
    header_value(raw, "Origin").is_none_or(is_loopback_origin)
}

fn request_target(raw: &str) -> Option<(&str, &str)> {
    let request_line = raw.lines().next()?.trim_end_matches('\r');
    let mut fields = request_line.split(' ');
    Some((fields.next()?, fields.next()?))
}

fn write_response(stream: &mut TcpStream, response: &StudioResponse) -> io::Result<()> {
    let phrase = if response.status == 413 {
        "Payload Too Large"
    } else {
        reason_phrase(response.status)
    };
    let header = format!(
        "HTTP/1.1 {} {phrase}\r\nContent-Type: {}\r\nContent-Length: {}\r\n\
         Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self' blob:; connect-src 'self'; img-src 'self' blob: data:\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Referrer-Policy: no-referrer\r\n\
         Connection: close\r\n\r\n",
        response.status,
        response.content_type,
        response.body.len()
    );
    write_all(stream, header.as_bytes())?;
    write_all(stream, &response.body)
}

fn handle_connection(stream: &mut TcpStream, token: Option<&str>) -> io::Result<()> {
    let raw = match read_request(stream, MAX_REQUEST_SIZE) {
        ReadResult::Empty => return Ok(()),
        ReadResult::Incomplete => {
            return write_response(stream, &StudioResponse::error(400, "incomplete request"));
        }
        ReadResult::TooLarge => {
            return write_response(stream, &StudioResponse::error(413, "request too large"));
        }
        ReadResult::Request(raw) => raw,
    };
    let Ok(raw_text) = std::str::from_utf8(&raw) else {
        return write_response(stream, &StudioResponse::error(400, "incomplete request"));
    };
    if !host_allowed(raw_text) || !origin_allowed(raw_text) {
        return write_response(stream, &StudioResponse::error(403, "loopback origin only"));
    }
    let Some((method, path)) = request_target(raw_text) else {
        return Ok(());
    };
    let is_get_doc = method == "GET" && matches!(path.split('?').next(), Some("/" | "/index.html"));
    if let Some(expected) = token
        && !is_get_doc
        && !has_bearer_token(raw_text, expected)
    {
        return write_unauthorized(stream, "authorization required");
    }
    let body = find_body(&raw).unwrap_or_default();
    write_response(stream, &route(method, path, body))
}

/// Bind and serve the studio.
pub(crate) fn run_from_args(args: &[String]) -> Outcome {
    let mut port = DEFAULT_PORT;
    let mut once = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--studio" => {}
            "--once" => once = true,
            "--port" => {
                i += 1;
                let Some(raw) = args.get(i) else {
                    return Outcome::stderr(format!("error: {STUDIO_USAGE}"), 2);
                };
                match raw.parse::<u16>() {
                    Ok(value) => port = value,
                    Err(_) => return Outcome::stderr(format!("error: {STUDIO_USAGE}"), 2),
                }
            }
            other if other.starts_with('-') => {
                return Outcome::stderr(format!("error: {STUDIO_USAGE}"), 2);
            }
            _ => {}
        }
        i += 1;
    }

    let listener = match TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port)) {
        Ok(listener) => listener,
        Err(error) => {
            return Outcome::stderr(format!("error: browser studio bind failed: {error}\n"), 2);
        }
    };
    let bound = match listener.local_addr() {
        Ok(addr) => addr.port(),
        Err(error) => {
            return Outcome::stderr(format!("error: browser studio bind failed: {error}\n"), 2);
        }
    };
    let token = token_from_env();
    let auth = if token.is_some() {
        "auth=bearer"
    } else {
        "auth=off"
    };
    let url = format!("http://127.0.0.1:{bound}/");
    let banner = format!(
        "browser-studio url={url}\nbrowser-studio mode=loopback\nbrowser-studio engine=deterministic-local\nembedded_browser=false\nneural_vision=false\nneural_stt=false\nneural_tts=false\n{auth}; POST /analyze /fuse, GET / /health /capabilities; loopback only\n"
    );
    eprint!("{banner}");
    let _ = io::stderr().flush();

    if once {
        match listener.accept() {
            Ok((mut stream, _)) => {
                if let Err(error) = handle_connection(&mut stream, token.as_deref()) {
                    return Outcome::stderr(format!("error: browser studio: {error}\n"), 2);
                }
            }
            Err(error) => {
                return Outcome::stderr(format!("error: browser studio: {error}\n"), 2);
            }
        }
        return Outcome {
            stdout: banner,
            stderr: String::new(),
            exit_code: 0,
        };
    }

    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let stop_flag = std::sync::Arc::clone(&stop);
    let _ = ctrlc::set_handler(move || {
        stop_flag.store(true, Ordering::SeqCst);
    });
    while !stop.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((mut stream, _)) => {
                if let Err(error) = handle_connection(&mut stream, token.as_deref())
                    && !stop.load(Ordering::SeqCst)
                {
                    eprintln!("browser studio: {error}");
                }
            }
            Err(error) => {
                if stop.load(Ordering::SeqCst) {
                    break;
                }
                eprintln!("browser studio: {error}");
            }
        }
    }
    Outcome {
        stdout: banner,
        stderr: String::new(),
        exit_code: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body_json(response: &StudioResponse) -> Value {
        serde_json::from_slice(&response.body).expect("json body")
    }

    #[test]
    fn health_and_capabilities_are_claim_honest() {
        let health = route("GET", "/health", b"");
        assert_eq!(health.status, 200);
        let health_json = body_json(&health);
        assert_eq!(health_json["status"], "ok");
        assert_eq!(health_json["surface"], "browser-studio");

        let caps = route("GET", "/capabilities", b"");
        let value = body_json(&caps);
        assert_eq!(value["type"], "abi.browser.studio");
        assert_eq!(value["neural_vision"], false);
        assert_eq!(value["neural_stt"], false);
        assert_eq!(value["neural_tts"], false);
        assert_eq!(value["embedded_browser"], false);
        assert_eq!(value["loopback"], true);
        assert_eq!(value["limits"]["max_side"], MAX_SIDE);
    }

    #[test]
    fn index_serves_the_studio_page() {
        let page = route("GET", "/", b"");
        assert_eq!(page.status, 200);
        assert_eq!(page.content_type, "text/html; charset=utf-8");
        let html = String::from_utf8(page.body.clone()).expect("utf8");
        assert!(html.contains("ABI browser studio"));
        assert!(html.contains("not a neural vision"));
        assert!(html.contains("getUserMedia"));
    }

    #[test]
    fn analyze_image_and_fuse_round_trip() {
        let image = route(
            "POST",
            "/analyze",
            br#"{"kind":"image","width":2,"height":2,"pixels":[0,64,128,255]}"#,
        );
        assert_eq!(image.status, 200);
        let image_json = body_json(&image);
        assert_eq!(image_json["kind"], "image");
        assert_eq!(image_json["engine"], "deterministic-local");
        assert!(
            image_json["disclosure"]
                .as_str()
                .expect("disclosure")
                .contains("Not a neural vision model")
        );
        let embedding = image_json["embedding"].clone();
        let fuse_body = json!({"vision": embedding, "audio": null, "video": null}).to_string();
        let fused = route("POST", "/fuse", fuse_body.as_bytes());
        assert_eq!(fused.status, 200);
        assert_eq!(body_json(&fused)["kind"], "fuse");
    }

    #[test]
    fn analyze_audio_and_voice() {
        let audio = route(
            "POST",
            "/analyze",
            br#"{"kind":"audio","sample_rate":16000,"samples":[0.1,-0.2,0.0,0.3]}"#,
        );
        assert_eq!(audio.status, 200);
        assert_eq!(body_json(&audio)["kind"], "audio");

        let voice = route(
            "POST",
            "/analyze",
            br#"{"kind":"voice","transcript":"hello studio"}"#,
        );
        assert_eq!(voice.status, 200);
        assert_eq!(body_json(&voice)["kind"], "voice");
        assert_eq!(body_json(&voice)["stats"]["has_audio"], false);
    }

    #[test]
    fn analyze_video_and_unknown_routes() {
        let video = route(
            "POST",
            "/analyze",
            br#"{"kind":"video","width":2,"height":2,"frames":[[0,0,0,0],[255,255,255,255]]}"#,
        );
        assert_eq!(video.status, 200);
        assert_eq!(body_json(&video)["kind"], "video");

        let missing = route("DELETE", "/nope", b"");
        assert_eq!(missing.status, 404);
        let bad_kind = route("POST", "/analyze", br#"{"kind":"smell"}"#);
        assert_eq!(bad_kind.status, 400);
    }

    #[test]
    fn loopback_host_and_origin_checks() {
        assert!(is_loopback_host("127.0.0.1:8095"));
        assert!(is_loopback_host("localhost"));
        assert!(!is_loopback_host("evil.example"));
        assert!(is_loopback_origin("http://127.0.0.1:8095"));
        assert!(!is_loopback_origin("https://evil.example"));
    }
}
