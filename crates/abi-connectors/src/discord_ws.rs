//! Minimal WebSocket framing for the Discord gateway reference client.
//!
//! Ported from `src/connectors/discord_ws_client.zig`. Implements the HTTP
//! upgrade request, client→server masked frames, and server→client frame
//! parse. **TLS is not linked** — a live path to `wss://gateway.discord.gg`
//! requires a TLS-terminating proxy (disclosed honesty boundary).

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use base64::Engine as _;
use std::io::{Read, Write};

/// Discord gateway path used in the upgrade request.
pub const GATEWAY_PATH: &str = "/?v=10&encoding=json";
/// Default Discord gateway host.
pub const GATEWAY_HOST: &str = "gateway.discord.gg";

/// Errors from WebSocket framing / handshake.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsError {
    /// Handshake response was not HTTP 101.
    HandshakeFailed,
    /// Peer closed the socket.
    SocketClosed,
    /// Frame was malformed or incomplete.
    BadFrame,
    /// I/O failure description.
    Io(String),
}

impl std::fmt::Display for WsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HandshakeFailed => write!(f, "websocket handshake failed"),
            Self::SocketClosed => write!(f, "socket closed"),
            Self::BadFrame => write!(f, "malformed websocket frame"),
            Self::Io(msg) => write!(f, "websocket io: {msg}"),
        }
    }
}

impl std::error::Error for WsError {}

/// Build the HTTP/1.1 upgrade request for a Discord-style gateway.
#[must_use]
pub fn build_handshake_request(host: &str, path: &str, key_b64: &str) -> String {
    format!(
        "GET {path} HTTP/1.1\r\n\
         Host: {host}\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Key: {key_b64}\r\n\
         Sec-WebSocket-Version: 13\r\n\r\n"
    )
}

/// Encode a client→server text frame with a fixed 4-byte mask (testable).
#[must_use]
pub fn encode_masked_text_frame(payload: &[u8], mask: [u8; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 + 8 + 4 + payload.len());
    out.push(0x81); // FIN + text opcode
    let len = payload.len();
    if len < 126 {
        out.push(0x80 | u8::try_from(len).unwrap_or(0));
    } else if len <= 0xFFFF {
        out.push(0x80 | 0x7e);
        out.extend_from_slice(&(u16::try_from(len).unwrap_or(0)).to_be_bytes());
    } else {
        out.push(0x80 | 0x7f);
        out.extend_from_slice(&(len as u64).to_be_bytes());
    }
    out.extend_from_slice(&mask);
    for (i, b) in payload.iter().enumerate() {
        out.push(b ^ mask[i % 4]);
    }
    out
}

/// A decoded server→client frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Frame {
    /// Text payload.
    Text(Vec<u8>),
    /// Binary payload.
    Binary(Vec<u8>),
    /// Close.
    Close,
    /// Ping payload.
    Ping(Vec<u8>),
    /// Pong payload.
    Pong(Vec<u8>),
}

/// Try to parse one complete WebSocket frame from the front of `buf`.
///
/// On success returns `(frame, bytes_consumed)`. Server frames are unmasked.
pub fn try_parse_frame(buf: &[u8]) -> Result<Option<(Frame, usize)>, WsError> {
    if buf.len() < 2 {
        return Ok(None);
    }
    let opcode = buf[0] & 0x0F;
    let masked = (buf[1] & 0x80) != 0;
    let mut len = usize::from(buf[1] & 0x7F);
    let mut offset = 2_usize;
    if len == 126 {
        if buf.len() < 4 {
            return Ok(None);
        }
        len = usize::from(u16::from_be_bytes([buf[2], buf[3]]));
        offset = 4;
    } else if len == 127 {
        if buf.len() < 10 {
            return Ok(None);
        }
        let n = u64::from_be_bytes(buf[2..10].try_into().unwrap());
        len = usize::try_from(n).map_err(|_| WsError::BadFrame)?;
        offset = 10;
    }
    let mask_len = if masked { 4 } else { 0 };
    if buf.len() < offset + mask_len + len {
        return Ok(None);
    }
    let mask = if masked {
        [
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
        ]
    } else {
        [0; 4]
    };
    if masked {
        offset += 4;
    }
    let mut payload = buf[offset..offset + len].to_vec();
    if masked {
        for (i, b) in payload.iter_mut().enumerate() {
            *b ^= mask[i % 4];
        }
    }
    let frame = match opcode {
        0x1 => Frame::Text(payload),
        0x2 => Frame::Binary(payload),
        0x8 => Frame::Close,
        0x9 => Frame::Ping(payload),
        0xA => Frame::Pong(payload),
        _ => return Err(WsError::BadFrame),
    };
    Ok(Some((frame, offset + len)))
}

/// Encode a random-looking base64 WebSocket key from 16 seed bytes.
#[must_use]
pub fn key_b64_from_seed(seed: [u8; 16]) -> String {
    base64::engine::general_purpose::STANDARD.encode(seed)
}

/// Perform the HTTP upgrade over an already-connected stream (plaintext).
///
/// Honesty: Discord requires TLS; callers must put a TLS proxy in front.
pub fn handshake_on_stream<S: Read + Write>(
    stream: &mut S,
    host: &str,
    path: &str,
    key_seed: [u8; 16],
) -> Result<(), WsError> {
    let key = key_b64_from_seed(key_seed);
    let request = build_handshake_request(host, path, &key);
    stream
        .write_all(request.as_bytes())
        .map_err(|e| WsError::Io(e.to_string()))?;
    let mut buf = Vec::new();
    let mut tmp = [0_u8; 1024];
    loop {
        let n = stream
            .read(&mut tmp)
            .map_err(|e| WsError::Io(e.to_string()))?;
        if n == 0 {
            return Err(WsError::SocketClosed);
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_header_end(&buf) {
            let headers = std::str::from_utf8(&buf[..pos]).unwrap_or("");
            if !headers.contains("101") {
                return Err(WsError::HandshakeFailed);
            }
            return Ok(());
        }
        if buf.len() > 16_384 {
            return Err(WsError::HandshakeFailed);
        }
    }
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handshake_request_carries_upgrade_headers() {
        let req = build_handshake_request(GATEWAY_HOST, GATEWAY_PATH, "dGhlIHNhbXBsZSBub25jZQ==");
        assert!(req.contains("Upgrade: websocket"));
        assert!(req.contains("Sec-WebSocket-Version: 13"));
        assert!(req.contains("Host: gateway.discord.gg"));
        assert!(req.contains("dGhlIHNhbXBsZSBub25jZQ=="));
    }

    #[test]
    fn masked_text_frame_round_trips_payload() {
        let payload = br#"{"op":1,"d":null}"#;
        let mask = [0x11, 0x22, 0x33, 0x44];
        let frame = encode_masked_text_frame(payload, mask);
        // Strip mask for server-style parse: rebuild as unmasked server frame.
        let mut server = vec![0x81, u8::try_from(payload.len()).unwrap()];
        server.extend_from_slice(payload);
        let (parsed, n) = try_parse_frame(&server).unwrap().unwrap();
        assert_eq!(n, server.len());
        assert_eq!(parsed, Frame::Text(payload.to_vec()));
        // Masked frame is longer by 4 mask bytes.
        assert_eq!(frame.len(), 2 + 4 + payload.len());
    }

    #[test]
    fn incomplete_frame_returns_none() {
        assert!(try_parse_frame(&[0x81]).unwrap().is_none());
        assert!(
            try_parse_frame(&[0x81, 0x05, b'h', b'i'])
                .unwrap()
                .is_none()
        );
    }
}
