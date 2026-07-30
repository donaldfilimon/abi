//! Soul-prompt layout and bootstrap for persona neural routing.
//!
//! Ported from `src/features/ai/soul_layout.zig`. Parses a JSON array of
//! labeled records, optionally derives 3-D points from the label text, and
//! warms a [`crate::point_net::PointNeuralNetwork`] for [`crate::route_with_soul`].

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::point_net::{Point, PointNeuralNetwork, TopologyReport};

/// One labeled soul record.
#[derive(Debug, Clone, PartialEq)]
pub struct SoulRecord {
    /// Human-readable label.
    pub label: String,
    /// Geometry used to seed the network.
    pub point: Point,
}

/// Parsed soul layout.
#[derive(Debug, Clone, PartialEq)]
pub struct SoulLayout {
    /// Ordered records from the prompt file.
    pub records: Vec<SoulRecord>,
}

/// Errors while parsing a soul prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SoulError {
    /// Top-level JSON is not an array.
    InvalidFormat,
    /// Array is empty.
    EmptyPrompt,
    /// An array entry is not an object.
    InvalidRecord,
    /// Missing or non-string `label`.
    MissingLabel,
    /// Underlying I/O or JSON parse failure.
    Parse(String),
}

impl std::fmt::Display for SoulError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat => write!(f, "invalid soul format (expected JSON array)"),
            Self::EmptyPrompt => write!(f, "empty soul prompt"),
            Self::InvalidRecord => write!(f, "invalid soul record"),
            Self::MissingLabel => write!(f, "soul record missing label"),
            Self::Parse(msg) => write!(f, "soul parse error: {msg}"),
        }
    }
}

impl std::error::Error for SoulError {}

fn json_f32(value: &serde_json::Value) -> Option<f32> {
    match value {
        serde_json::Value::Number(n) => n.as_f64().map(|v| v as f32),
        _ => None,
    }
}

impl SoulLayout {
    /// Parse a soul prompt from JSON text.
    ///
    /// Expected shape: `[{"label":"...", "x":.., "y":.., "z":.., "value":..}, ...]`.
    /// Missing coordinates fall back to [`Point::from_text`] on the label.
    pub fn from_json(json_text: &str) -> Result<Self, SoulError> {
        let parsed: serde_json::Value =
            serde_json::from_str(json_text).map_err(|e| SoulError::Parse(e.to_string()))?;
        let arr = parsed.as_array().ok_or(SoulError::InvalidFormat)?;
        if arr.is_empty() {
            return Err(SoulError::EmptyPrompt);
        }
        let mut records = Vec::with_capacity(arr.len());
        for item in arr {
            let obj = item.as_object().ok_or(SoulError::InvalidRecord)?;
            let label = obj
                .get("label")
                .and_then(serde_json::Value::as_str)
                .ok_or(SoulError::MissingLabel)?
                .to_string();
            let point = if let (Some(x), Some(y), Some(z)) = (
                obj.get("x").and_then(json_f32),
                obj.get("y").and_then(json_f32),
                obj.get("z").and_then(json_f32),
            ) {
                Point {
                    x,
                    y,
                    z,
                    value: obj.get("value").and_then(json_f32).unwrap_or(1.0),
                }
            } else {
                Point::from_text(&label)
            };
            records.push(SoulRecord { label, point });
        }
        Ok(Self { records })
    }

    /// Warm `net` on this layout (autoencoder targets = point coordinates).
    ///
    /// Unlike Zig's optional WDBX store writes during bootstrap, this path is
    /// pure: it only trains the network. That keeps soul routing hermetic for
    /// CLI tests and avoids touching `~/.abi/`.
    pub fn bootstrap(&self, net: &mut PointNeuralNetwork) -> Result<(f32, TopologyReport), String> {
        if self.records.is_empty() {
            return Err("empty soul layout".into());
        }
        if net.output_size() != 3 {
            return Err("soul bootstrap requires a 3-output network".into());
        }
        let points: Vec<Point> = self.records.iter().map(|r| r.point).collect();
        let targets: Vec<Vec<f32>> = points.iter().map(|p| p.to_array().to_vec()).collect();
        let loss = net.train(&points, &targets, 100)?;
        let report = net.optimize_topology(&points);
        Ok((loss, report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_json_parses_label_only_and_derives_point() {
        let layout = SoulLayout::from_json(r#"[{"label":"hello world"}]"#).unwrap();
        assert_eq!(layout.records.len(), 1);
        assert_eq!(layout.records[0].label, "hello world");
        assert!(layout.records[0].point.x > 0.0);
        assert!((layout.records[0].point.value - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn from_json_honors_explicit_coordinates() {
        let layout =
            SoulLayout::from_json(r#"[{"label":"a","x":1,"y":2,"z":3,"value":0.5}]"#).unwrap();
        let p = layout.records[0].point;
        assert!((p.x - 1.0).abs() < f32::EPSILON);
        assert!((p.y - 2.0).abs() < f32::EPSILON);
        assert!((p.z - 3.0).abs() < f32::EPSILON);
        assert!((p.value - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn from_json_rejects_empty_and_non_array() {
        assert_eq!(
            SoulLayout::from_json("[]").unwrap_err(),
            SoulError::EmptyPrompt
        );
        assert_eq!(
            SoulLayout::from_json("{}").unwrap_err(),
            SoulError::InvalidFormat
        );
    }

    #[test]
    fn bootstrap_trains_a_three_output_network() {
        let layout = SoulLayout::from_json(
            r#"[
              {"label":"empathic teaching abbey"},
              {"label":"direct expert aviva"},
              {"label":"orchestration governance abi"}
            ]"#,
        )
        .unwrap();
        let mut net = PointNeuralNetwork::init(&[3, 8, 3], 0.05).unwrap();
        let (loss, report) = layout.bootstrap(&mut net).unwrap();
        assert!(loss.is_finite());
        assert!(report.spread.is_finite());
        let out = net
            .forward(&Point::from_text("empathic teaching").to_array())
            .unwrap();
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
