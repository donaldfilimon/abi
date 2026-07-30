//! In-process metrics registry (counters + gauges).
//!
//! Ported from `src/features/metrics/mod.zig`. Complements the process-wide
//! Prometheus counter table in the parent module with an owned registry for
//! local instrumentation.

use std::collections::BTreeMap;

/// Snapshot of one counter in an owned registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryCounter {
    /// Counter name.
    pub name: String,
    /// Counter value.
    pub value: u64,
}

/// Snapshot of one gauge.
#[derive(Debug, Clone, PartialEq)]
pub struct GaugeSnapshot {
    /// Gauge name.
    pub name: String,
    /// Gauge value.
    pub value: f64,
}

/// Owned metrics registry.
#[derive(Debug, Default, Clone)]
pub struct Metrics {
    counters: BTreeMap<String, u64>,
    gauges: BTreeMap<String, f64>,
    enabled: bool,
}

impl Metrics {
    /// Create an enabled registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            counters: BTreeMap::new(),
            gauges: BTreeMap::new(),
            enabled: true,
        }
    }

    /// Feature always enabled in the Rust workspace.
    #[must_use]
    pub const fn is_enabled() -> bool {
        true
    }

    /// Whether this instance accepts updates.
    #[must_use]
    pub const fn enabled(&self) -> bool {
        self.enabled
    }

    /// Increment a named counter.
    pub fn increment(&mut self, name: &str, delta: u64) {
        if !self.enabled || name.is_empty() || delta == 0 {
            return;
        }
        let entry = self.counters.entry(name.to_owned()).or_insert(0);
        *entry = entry.saturating_add(delta);
    }

    /// Set a named gauge.
    pub fn set_gauge(&mut self, name: &str, value: f64) {
        if !self.enabled || name.is_empty() {
            return;
        }
        self.gauges.insert(name.to_owned(), value);
    }

    /// Read a counter if present.
    #[must_use]
    pub fn get_counter(&self, name: &str) -> Option<u64> {
        self.counters.get(name).copied()
    }

    /// Read a gauge if present.
    #[must_use]
    pub fn get_gauge(&self, name: &str) -> Option<f64> {
        self.gauges.get(name).copied()
    }

    /// Snapshot all counters (name order).
    #[must_use]
    pub fn snapshot_counters(&self) -> Vec<RegistryCounter> {
        self.counters
            .iter()
            .map(|(name, value)| RegistryCounter {
                name: name.clone(),
                value: *value,
            })
            .collect()
    }

    /// Snapshot all gauges (name order).
    #[must_use]
    pub fn snapshot_gauges(&self) -> Vec<GaugeSnapshot> {
        self.gauges
            .iter()
            .map(|(name, value)| GaugeSnapshot {
                name: name.clone(),
                value: *value,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_and_gauges_round_trip() {
        let mut m = Metrics::new();
        m.increment("events", 2);
        m.increment("events", 3);
        m.set_gauge("load", 0.5);
        assert_eq!(m.get_counter("events"), Some(5));
        assert_eq!(m.get_gauge("load"), Some(0.5));
        assert_eq!(m.snapshot_counters().len(), 1);
        assert_eq!(m.snapshot_gauges()[0].name, "load");
        assert!(Metrics::is_enabled());
    }
}
