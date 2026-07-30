//! Adaptive EMA persona-router weights.
//!
//! Ported from `src/features/ai/router_modulator.zig`. Pure: serialize/deserialize
//! and EMA update have no I/O. Loading from and saving to a WDBX store is the
//! caller's job (the SEA learn loop), so this crate stays free of a store
//! dependency.

use crate::identity::{DEFAULT_ABBEY_WEIGHT, DEFAULT_ABI_WEIGHT, DEFAULT_AVIVA_WEIGHT};
use crate::router::ProfileWeights;

/// The WDBX key under which the modulator's state is stored.
pub const STORE_KEY: &str = "modulator:weights";

const DEFAULT_ALPHA: f32 = 0.3;

/// Adaptive EMA modulator that smooths routing weights over successive turns.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveModulator {
    w_ema: ProfileWeights,
    alpha: f32,
    update_count: u32,
}

impl AdaptiveModulator {
    /// Fresh modulator at the neutral prior with the default alpha.
    #[must_use]
    pub fn new() -> Self {
        Self::with_alpha(DEFAULT_ALPHA)
    }

    /// Fresh modulator at the neutral prior with a custom alpha.
    #[must_use]
    pub fn with_alpha(alpha: f32) -> Self {
        Self {
            w_ema: ProfileWeights {
                w_abbey: DEFAULT_ABBEY_WEIGHT,
                w_aviva: DEFAULT_AVIVA_WEIGHT,
                w_abi: DEFAULT_ABI_WEIGHT,
            },
            alpha,
            update_count: 0,
        }
    }

    /// Update the EMA: `new = alpha * observed + (1 - alpha) * old`, then
    /// renormalize. Saturates `update_count` at `u32::MAX`.
    pub fn update(&mut self, observed: ProfileWeights) {
        let a = self.alpha;
        let b = 1.0 - a;
        self.w_ema.w_abbey = a * observed.w_abbey + b * self.w_ema.w_abbey;
        self.w_ema.w_aviva = a * observed.w_aviva + b * self.w_ema.w_aviva;
        self.w_ema.w_abi = a * observed.w_abi + b * self.w_ema.w_abi;
        self.w_ema.normalize();
        self.update_count = self.update_count.saturating_add(1);
    }

    /// Current smoothed weights.
    #[must_use]
    pub fn weights(&self) -> ProfileWeights {
        self.w_ema
    }

    /// EMA smoothing factor.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Number of updates applied since init (saturating).
    #[must_use]
    pub fn update_count(&self) -> u32 {
        self.update_count
    }

    /// Serialize to the CSV form stored under [`STORE_KEY`].
    ///
    /// Format: `abbey,aviva,abi,update_count,alpha` with six-decimal floats.
    #[must_use]
    pub fn serialize(&self) -> String {
        format!(
            "{:.6},{:.6},{:.6},{},{:.6}",
            self.w_ema.w_abbey, self.w_ema.w_aviva, self.w_ema.w_abi, self.update_count, self.alpha
        )
    }

    /// Deserialize, falling back to [`Self::new`] on any validation failure.
    ///
    /// Rejects non-finite/negative weights, a non-positive total, alpha outside
    /// `[0,1]`, wrong field counts, and parse errors — matching Zig's
    /// `deserializeValidated` hardening so corrupted store values cannot poison
    /// routing.
    #[must_use]
    pub fn deserialize(data: &str) -> Self {
        Self::deserialize_validated(data).unwrap_or_default()
    }

    fn deserialize_validated(data: &str) -> Option<Self> {
        let mut fields = data.split(',');
        let abbey_text = fields.next()?;
        let aviva_text = fields.next()?;
        let abi_text = fields.next()?;
        let update_count_text = fields.next()?;
        let alpha_text = fields.next()?;
        if fields.next().is_some() {
            return None;
        }

        let abbey_weight: f32 = abbey_text.parse().ok()?;
        let aviva_weight: f32 = aviva_text.parse().ok()?;
        let abi_weight: f32 = abi_text.parse().ok()?;
        let update_count: u32 = update_count_text.parse().ok()?;
        let alpha: f32 = alpha_text.parse().ok()?;

        if !abbey_weight.is_finite()
            || !aviva_weight.is_finite()
            || !abi_weight.is_finite()
            || abbey_weight < 0.0
            || aviva_weight < 0.0
            || abi_weight < 0.0
        {
            return None;
        }

        let total = abbey_weight + aviva_weight + abi_weight;
        if !total.is_finite() || total <= 0.0 {
            return None;
        }
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return None;
        }

        let mut modulator = Self {
            w_ema: ProfileWeights {
                w_abbey: abbey_weight,
                w_aviva: aviva_weight,
                w_abi: abi_weight,
            },
            alpha,
            update_count,
        };
        modulator.w_ema.normalize();
        Some(modulator)
    }
}

impl Default for AdaptiveModulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::ProfileWeights;

    #[test]
    fn ema_smoothing_raises_the_observed_persona() {
        let mut modulator = AdaptiveModulator::with_alpha(0.5);
        modulator.update(ProfileWeights {
            w_abbey: 1.0,
            w_aviva: 0.0,
            w_abi: 0.0,
        });
        assert!(modulator.weights().w_abbey > 0.5);
        assert_eq!(modulator.update_count(), 1);
        let total =
            modulator.weights().w_abbey + modulator.weights().w_aviva + modulator.weights().w_abi;
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn serialize_deserialize_round_trips() {
        let mut modulator = AdaptiveModulator::with_alpha(0.25);
        modulator.update(ProfileWeights {
            w_abbey: 0.8,
            w_aviva: 0.1,
            w_abi: 0.1,
        });
        let restored = AdaptiveModulator::deserialize(&modulator.serialize());
        assert!((modulator.weights().w_abbey - restored.weights().w_abbey).abs() < 0.001);
        assert!((modulator.weights().w_aviva - restored.weights().w_aviva).abs() < 0.001);
        assert!((modulator.weights().w_abi - restored.weights().w_abi).abs() < 0.001);
        assert_eq!(modulator.update_count(), restored.update_count());
        assert!((modulator.alpha() - restored.alpha()).abs() < 0.001);
    }

    #[test]
    fn empty_string_falls_back_to_defaults() {
        let modulator = AdaptiveModulator::deserialize("");
        assert!((modulator.weights().w_abbey - DEFAULT_ABBEY_WEIGHT).abs() < 0.01);
        assert_eq!(modulator.update_count(), 0);
    }

    #[test]
    fn rejects_invalid_persisted_state() {
        let invalid = [
            "nan,0.3,0.4,1,0.3",
            "inf,0.3,0.4,1,0.3",
            "-inf,0.3,0.4,1,0.3",
            "-0.1,0.3,0.8,1,0.3",
            "0,0,0,1,0.3",
            "3.4028235e38,3.4028235e38,3.4028235e38,1,0.3",
            "0.3,0.3,0.4,1,nan",
            "0.3,0.3,0.4,1,inf",
            "0.3,0.3,0.4,1,-0.1",
            "0.3,0.3,0.4,1,1.1",
            "malformed,0.3,0.4,1,0.3",
            "0.3,,0.4,1,0.3",
            "0.3,0.3,0.4,not-a-count,0.3",
            "0.3,0.3,0.4,4294967296,0.3",
            "0.3,0.3,0.4,1",
            "0.3,0.3,0.4,1,0.3,",
            "0.3,0.3,0.4,1,0.3,extra",
        ];
        for state in invalid {
            let restored = AdaptiveModulator::deserialize(state);
            assert!(
                (restored.weights().w_abbey - DEFAULT_ABBEY_WEIGHT).abs() < 0.0001,
                "{state}"
            );
            assert_eq!(restored.update_count(), 0, "{state}");
            assert!((restored.alpha() - 0.3).abs() < 0.0001, "{state}");
        }
    }

    #[test]
    fn normalizes_valid_persisted_weights() {
        let restored = AdaptiveModulator::deserialize("2,3,5,42,0.75");
        assert!((restored.weights().w_abbey - 0.2).abs() < 0.0001);
        assert!((restored.weights().w_aviva - 0.3).abs() < 0.0001);
        assert!((restored.weights().w_abi - 0.5).abs() < 0.0001);
        assert_eq!(restored.update_count(), 42);
        assert!((restored.alpha() - 0.75).abs() < 0.0001);
    }

    #[test]
    fn accepts_alpha_boundaries() {
        let zero = AdaptiveModulator::deserialize("2,3,5,7,0");
        assert_eq!(zero.update_count(), 7);
        assert_eq!(zero.alpha().to_bits(), 0.0_f32.to_bits());
        let one = AdaptiveModulator::deserialize("2,3,5,9,1");
        assert_eq!(one.update_count(), 9);
        assert_eq!(one.alpha().to_bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn saturates_a_maximum_update_count() {
        let mut restored = AdaptiveModulator::deserialize("0.2,0.3,0.5,4294967295,0.5");
        restored.update(ProfileWeights {
            w_abbey: 1.0,
            w_aviva: 0.0,
            w_abi: 0.0,
        });
        assert_eq!(restored.update_count(), u32::MAX);
    }
}
