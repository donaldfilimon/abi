//! Provider-neutral usage accounting and run budgets.
//!
//! # No pricing, ever
//!
//! [`Usage`] carries counts and elapsed wall time and nothing else. This crate
//! never converts usage into money: it has no price table, no currency, and no
//! `estimated_cost` field. Token counts are *reported by the provider adapter*
//! — this crate does not tokenize text and cannot verify them.

use std::time::Duration;

/// Metering for one run: counts reported by the provider plus measured time.
///
/// Token fields are provider-reported. A deterministic test provider documents
/// its own counting rule (see [`EchoProvider`], which counts whitespace-
/// separated words and is explicitly not a tokenizer).
///
/// [`EchoProvider`]: crate::EchoProvider
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    /// Prompt tokens reported by the provider.
    pub input_tokens: u64,
    /// Completion tokens reported by the provider.
    pub output_tokens: u64,
    /// Wall-clock duration measured by the runtime, not reported by the provider.
    pub duration: Duration,
}

impl Usage {
    /// Usage with both counts zero and no elapsed time.
    #[must_use]
    pub fn zero() -> Self {
        Self::default()
    }

    /// Total reported tokens, saturating on overflow.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }

    /// Combine two runs' usage, saturating every field.
    #[must_use]
    pub fn merged(self, other: Self) -> Self {
        Self {
            input_tokens: self.input_tokens.saturating_add(other.input_tokens),
            output_tokens: self.output_tokens.saturating_add(other.output_tokens),
            duration: self.duration.saturating_add(other.duration),
        }
    }
}

/// The budget dimension that stopped a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetLimit {
    /// Too many events were about to be delivered.
    Events,
    /// Too many output tokens had been reported.
    OutputTokens,
    /// The run had been going for too long.
    Duration,
}

impl BudgetLimit {
    /// Stable lowercase label, suitable for logs and audit records.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Events => "events",
            Self::OutputTokens => "output_tokens",
            Self::Duration => "duration",
        }
    }
}

/// Cooperative ceilings applied to one run.
///
/// Like cancellation, a budget is enforced at event boundaries only: it stops
/// the next event from being delivered, it does not interrupt work already in
/// progress inside a provider. Every dimension is optional and unset by
/// default, so [`RunBudget::unlimited`] runs are fully deterministic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunBudget {
    /// Maximum events delivered to the sink, excluding the terminal event.
    ///
    /// Every delivered event counts, including a provider's leading
    /// [`ModelEvent::Started`]: a provider that announces itself spends one
    /// unit of this ceiling before it streams any content.
    ///
    /// [`ModelEvent::Started`]: crate::ModelEvent::Started
    pub max_events: Option<u64>,
    /// Maximum provider-reported output tokens.
    pub max_output_tokens: Option<u64>,
    /// Maximum elapsed wall-clock time.
    pub max_duration: Option<Duration>,
}

impl RunBudget {
    /// A budget with no ceiling on any dimension.
    #[must_use]
    pub fn unlimited() -> Self {
        Self::default()
    }

    /// Set the delivered-event ceiling.
    #[must_use]
    pub fn with_max_events(mut self, events: u64) -> Self {
        self.max_events = Some(events);
        self
    }

    /// Set the output-token ceiling.
    #[must_use]
    pub fn with_max_output_tokens(mut self, tokens: u64) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// Set the elapsed-time ceiling.
    #[must_use]
    pub fn with_max_duration(mut self, duration: Duration) -> Self {
        self.max_duration = Some(duration);
        self
    }

    /// Which ceiling, if any, is already reached for the supplied progress.
    ///
    /// Dimensions are checked in a fixed order — events, then output tokens,
    /// then duration — so a run that breaches two at once reports the same
    /// limit every time.
    #[must_use]
    pub fn exceeded(
        &self,
        events: u64,
        output_tokens: u64,
        elapsed: Duration,
    ) -> Option<BudgetLimit> {
        if self.max_events.is_some_and(|max| events >= max) {
            return Some(BudgetLimit::Events);
        }
        if self
            .max_output_tokens
            .is_some_and(|max| output_tokens >= max)
        {
            return Some(BudgetLimit::OutputTokens);
        }
        if self.max_duration.is_some_and(|max| elapsed >= max) {
            return Some(BudgetLimit::Duration);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{BudgetLimit, RunBudget, Usage};
    use std::time::Duration;

    #[test]
    fn usage_totals_and_merges_saturate() {
        let usage = Usage {
            input_tokens: 3,
            output_tokens: 4,
            duration: Duration::from_millis(5),
        };
        assert_eq!(usage.total_tokens(), 7);

        let merged = usage.merged(usage);
        assert_eq!(merged.input_tokens, 6);
        assert_eq!(merged.output_tokens, 8);
        assert_eq!(merged.duration, Duration::from_millis(10));

        let huge = Usage {
            input_tokens: u64::MAX,
            output_tokens: u64::MAX,
            duration: Duration::ZERO,
        };
        assert_eq!(huge.total_tokens(), u64::MAX);
        assert_eq!(huge.merged(usage).input_tokens, u64::MAX);
    }

    #[test]
    fn zero_usage_is_empty() {
        let usage = Usage::zero();
        assert_eq!(usage.total_tokens(), 0);
        assert_eq!(usage.duration, Duration::ZERO);
    }

    #[test]
    fn unlimited_budget_never_trips() {
        let budget = RunBudget::unlimited();
        assert_eq!(
            budget.exceeded(u64::MAX, u64::MAX, Duration::from_secs(600)),
            None
        );
    }

    #[test]
    fn each_dimension_trips_in_a_fixed_order() {
        let budget = RunBudget::unlimited()
            .with_max_events(2)
            .with_max_output_tokens(10)
            .with_max_duration(Duration::from_secs(1));

        assert_eq!(budget.exceeded(0, 0, Duration::ZERO), None);
        assert_eq!(
            budget.exceeded(2, 0, Duration::ZERO),
            Some(BudgetLimit::Events)
        );
        assert_eq!(
            budget.exceeded(0, 10, Duration::ZERO),
            Some(BudgetLimit::OutputTokens)
        );
        assert_eq!(
            budget.exceeded(0, 0, Duration::from_secs(2)),
            Some(BudgetLimit::Duration)
        );
        // Two breaches at once still report the first dimension.
        assert_eq!(
            budget.exceeded(2, 99, Duration::ZERO),
            Some(BudgetLimit::Events)
        );
    }

    #[test]
    fn limit_labels_are_stable() {
        assert_eq!(BudgetLimit::Events.as_str(), "events");
        assert_eq!(BudgetLimit::OutputTokens.as_str(), "output_tokens");
        assert_eq!(BudgetLimit::Duration.as_str(), "duration");
    }
}
