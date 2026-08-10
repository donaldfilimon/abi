//! Cooperative cancellation.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// Cooperative cancellation flag shared between a caller and a running provider.
///
/// Cancellation is *cooperative*: setting the flag never interrupts a provider
/// mid-event. The token is consulted by [`RunContext::checkpoint`], including
/// before each event is delivered, so a cancelled run stops at the next
/// provider checkpoint and the event that would have crossed it is never
/// delivered to the sink.
///
/// [`RunContext::checkpoint`]: crate::RunContext::checkpoint
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a token that has not been cancelled.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation. Idempotent.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::CancellationToken;

    #[test]
    fn starts_uncancelled_and_latches() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn clones_share_one_flag() {
        let token = CancellationToken::new();
        let clone = token.clone();
        clone.cancel();
        assert!(token.is_cancelled());
    }
}
