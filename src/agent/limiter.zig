pub const TokenBucket = struct {
    capacity: u32,
    refill_ms: u32,
    tokens: u32,
    last_refill_ms: u64,

    pub fn init(capacity: u32, refill_ms: u32, now_ms: u64) TokenBucket {
        return .{ .capacity = capacity, .refill_ms = refill_ms, .tokens = capacity, .last_refill_ms = now_ms };
    }

    pub fn acquire(self: *TokenBucket, now_ms: u64, amount: u32) bool {
        self.refill(now_ms);
        if (amount > self.tokens) return false;
        self.tokens -= amount;
        return true;
    }

    fn refill(self: *TokenBucket, now_ms: u64) void {
        if (now_ms <= self.last_refill_ms) return;
        const elapsed = now_ms - self.last_refill_ms;
        if (elapsed < self.refill_ms) return;
        self.tokens = self.capacity;
        self.last_refill_ms = now_ms;
    }
};
