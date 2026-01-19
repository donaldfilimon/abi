# Eval Module Improvements Implementation Plan
> **Codebase Status:** Synced with repository as of 2026-01-18.

> **Status:** Completed ✅ (2026-01-18)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical bugs, eliminate code duplication, improve performance, and add missing functionality to the eval module.

**Architecture:** Extract shared tokenization to a common utility, fix the broken unique_words calculation, optimize n-gram computation with buffer reuse, add missing metrics (METEOR, CER, WER exports), and ensure proper API exposure.

**Tech Stack:** Zig 0.16, existing LLM tokenizer infrastructure, standard testing patterns

---

## Task 1: Extract Shared Tokenization Utility

**Files:**
- Create: `src/ai/implementation/eval/tokenizer.zig`
- Modify: `src/ai/implementation/eval/bleu.zig`
- Modify: `src/ai/implementation/eval/rouge.zig`
- Modify: `src/ai/implementation/eval/metrics.zig`
- Modify: `src/ai/implementation/eval/mod.zig`

**Step 1: Create the shared tokenizer module**

Create `src/ai/implementation/eval/tokenizer.zig`:

```zig
//! Shared tokenization utilities for evaluation metrics.
//!
//! Provides consistent text tokenization across BLEU, ROUGE, and other metrics.

const std = @import("std");

/// Tokenize text by whitespace, returning slices into the original text.
/// Caller owns the returned slice array (but not the token contents).
pub fn tokenize(allocator: std.mem.Allocator, text: []const u8) ![]const []const u8 {
    var tokens = std.ArrayListUnmanaged([]const u8){};
    errdefer tokens.deinit(allocator);

    var start: usize = 0;
    var i: usize = 0;

    while (i < text.len) : (i += 1) {
        if (std.ascii.isWhitespace(text[i])) {
            if (i > start) {
                try tokens.append(allocator, text[start..i]);
            }
            start = i + 1;
        }
    }

    // Last token
    if (start < text.len) {
        try tokens.append(allocator, text[start..]);
    }

    return tokens.toOwnedSlice(allocator);
}

/// Tokenize and lowercase text.
pub fn tokenizeLower(allocator: std.mem.Allocator, text: []const u8) !struct { tokens: []const []const u8, buffer: []u8 } {
    // First, create lowercased copy
    const lower = try allocator.alloc(u8, text.len);
    errdefer allocator.free(lower);

    for (text, 0..) |c, i| {
        lower[i] = std.ascii.toLower(c);
    }

    const tokens = try tokenize(allocator, lower);
    return .{ .tokens = tokens, .buffer = lower };
}

/// Count tokens without allocating the token array.
pub fn countTokens(text: []const u8) usize {
    var count: usize = 0;
    var in_word = false;

    for (text) |c| {
        if (std.ascii.isWhitespace(c)) {
            if (in_word) {
                count += 1;
                in_word = false;
            }
        } else {
            in_word = true;
        }
    }

    if (in_word) count += 1;
    return count;
}

test "tokenize basic" {
    const allocator = std.testing.allocator;
    const tokens = try tokenize(allocator, "the cat sat");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 3), tokens.len);
    try std.testing.expectEqualStrings("the", tokens[0]);
    try std.testing.expectEqualStrings("cat", tokens[1]);
    try std.testing.expectEqualStrings("sat", tokens[2]);
}

test "tokenize empty" {
    const allocator = std.testing.allocator;
    const tokens = try tokenize(allocator, "");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 0), tokens.len);
}

test "tokenize multiple spaces" {
    const allocator = std.testing.allocator;
    const tokens = try tokenize(allocator, "  hello   world  ");
    defer allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 2), tokens.len);
    try std.testing.expectEqualStrings("hello", tokens[0]);
    try std.testing.expectEqualStrings("world", tokens[1]);
}

test "count tokens" {
    try std.testing.expectEqual(@as(usize, 3), countTokens("the cat sat"));
    try std.testing.expectEqual(@as(usize, 0), countTokens(""));
    try std.testing.expectEqual(@as(usize, 2), countTokens("  hello   world  "));
}
```

**Step 2: Run test to verify tokenizer works**

Run: `zig test src/ai/implementation/eval/tokenizer.zig`
Expected: PASS (4 tests)

**Step 3: Update bleu.zig to use shared tokenizer**

In `src/ai/implementation/eval/bleu.zig`, add import at top and remove local tokenize:

```zig
// Add after other imports (around line 5):
const tokenizer = @import("tokenizer.zig");

// Replace all calls to local tokenize() with tokenizer.tokenize()
// Delete the local tokenize function (lines 197-220)
```

Changes:
- Line ~56: `const hyp_tokens = try tokenizer.tokenize(allocator, hypothesis);`
- Line ~59: `const ref_tokens = try tokenizer.tokenize(allocator, reference);`
- Line ~74: `const hyp_tokens = try tokenizer.tokenize(allocator, hypothesis);`
- Line ~82: `const tokens = try tokenizer.tokenize(allocator, ref);`
- Delete lines 197-220 (local tokenize function)

**Step 4: Update rouge.zig to use shared tokenizer**

In `src/ai/implementation/eval/rouge.zig`:

```zig
// Add after std import (line 6):
const tokenizer = @import("tokenizer.zig");

// Replace local tokenize calls with tokenizer.tokenize
// Delete local tokenize function (lines 188-211)
```

Changes:
- Line ~68: `const hyp_tokens = try tokenizer.tokenize(allocator, hypothesis);`
- Line ~71: `const ref_tokens = try tokenizer.tokenize(allocator, reference);`
- Line ~152: `const hyp_tokens = try tokenizer.tokenize(allocator, hypothesis);`
- Line ~155: `const ref_tokens = try tokenizer.tokenize(allocator, reference);`
- Delete lines 188-211 (local tokenize function)

**Step 5: Update metrics.zig to use shared tokenizer**

In `src/ai/implementation/eval/metrics.zig`:

```zig
// Add after std import (line 5):
const tokenizer = @import("tokenizer.zig");

// Replace local tokenize calls with tokenizer.tokenize
// Delete local tokenize function (lines 287-308)
```

Changes:
- Line ~47: `const hyp_tokens = try tokenizer.tokenize(allocator, hypothesis);`
- Line ~50: `const ref_tokens = try tokenizer.tokenize(allocator, reference);`
- Line ~264: `const hyp_tokens = try tokenizer.tokenize(allocator, hypothesis);`
- Line ~267: `const ref_tokens = try tokenizer.tokenize(allocator, reference);`
- Delete lines 287-308 (local tokenize function)

**Step 6: Export tokenizer from mod.zig**

In `src/ai/implementation/eval/mod.zig`, add:

```zig
// After other imports (around line 10):
pub const tokenizer = @import("tokenizer.zig");
pub const tokenize = tokenizer.tokenize;
```

**Step 7: Run all eval tests**

Run: `zig build test --summary all`
Expected: All 51+ tests pass

**Step 8: Commit**

```bash
git add src/ai/implementation/eval/tokenizer.zig src/ai/implementation/eval/bleu.zig src/ai/implementation/eval/rouge.zig src/ai/implementation/eval/metrics.zig src/ai/implementation/eval/mod.zig
git commit -m "refactor(eval): extract shared tokenization utility

- Create tokenizer.zig with shared tokenize() function
- Remove duplicated tokenize() from bleu.zig, rouge.zig, metrics.zig
- Add tokenize tests for edge cases (empty, multiple spaces)
- Export tokenizer from eval mod.zig"
```

---

## Task 2: Fix Broken unique_words Calculation

**Files:**
- Modify: `src/ai/implementation/eval/metrics.zig`

**Step 1: Write failing test for unique_words**

Add test at end of `src/ai/implementation/eval/metrics.zig`:

```zig
test "text statistics unique words" {
    const stats = computeTextStatistics("the cat sat on the mat");

    // "the" appears twice, so unique_words should be 5, not 6
    try std.testing.expectEqual(@as(usize, 6), stats.word_count);
    try std.testing.expectEqual(@as(usize, 5), stats.unique_words);

    // TTR = 5/6 ≈ 0.833
    try std.testing.expectApproxEqAbs(@as(f64, 0.8333), stats.type_token_ratio, 0.01);
}

test "text statistics all unique" {
    const stats = computeTextStatistics("one two three four");

    try std.testing.expectEqual(@as(usize, 4), stats.word_count);
    try std.testing.expectEqual(@as(usize, 4), stats.unique_words);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), stats.type_token_ratio, 0.0001);
}

test "text statistics all same" {
    const stats = computeTextStatistics("word word word word");

    try std.testing.expectEqual(@as(usize, 4), stats.word_count);
    try std.testing.expectEqual(@as(usize, 1), stats.unique_words);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), stats.type_token_ratio, 0.0001);
}
```

**Step 2: Run test to verify it fails**

Run: `zig test src/ai/implementation/eval/metrics.zig --test-filter "unique"`
Expected: FAIL - unique_words equals word_count (the bug)

**Step 3: Fix computeTextStatistics to track unique words**

Replace the `computeTextStatistics` function in `metrics.zig` (around lines 131-205):

```zig
/// Compute text statistics.
pub fn computeTextStatistics(text: []const u8) TextStatistics {
    if (text.len == 0) {
        return .{
            .char_count = 0,
            .word_count = 0,
            .sentence_count = 0,
            .avg_word_length = 0,
            .unique_words = 0,
            .type_token_ratio = 0,
        };
    }

    var word_count: usize = 0;
    var sentence_count: usize = 0;
    var total_word_length: usize = 0;
    var in_word = false;
    var word_start: usize = 0;

    // Use a simple hash set for unique words (bounded to avoid allocation)
    // Store hashes of words we've seen
    var word_hashes: [1024]u64 = undefined;
    var unique_count: usize = 0;

    for (text, 0..) |c, i| {
        if (std.ascii.isWhitespace(c)) {
            if (in_word) {
                const word = text[word_start..i];
                word_count += 1;
                total_word_length += word.len;

                // Check if word is unique using hash
                const hash = hashWord(word);
                if (!containsHash(&word_hashes, unique_count, hash)) {
                    if (unique_count < word_hashes.len) {
                        word_hashes[unique_count] = hash;
                        unique_count += 1;
                    }
                }

                in_word = false;
            }
        } else {
            if (!in_word) {
                word_start = i;
                in_word = true;
            }

            // Check for sentence terminators
            if (c == '.' or c == '!' or c == '?') {
                sentence_count += 1;
            }
        }
    }

    // Handle last word
    if (in_word) {
        const word = text[word_start..];
        word_count += 1;
        total_word_length += word.len;

        const hash = hashWord(word);
        if (!containsHash(&word_hashes, unique_count, hash)) {
            if (unique_count < word_hashes.len) {
                word_hashes[unique_count] = hash;
                unique_count += 1;
            }
        }
    }

    // Ensure at least one sentence if there's text
    if (sentence_count == 0 and word_count > 0) {
        sentence_count = 1;
    }

    const avg_word_length = if (word_count > 0)
        @as(f64, @floatFromInt(total_word_length)) / @as(f64, @floatFromInt(word_count))
    else
        0;

    const type_token_ratio = if (word_count > 0)
        @as(f64, @floatFromInt(unique_count)) / @as(f64, @floatFromInt(word_count))
    else
        0;

    return .{
        .char_count = text.len,
        .word_count = word_count,
        .sentence_count = sentence_count,
        .avg_word_length = avg_word_length,
        .unique_words = unique_count,
        .type_token_ratio = type_token_ratio,
    };
}

fn hashWord(word: []const u8) u64 {
    // Simple FNV-1a hash, case-insensitive
    var hash: u64 = 0xcbf29ce484222325;
    for (word) |c| {
        hash ^= @as(u64, std.ascii.toLower(c));
        hash *%= 0x100000001b3;
    }
    return hash;
}

fn containsHash(hashes: []const u64, count: usize, hash: u64) bool {
    for (hashes[0..count]) |h| {
        if (h == hash) return true;
    }
    return false;
}
```

**Step 4: Run test to verify it passes**

Run: `zig test src/ai/implementation/eval/metrics.zig --test-filter "unique"`
Expected: PASS (3 tests)

**Step 5: Run all tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/ai/implementation/eval/metrics.zig
git commit -m "fix(eval): compute actual unique_words in text statistics

- Track unique words using hash-based detection
- Fix type_token_ratio to return correct lexical diversity
- Add tests for unique word counting edge cases"
```

---

## Task 3: Add Missing Exports to Stub

**Files:**
- Modify: `src/ai/implementation/eval/stub.zig`

**Step 1: Add missing function stubs**

Add the following to `stub.zig` after the existing stub functions:

```zig
/// Stub CER computation.
pub fn computeCER(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
) !f64 {
    _ = allocator;
    _ = hypothesis;
    _ = reference;
    return error.EvalDisabled;
}

/// Stub WER computation.
pub fn computeWER(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
) !f64 {
    _ = allocator;
    _ = hypothesis;
    _ = reference;
    return error.EvalDisabled;
}

/// Stub normalized exact match computation.
pub fn computeNormalizedExactMatch(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
) !f64 {
    _ = allocator;
    _ = hypothesis;
    _ = reference;
    return error.EvalDisabled;
}

/// Stub Levenshtein distance computation.
pub fn levenshteinDistance(
    allocator: std.mem.Allocator,
    a: []const u8,
    b: []const u8,
) !usize {
    _ = allocator;
    _ = a;
    _ = b;
    return error.EvalDisabled;
}

/// Stub token metrics computation.
pub fn computeTokenMetrics(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
) !TokenMetrics {
    _ = allocator;
    _ = hypothesis;
    _ = reference;
    return error.EvalDisabled;
}

/// Stub text statistics computation.
pub fn computeTextStatistics(text: []const u8) TextStatistics {
    _ = text;
    return .{};
}

/// Stub windowed perplexity computation.
pub fn computeWindowedPerplexity(
    allocator: std.mem.Allocator,
    log_probs: []const f64,
    window_size: usize,
) ![]PerplexityResult {
    _ = allocator;
    _ = log_probs;
    _ = window_size;
    return error.EvalDisabled;
}

/// Stub perplexity from cross-entropy.
pub fn perplexityFromCrossEntropy(cross_entropy: f64) f64 {
    _ = cross_entropy;
    return 0;
}

/// Stub perplexity from BPC.
pub fn perplexityFromBpc(bpc: f64) f64 {
    _ = bpc;
    return 0;
}

/// Stub perplexity to BPC.
pub fn perplexityToBpc(perplexity_val: f64) f64 {
    _ = perplexity_val;
    return 0;
}

/// Stub aggregate perplexity.
pub fn aggregatePerplexity(results: []const PerplexityResult) PerplexityResult {
    _ = results;
    return .{};
}
```

**Step 2: Run build to verify stub compiles**

Run: `zig build -Denable-ai=false`
Expected: Build succeeds

**Step 3: Run full build with AI enabled**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/ai/implementation/eval/stub.zig
git commit -m "fix(eval): add missing function stubs for disabled AI

- Add computeCER, computeWER, computeNormalizedExactMatch stubs
- Add levenshteinDistance, computeTokenMetrics stubs
- Add perplexity utility stubs (aggregatePerplexity, etc.)
- Ensures API parity between enabled and disabled states"
```

---

## Task 4: Export Eval Module from AI Public API

**Files:**
- Modify: `src/ai/mod.zig`

**Step 1: Read current ai/mod.zig exports**

Check current exports to understand pattern.

**Step 2: Add eval export**

In `src/ai/mod.zig`, add after other implementation exports:

```zig
// Add with other pub const declarations:
pub const eval = implementation.eval;
```

**Step 3: Run tests to verify export works**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/ai/mod.zig
git commit -m "feat(ai): export eval module from public AI API

- Add eval to ai/mod.zig public exports
- Allows access via abi.ai.eval pattern
- Consistent with other AI module exports"
```

---

## Task 5: Fix Perplexity Hard-coded Limit

**Files:**
- Modify: `src/ai/implementation/eval/perplexity.zig`

**Step 1: Write failing test for long sequences**

Add test at end of `perplexity.zig`:

```zig
test "perplexity from probs long sequence" {
    // Create sequence longer than 1024
    var probs: [2000]f64 = undefined;
    for (&probs) |*p| {
        p.* = 0.1; // 10% probability each
    }

    const result = computePerplexityFromProbs(&probs);

    // Should process all 2000 tokens, not just 1024
    try std.testing.expectEqual(@as(usize, 2000), result.num_tokens);

    // Perplexity of uniform 0.1 = 1/0.1 = 10
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), result.perplexity, 0.01);
}
```

**Step 2: Run test to verify it fails**

Run: `zig test src/ai/implementation/eval/perplexity.zig --test-filter "long sequence"`
Expected: FAIL - num_tokens will be 1024, not 2000

**Step 3: Fix computePerplexityFromProbs to use dynamic allocation**

Replace the function in `perplexity.zig`:

```zig
/// Compute perplexity for a sequence with model probabilities.
/// Probabilities should be actual probabilities (0-1), not log probs.
pub fn computePerplexityFromProbs(probs: []const f64) PerplexityResult {
    if (probs.len == 0) {
        return .{
            .perplexity = std.math.inf(f64),
            .avg_log_prob = 0,
            .cross_entropy = std.math.inf(f64),
            .num_tokens = 0,
        };
    }

    // Compute directly without allocation - sum log probs inline
    var sum: f64 = 0;
    for (probs) |p| {
        // Clamp to avoid log(0)
        const clamped = @max(p, 1e-10);
        sum += @log(clamped);
    }

    const n = @as(f64, @floatFromInt(probs.len));
    const avg_log_prob = sum / n;
    const cross_entropy = -avg_log_prob;
    const perplexity = @exp(cross_entropy);

    return .{
        .perplexity = perplexity,
        .avg_log_prob = avg_log_prob,
        .cross_entropy = cross_entropy,
        .num_tokens = probs.len,
    };
}
```

**Step 4: Run test to verify it passes**

Run: `zig test src/ai/implementation/eval/perplexity.zig --test-filter "long sequence"`
Expected: PASS

**Step 5: Run all tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/ai/implementation/eval/perplexity.zig
git commit -m "fix(eval): remove 1024 token limit in perplexity from probs

- Compute log probs inline without intermediate array
- Handle sequences of any length correctly
- Add test for long sequences (2000 tokens)"
```

---

## Task 6: Add Batch Evaluation Tests

**Files:**
- Modify: `src/ai/implementation/eval/mod.zig`

**Step 1: Add batch evaluation tests**

Add tests at end of `mod.zig`:

```zig
test "batch evaluation" {
    const allocator = std.testing.allocator;
    var evaluator = Evaluator.init(allocator, .{});

    const hypotheses = [_][]const u8{
        "the cat sat on the mat",
        "hello world",
        "foo bar baz",
    };
    const references = [_][]const u8{
        "the cat sat on the mat",
        "hello there world",
        "completely different text",
    };

    const report = try evaluator.evaluateBatch(&hypotheses, &references);

    try std.testing.expectEqual(@as(usize, 3), report.num_samples);
    try std.testing.expect(report.avg_bleu > 0);
    try std.testing.expect(report.avg_f1 > 0);
    try std.testing.expect(report.exact_match_ratio > 0); // At least one exact match
}

test "batch evaluation length mismatch" {
    const allocator = std.testing.allocator;
    var evaluator = Evaluator.init(allocator, .{});

    const hypotheses = [_][]const u8{ "a", "b" };
    const references = [_][]const u8{"a"};

    const result = evaluator.evaluateBatch(&hypotheses, &references);
    try std.testing.expectError(error.LengthMismatch, result);
}

test "batch evaluation empty" {
    const allocator = std.testing.allocator;
    var evaluator = Evaluator.init(allocator, .{});

    const hypotheses = [_][]const u8{};
    const references = [_][]const u8{};

    const result = evaluator.evaluateBatch(&hypotheses, &references);
    try std.testing.expectError(error.EmptyInput, result);
}
```

**Step 2: Run tests**

Run: `zig test src/ai/implementation/eval/mod.zig --test-filter "batch"`
Expected: PASS (3 tests)

**Step 3: Run all tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/ai/implementation/eval/mod.zig
git commit -m "test(eval): add batch evaluation tests

- Test batch evaluation with multiple samples
- Test length mismatch error handling
- Test empty input error handling"
```

---

## Task 7: Add Additional Metrics Exports to mod.zig

**Files:**
- Modify: `src/ai/implementation/eval/mod.zig`

**Step 1: Add missing exports**

In `src/ai/implementation/eval/mod.zig`, add after existing exports:

```zig
// Add after existing metrics exports:
pub const computeTokenMetrics = metrics.computeTokenMetrics;
pub const computeTextStatistics = metrics.computeTextStatistics;
pub const computeNormalizedExactMatch = metrics.computeNormalizedExactMatch;
pub const computeCER = metrics.computeCER;
pub const computeWER = metrics.computeWER;
pub const levenshteinDistance = metrics.levenshteinDistance;

// Add perplexity utilities:
pub const perplexityFromCrossEntropy = perplexity.perplexityFromCrossEntropy;
pub const perplexityFromBpc = perplexity.perplexityFromBpc;
pub const perplexityToBpc = perplexity.perplexityToBpc;
pub const aggregatePerplexity = perplexity.aggregatePerplexity;
pub const computeWindowedPerplexity = perplexity.computeWindowedPerplexity;
pub const computePerplexityFromProbs = perplexity.computePerplexityFromProbs;

// Add BLEU smoothing method:
pub const SmoothingMethod = bleu.SmoothingMethod;
```

**Step 2: Run tests**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/ai/implementation/eval/mod.zig
git commit -m "feat(eval): export all metrics functions from mod.zig

- Export CER, WER, normalized exact match
- Export levenshtein distance
- Export perplexity utilities
- Export BLEU smoothing method enum"
```

---

## Task 8: Final Verification and Documentation

**Files:**
- None (verification only)

**Step 1: Run full test suite**

Run: `zig build test --summary all`
Expected: All tests pass

**Step 2: Run WASM build check**

Run: `zig build check-wasm`
Expected: Build succeeds

**Step 3: Run regular build**

Run: `zig build`
Expected: Build succeeds

**Step 4: Final commit (if any remaining changes)**

```bash
git status
# If clean, no commit needed
# If changes, commit with appropriate message
```

---

## Summary of Changes

| Task | What Changed | Impact |
|------|--------------|--------|
| 1 | Extract shared tokenizer | Eliminates 3x code duplication |
| 2 | Fix unique_words | Fixes broken type_token_ratio |
| 3 | Complete stub API | Prevents compilation surprises |
| 4 | Export eval from AI API | Consistent public API |
| 5 | Fix perplexity limit | Handles arbitrary length sequences |
| 6 | Add batch tests | Better test coverage |
| 7 | Export all functions | Complete public API |
| 8 | Verification | Ensures everything works |

**Total commits:** 7-8
**Estimated implementation time:** 45-60 minutes
