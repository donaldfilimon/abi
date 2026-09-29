# Provable Member Erasure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement provable member erasure so gated member fact text lives in WDBX as per-member AES-256-GCM ciphertext, a host operator can erase any member with a signed receipt, the Keychain master rotates at most hourly and is this-device-only, and erasure clears every channel summary in the scope.

**Architecture:** WDBX gains episode variants only. The abi gateway owns member keys and record keys, wrapped by a Keychain master the bot never sees. abbey-bot keeps a RAM cache and deletes local copies only after the ledger tombstone. Keyed commitments are not part of the design. This plan is future work. None of it is implemented. Do not start until `docs/superpowers/specs/2026-09-29-spec-member-erasure.md` §12 records Donald's approval.

**Tech Stack:** Rust edition 2024, wdbx nightly pin, abbey-bot stable 1.98.0, AES-256-GCM, macOS Keychain via `security_framework`, existing v3 canonical CBOR and detached Ed25519 signatures.

**Spec:** `docs/superpowers/specs/2026-09-29-spec-member-erasure.md`

**Not in this plan:** COSE, evidence-weighted retrieval, retrieval-time staleness, `task_regime`, `regime_posterior`, C3–C7 promotion, hosted federation, redacted derivative blocks, high-rate garbage collection, a future-learning opt-out, a DM `guild_ref` scheme beyond one key per `(guild_ref, member)`, and a resume flag after operator rollback. abi, abbey, and wdbx stay sibling path dependencies. abbey-bot does not gain an abi or wdbx crate dependency.

---

### Task 1: Reject a keyed commitment on the new record

**Files:**
- Create: `../wdbx/crates/abi-wdbx/tests/v3_member_erasure.rs`
- Modify: `../wdbx/crates/abi-wdbx/src/v3/episode/types.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn encrypted_memory_record_has_no_keyed_commitment_field() {
    let src = include_str!("../src/v3/episode/types.rs");
    let start = src.find("struct EncryptedMemoryRecord").expect("type");
    let end = src[start..].find("\n}\n").expect("end");
    let body = &src[start..start + end];
    assert!(body.contains("ciphertext"));
    assert!(!body.contains("keyed_commitment"));
    assert!(!body.contains("payload_commitment"));
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `cargo test --manifest-path ../wdbx/Cargo.toml --test v3_member_erasure encrypted_memory_record_has_no_keyed_commitment_field`

Expected: FAIL because `EncryptedMemoryRecord` does not exist yet.

- [ ] **Step 3: Add the struct from the spec**

```rust
pub struct EncryptedMemoryRecord {
    pub class: MemoryClass,
    pub retention: RetentionClass,
    pub member_key: [u8; 16],
    pub record_key: [u8; 16],
    pub nonce: [u8; 12],
    pub ciphertext: Vec<u8>,
    pub supersedes: Option<[u8; 32]>,
    pub migrates: Option<[u8; 32]>,
}
```

Do not add a commitment field. Plaintext is not a field.

- [ ] **Step 4: Re-run the test**

Expected: PASS.

- [ ] **Step 5: Commit** in wdbx only, after that repo's fmt and the focused test pass. Do not commit unrelated dirty files.

### Task 2: Shape rule for nonzero record keys and 64-byte buckets

**Files:**
- Modify: `../wdbx/crates/abi-wdbx/src/v3/episode/validate.rs`
- Modify: `../wdbx/crates/abi-wdbx/tests/v3_member_erasure.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn zero_record_key_and_short_ciphertext_are_invalid_input() {
    let bad_key = sample_record([0u8; 16], vec![0u8; 80]);
    let bad_len = sample_record([1u8; 16], vec![0u8; 16 + 63]);
    assert_eq!(validate_encrypted(&bad_key).unwrap_err(), ShapeError::InvalidInput);
    assert_eq!(validate_encrypted(&bad_len).unwrap_err(), ShapeError::InvalidInput);
    let ok = sample_record([2u8; 16], vec![0u8; 16 + 64]);
    assert!(validate_encrypted(&ok).is_ok());
}
```

- [ ] **Step 2: Run it**

Expected: FAIL to compile, `validate_encrypted` is missing.

- [ ] **Step 3: Implement the rule**

```rust
pub fn validate_encrypted(record: &EncryptedMemoryRecord) -> Result<(), ShapeError> {
    if record.member_key == [0u8; 16] || record.record_key == [0u8; 16] {
        return Err(ShapeError::InvalidInput);
    }
    let ct = record.ciphertext.len();
    if ct < 16 + 64 || (ct - 16) % 64 != 0 || ct > 4096 {
        return Err(ShapeError::InvalidInput);
    }
    Ok(())
}
```

- [ ] **Step 4: Re-run the test**

Expected: PASS.

- [ ] **Step 5: Commit** in wdbx.

### Task 3: Tombstone append leaves ciphertext bytes unchanged

**Files:**
- Modify: `../wdbx/crates/abi-wdbx/src/v3/episode/types.rs`
- Modify: `../wdbx/crates/abi-wdbx/tests/v3_member_erasure.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn erasure_does_not_rewrite_ciphertext() {
    let mut store = open_empty();
    let digest = store.append_encrypted(sample_record([9u8; 16], vec![7u8; 80])).unwrap();
    let before = store.ciphertext_of(digest);
    store.append_member_erasure(MemberErasure {
        member_key: [9u8; 16],
        erased_records: vec![digest],
    }).unwrap();
    assert_eq!(store.ciphertext_of(digest), before);
    assert!(store.replay_verify().is_ok());
}
```

- [ ] **Step 2: Run it**

Expected: FAIL to compile.

- [ ] **Step 3: Append `MemberErasure` as a new event. Do not mutate the earlier line. Destroying keys is the gateway's job in Task 5, not a ledger rewrite.**

- [ ] **Step 4: Re-run**

Expected: PASS, and the existing memory-candidate golden `3c19a479…` and memory-edge golden `dfc839a6…` still match.

- [ ] **Step 5: Commit** in wdbx. Run `cargo fmt --all --check`, `cargo clippy --workspace --all-targets`, and `cargo test --workspace` from `../wdbx`.

### Task 4: Hourly rotation decision

**Files:**
- Create: `crates/abi-wdbx-gateway/src/member_keys.rs`
- Create: `crates/abi-wdbx-gateway/src/member_keys/tests.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn rotation_is_due_after_erasure_and_at_most_hourly() {
    let hour = 3_600_000;
    assert!(rotation_due(0, None, true));
    assert!(!rotation_due(hour, Some(1), true));
    assert!(rotation_due(hour, Some(0), true));
    assert!(!rotation_due(hour + 1, Some(0), false));
}
```

- [ ] **Step 2: Run** `./tools/cargo.sh test -p abi-wdbx-gateway rotation_is_due -- --exact < /dev/null`

Expected: FAIL to compile.

- [ ] **Step 3: Implement**

```rust
pub fn rotation_due(now_ms: u64, last_rotation_ms: Option<u64>, erasure_since_rotation: bool) -> bool {
    if !erasure_since_rotation {
        return false;
    }
    match last_rotation_ms {
        None => true,
        Some(last) => now_ms.saturating_sub(last) >= 3_600_000,
    }
}
```

The Keychain write itself uses after-first-unlock and this-device-only. Measure the macOS 27.2 attribute in this task and record it in the spec before any receipt text mentions backups. Deleting wrapped keys for an erasure does not require the master key to be readable.

- [ ] **Step 4: Re-run the test**

Expected: PASS.

- [ ] **Step 5: Commit** in abi only if `git status` shows this task's files and nothing else.

### Task 5: Host-operator erasure receipt

**Files:**
- Modify: `crates/abi-wdbx-gateway/src/member_keys.rs`
- Modify: `crates/abi-cli/src/wdbx/episode.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn operator_erasure_requires_the_operator_token_and_returns_a_receipt() {
    let gateway = TestGateway::new();
    let denied = gateway.erase(EraseRequest {
        token: Token::Service,
        scope: "discord:1",
        subject: "user:9",
    });
    assert_eq!(denied.unwrap_err(), EraseError::Unauthorized);
    let ok = gateway.erase(EraseRequest {
        token: Token::Operator,
        scope: "discord:1",
        subject: "user:9",
    }).unwrap();
    assert_ne!(ok.tombstone_digest, [0u8; 32]);
    assert!(ok.signed);
    assert!(ok.kek_rotation_pending);
}
```

- [ ] **Step 2: Run the test**

Expected: FAIL to compile.

- [ ] **Step 3: Implement `EraseMember` for `ActorKind::HostOperator` only when the operator token matches. Append the tombstone, delete that member's wrapped key and every record key under it, and set `kek_rotation_pending` from `rotation_due`. Service-token callers cannot use the operator path.**

- [ ] **Step 4: Re-run**

Expected: PASS.

- [ ] **Step 5: Commit** in abi under the same clean-status rule. Gate with `./tools/check.sh < /dev/null` when this phase's abi edits are the only dirty files.

### Task 6: Clear every summary in the scope

**Files:**
- Create: `../abbey-bot/src/member_erasure.rs`
- Modify: `../abbey-bot/src/member_erasure.rs` (test module in the same file)

abbey-bot stays a binary crate with no abi or wdbx dependency. Transcribe the receipt fields; do not import gateway crates.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn erasure_clears_all_summaries_in_the_scope() {
    let mut scope = ScopeMemory {
        summaries: vec!["kept".into(), "also".into()],
        facts: vec![Fact { member: "user:9", text: "secret" }],
    };
    let receipt = plan_erasure(&mut scope, "user:9");
    assert!(scope.summaries.is_empty());
    assert!(scope.facts.is_empty());
    assert!(receipt.removed.iter().any(|row| row == "summaries:2"));
}
```

- [ ] **Step 2: Run** `cargo test --locked --manifest-path ../abbey-bot/Cargo.toml erasure_clears_all_summaries -- --exact`

Expected: FAIL to compile. A filter that matches nothing can still exit 0 in this crate, so confirm the test name appears in the output.

- [ ] **Step 3: Implement `plan_erasure` so one member erasure in a scope drops every channel summary in that scope, not only summaries that mention the member, and lists the count on the receipt.**

```rust
pub fn plan_erasure(scope: &mut ScopeMemory, member: &str) -> Receipt {
    let summary_count = scope.summaries.len();
    scope.summaries.clear();
    scope.facts.retain(|fact| fact.member != member);
    Receipt { removed: vec![format!("summaries:{summary_count}")] }
}
```

- [ ] **Step 4: Re-run the focused test, then `./check.sh` with no pipe** once this is the only abbey-bot change.

Expected: the focused test passes and `check.sh` prints its own success line with exit 0.

- [ ] **Step 5: Commit** in abbey-bot only when status is limited to this task.

### Task 7: Seven clean shadow days, then stop after rollback

**Files:**
- Modify: `../abbey-bot/src/member_erasure.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn switchover_waits_for_seven_clean_days_and_stops_after_rollback() {
    let mut clock = ShadowClock::default();
    assert!(!clock.ready(days(6), 7));
    assert!(clock.ready(days(7), 7));
    assert!(!clock.ready(days(7), 6));
    clock.rollback();
    assert!(!clock.ready(days(30), 30));
}
```

- [ ] **Step 2: Run the focused test**

Expected: FAIL to compile.

- [ ] **Step 3: Implement**

```rust
pub struct ShadowClock {
    rolled_back: bool,
}

impl ShadowClock {
    pub fn ready(&self, elapsed_ms: u64, clean_compares: u32) -> bool {
        !self.rolled_back && elapsed_ms >= 7 * 24 * 3_600_000 && clean_compares >= 7
    }

    pub fn rollback(&mut self) {
        self.rolled_back = true;
    }
}
```

A mismatch resets the clean window. `ABBEY_MEMORY_AUTO_SWITCHOVER=off` forces `ready` false without resetting the window. Do not add `--resume-auto`.

- [ ] **Step 4: Re-run the test**

Expected: PASS.

- [ ] **Step 5: Commit** in abbey-bot under the clean-status rule.

### Task 8: Honest boundary after the code lands

**Files:**
- Modify: `docs/spec/wdbx-north-star.mdx`
- Modify: `docs/superpowers/specs/2026-09-29-wdbx-completion-design.md`

- [ ] **Step 1: After Tasks 1–7 have passing tests, change the north-star erasure row from Proposed to the narrow Current statement the tests prove. Leave C3–C7, COSE, and evidence-weighted retrieval unpromoted.**

- [ ] **Step 2: Re-read both docs and delete any sentence that still says the whole completion design is unimplemented if only sub-project 1 landed. Items 2–12 of the completion design stay deferred.**

- [ ] **Step 3: Run `./tools/check.sh < /dev/null` in abi and `./check.sh` with no pipe in abbey-bot. Read each verdict line and exit code.**

- [ ] **Step 4: Commit** each repo only when its status is limited to that repo's task files. Do not push unless Donald asks.

## Self-review

Spec coverage: encrypted payloads and the rejected keyed commitment are Tasks 1–2; tombstone stability is Task 3; hourly this-device-only rotation is Task 4; direct operator erasure is Task 5; summaries cleared is Task 6; seven-day switch and no resume flag are Task 7. Deferred Program 4 work has no task. The plan does not say any of this code already exists.
