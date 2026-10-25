# Bound worker-journal lock acquisition

**Goal:** Permit a short transient lock handoff while retaining exclusive, fail-closed worker history ownership and distinguishing genuine I/O failure.
**Architecture:** One private acquisition helper used before create/open reads or writes; retry only lock contention, never effects, replay, approval or provider execution.
**Specification:** Approved modernization M2 baseline and M8 restart/uncertain-outcome recovery; ABI C03 directly, C06/C07 integration contribution. Existing source status is not changed by this packet.
**Tech stack:** Rust 1.99.1, standard File::try_lock, repository wrapper and existing controlled-I/O test conventions.

## Global Constraints

- Preserve worker history format, lifetime exclusivity, file permissions, budgets, corruption refusals, poison state and replay behavior.
- Never retry execution or effects to repair lock admission. Never admit a second owner after a timeout.
- Retry only `std::fs::TryLockError::WouldBlock`; genuine I/O errors fail immediately with a distinct bounded content-free error.
- No global time/syscall hooks, unsafe fork tests, arbitrary sleeps in tests, or serializing the existing crash test to conceal concurrency.
- Original worktrees and live stores are untouched; only this task's isolated ABI checkout is writable. WDBX and Abbey siblings are pinned consumer/producer verification inputs.

## Task 1: Retry transient lock contention before journal access

**Owned files:** Modify `crates/abi-worker/src/file_store.rs`, create `crates/abi-worker/src/file_store/lock.rs`, modify `crates/abi-worker/tests/durable_control.rs`. No transport, worker execution or Abbey implementation changes. Add no dependencies.

**Observed failure:** Preserved Abbey root gate failed in WDBX-feature `governed_relay::interruptions::kill_after_lifecycle_terminal`, `tests/governed_relay.rs:237`, opening ABI worker.log. ABI currently collapses every lock error into `history is busy`. The exact OS variant was discarded, so the observed cause is not proven. Existing WDBX `acquire_writer_lock` independently documents a fork/exec inherited-lock window and uses a 250 ms deadline with 1 ms waits. Match this bounded contention policy in ABI.

**Interface:** Private `acquire_history_lock(&File) -> Result<(), WorkerError>` called at both existing create/open acquisition sites, before header I/O. Success retains the actual locked file in `FileControlStore` until drop. Expiry keeps `ControlStore("history is busy")`; non-contention lock errors use `ControlStore("cannot lock history")`, without OS text or paths. Use a tiny instance-local internal test seam for acquisition, monotonic elapsed time and wait so the scheduling cases are deterministic.

- [ ] Read instructions and actual acquisition/callers; add failing deterministic tests first.
- [ ] Transient case holds a real file lock: first acquisition observes WouldBlock, injected wait releases owner and advances fake clock, subsequent real acquisition succeeds; another independent handle still cannot acquire while the acquired file remains alive; bytes unchanged.
- [ ] Persistent-owner case keeps actual lock through fake deadline: finite refusal, unchanged history bytes, successful acquisition after owner drop. No header/replay/effect work before lock acquisition.
- [ ] Inject non-WouldBlock I/O failure initially and after a transient contention; distinct error immediately, no extra retries or waits. Bound the deadline and wait interval exactly; use elapsed monotonic time, not wall clock.
- [ ] Strengthen existing public store exclusivity test to assert the busy outcome and untouched history while retaining create-new behavior, process-exit recovery and corrupt/partial-frame refusals.
- [ ] Implement helper, run focused library/file-store and durable-control tests, Rust formatting/Clippy for affected package, then the actual concurrent Abbey governed_relay test target through the exact sibling ABI producer. Use wrapper and stdin closed with isolated stores/build targets; set ABI_WDBX_PATH=:memory: and explicitly unset ABI_WDBX_PERSIST so persistence-specific tests can use their own scratch stores.
- [ ] Run full ABI gate on final source; preserve any independent failure and report. Do not repeat already-passing unrelated consumer suites without a reason.
- [ ] Self-review, commit owned files, record exact producer/consumer hashes and test outputs in task report. This does not establish power-loss or remote-platform qualification.

**Rollback:** Revert this local task commit. Existing journal bytes and protocol identities are unchanged; immediate-refusal behavior returns. No live data migration occurs.
