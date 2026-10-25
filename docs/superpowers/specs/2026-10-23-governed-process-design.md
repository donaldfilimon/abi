# Governed local process adapter design

Status: proposed implementation for Task 7 of the approved production ecosystem.
This is one process-adapter slice; it does not complete platform, messaging,
separate-host, public service or training acceptance.

## Intent and choices

Connect a real bounded Unix process to the existing governed host, exact local
approval, current authority and postcondition contracts. Trusted startup alone
chooses the resource and operation. The initial closed operation reports the
canonical working directory using the reviewed system `/bin/pwd` command and
empty JSON-object parameters. No shell, PATH resolution, inherited environment,
model-selected executable, argv or working directory is exposed.

Approaches considered: extracting the CLI runner would retain unbounded output
and thread joins; exposing generic commands would expand authorization and OS
containment requirements. Use a closed public operation over a private reusable
supervisor. This is the coordinator's implementation choice under the user's
instruction to continue all authorized work, not a separately selected user API.

## Contract

- Keep canonical main and all unrelated dirty work. No commits, CI, publication,
  live accounts or user stores. Rust 1.99.1; wrapper, locked commands, stdin EOF.
- Public types live in `abi_agent_host::platform_process`.
- `ProcessLimits {timeout_ms: u64, max_output_bytes: usize, cleanup_grace_ms: u64}`
  implements Default: timeout 1–30,000 milliseconds, aggregate raw output 1–65,536
  bytes, cleanup grace 1–1,000 milliseconds. Defaults 1,000/4,096/250.
- `ProcessPlan::working_directory(&Path, ProcessLimits)` captures canonical cwd,
  retained directory FD and identity, root-owned regular non-group/world-writable
  canonical system executable identity/content, empty env and limits. Reject
  unsupported platforms, symlink input cwd, mutable/unsupported executable, invalid
  bounds and non-UTF8 cwd or canonical cwd over 4,096 UTF8 bytes. Keep public executable/argv selection closed.
- `ProcessPlan::resource_digest()` binds a domain-separated versioned canonical
  plan including cwd/executable identity, operation, limits, env and predicates.
- `ProcessPlan::bind(self, &CompiledSource, Arc<dyn Clock>)` seals and returns
  `Result<(GovernedProcessExecutor, ProcessObservationHandle), ToolExecutionError>`. Validate exact resource
  digest, capability `local.process.cwd` version `1.0.0`, adapter
  `abi.pinned-process` API `1`, empty-object input, output contract, A1Actor
  confirmation/approval, required owned predicates/permissions and byte budgets.
  Capture entire compiled source/package/scope; no later rebind. Exact canonical
  schema equality is required, not permissive validation of one sample. Input
  schema is `{"type":"object","additionalProperties":false}`. Output schema is
  `{"type":"object","properties":{"cwd":{"type":"string"},"exit_code":{"const":0}},"required":["cwd","exit_code"],"additionalProperties":false}`.
  Require owned precondition `process.ready`, invariant `process.scoped`, both
  owned permission names and exactly both owned postconditions. Preserve other
  upstream prerequisites; refuse missing owned prerequisites or mismatched schema.
  Require PlatformRead side effect, Cooperative cancellation and LocalOnly residency.
- Tool name is `process_cwd`. Reject changed call/input, invalid request digest,
  mismatched request/source/scope/package/parameter/prepared binding and stopped
  context before spawn. Existing host remains authority for full grants and
  durable prepare; adapter does not claim `PreparedExecution` proves persistence.
- `ProcessObservationHandle::decorate(self, F)` consumes the opaque handle and
  returns `GovernedProcessFacts<F>`. Plan construction also returns a Result with
  `ToolExecutionError`; diagnostics are fixed bounded codes.
  This decorator retains the upstream authority/grants and independently removes
  then recomputes owned permissions `process.pinned_exec_access` and
  `process.pinned_cwd_search`, preconditions/invariants and captured pre-state.
  Preserve upstream observation timestamps/expiry and refuse stale/future/empty
  intervals; process observations may narrow but never extend authority freshness.
  Observe actual effective-credential OS access and metadata without a probe
  process via `faccessat(..., AT_EACCESS)`; unsupported access flags or mismatched
  requests refuse (never replace ACL-aware checks with mode-bit inference). Check path still maps to
  pinned directory; retain FD for execution. Do not manufacture upstream grants.
- Use private per-instance single-flight receipt state, no global/env test hooks.
  Clear receipt before each attempt including refusals. Concurrent/reentrant
  work refuses. Successful completion creates an opaque receipt with request and
  prepared identity, plan, actual OS status, output digest, EOF and cleanup facts,
  and trusted completion timestamp. No public successful-receipt constructor.
- `postconditions` accepts exactly `process.exited_zero` and
  `process.cwd_matches`. Consume the matching receipt once; evidence expiry is
  checked original completion +1,000ms, exclusive (never renewed). Completion in
  the future/clock rollback, exact expiry and timestamp overflow refuse. Every
  postconditions call takes/consumes the receipt before validation, including a
  wrong request or predicate set; any refused execute clears previous success.
  Require fresh identities, reaped
  zero exit, complete bounded UTF8 stdout exactly canonical cwd plus newline,
  empty stderr and no cancellation/deadline/overflow/cleanup uncertainty. Commit
  request-bound receipt/predicate evidence as TrustedLocalAdapter.
- Output is JSON `{cwd: string, exit_code: 0}` with no extra properties. Bound
  serialized output including UTF8 escaping and metadata before spawn against
  compiled ceiling. No raw OS errors, command contents or secrets in diagnostics.

## Unix supervisor and limits

Use target-Unix workspace libc only, narrowly scoped documented unsafe syscall
wrappers. `pre_exec` does only async-signal-safe fchdir using the retained FD;
process_group(0), env_clear, null stdin, piped stdout/stderr. Nonblocking pipes
and one polling loop enforce fair fixed-size reads, aggregate cap and stop
checks. Effective deadline is minimum host deadline and configured timeout.

Observe leader exit without reaping (`waitid WNOWAIT`) so group kill precedes
leader reap and cannot target a reused PGID. Kill same-group descendants on
completion/error/timeout/cancellation/overflow; close pipes by cleanup deadline.
Use bounded reap polling; if not reaped, retain the child handle in the executor,
refuse another dispatch until reaped, and disclose uncertainty. No unbounded
reader threads or blocking final wait. Construction/spawn and an uninterruptible
kernel operation are not strict real-time guarantees. Drop cleanup is bounded.

Retained cwd pins the object but does not freeze its historical pathname. The
system-executable identity check is not atomic descriptor exec. Restrict the
public operation to the reviewed immutable system program; privileged concurrent
replacement remains outside the guarantee. Groups do not contain escaped
setsid/setpgid descendants; host SIGKILL can orphan a child. This is not an OS
sandbox, arbitrary-process containment or crash-proof child lifetime service.
Actual macOS tests do not establish Linux/Windows runtime acceptance.

## Acceptance

Core tests cover bounds/source/resource/prepared mismatch, empty input, repeated
receipt consumption, expired receipt and wrong request, cwd rename/symlink
substitution, actual permission refusal/restoration where effective identity allows,
stale upstream authority, permissive schemas, exact receipt expiry, clock rollback,
wrong receipt request/predicates and refusal-after-success. Test private instance-owned
kill/wait/delayed-reap injection: bounded uncertainty without successful receipt,
no second launch while a child is retained, reuse only after confirmed reap.
Also cover zero inherited env, exact pwd output, invalid UTF8/nonzero/no output, aggregate
stdout+stderr overflow, continuous output, cancellation/deadline, and ordinary
same-group descendants retaining pipes. Private helper modes exercise reuse;
no public arbitrary command API. Use parent watchdogs so broken supervision cannot
hang the suite. Check actual direct-child lifetime/reap, not only return labels.

Host integration uses scratch local approvals/lifecycle, exact A1 approval, real
pwd once and actual verified postconditions. Missing/denied/expired/changed
approval, changed source/resource/principal and cancellation cause zero launches;
replay never launches again. Use counter at adapter entry, separate from success.
No live WDBX store or sibling source edits. Full host suite, Clippy, size/fmt,
complete ABI gate and docs policy/validation precede local qualification.
