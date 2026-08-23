---
name: aggressive-macos-cleanup
description: Use when a macOS user asks to free substantial storage, delete caches or duplicates, empty Trash, manage Time Machine snapshots, consolidate repositories, or reorganize files across the home directory.
---

# Aggressive macOS Cleanup

## Overview

Be aggressive about proving what is disposable, not about widening deletion
commands. A cleanup is complete only when exact targets, protected surfaces,
reclaimed bytes, residual blockers, and retained recovery paths are all
evidenced separately.

Load the machine's own home/disk, filing, iCloud-Git, and project-move guidance
before acting when those skills or files exist. Local instructions can narrow
this workflow further.

## Non-negotiable boundaries

Never:

- delete by filename similarity, equal dimensions, similar timestamps, or equal
  byte size alone;
- issue recursive deletion against a home directory, workspace root, broad
  `Library`, `/Applications`, `/Library`, mounted volume, unresolved variable,
  or glob;
- use `sudo`, `chmod`, `chown`, Full Disk Access, or another tool to bypass a
  permission boundary that has not been resolved to exact, authorized targets;
- mutate files inside Photos, Music, or another application-managed library
  package with filesystem-level deduplication;
- clean UV while its lock or mapped environments have holders, or remove
  plugin/model/package caches while a process runs from them;
- delete iCloud-managed repositories, force their materialization, or run
  `git gc`, `git prune`, `git repack`, `git fsck --lost-found`, or similar
  object-walking maintenance there;
- combine repositories because names or source trees look alike;
- delete a dirty, unpushed, remote-less, or worktree-bearing checkout;
- empty Trash as one undifferentiated action when it contains projects,
  documents, personal media, recovery directories, or root-owned bundles;
- count logical duplicate bytes, APFS clone extents, hard-linked names, Trash
  moves, or snapshot-pinned blocks as physical storage already released.
- mutate a mounted backup destination or any external volume unless the user
  separately names that exact volume and exact content as cleanup scope.

“Reinstallable,” “generated,” “under the user's home,” “already in Trash,” and
“the dependencies can be rehydrated” are triage signals, never sufficient
destructive evidence by themselves.

## Required workflow

### 1. Establish the authority and baseline

Read the user- and project-scope instructions. Confirm the startup Data volume,
mounted external volumes, Time Machine destination/status, local snapshots,
active builds, running apps, MCP/plugin hosts, sync providers, and repository
roots.

Keep discovery proportional: start with top-level size accounting and the most
likely high-yield roots. Do not recursively hash personal libraries, external
volumes, cloud providers, or the entire home merely to build a speculative
candidate list.

Record at least:

```zsh
/bin/df -k /System/Volumes/Data
/usr/bin/tmutil listlocalsnapshots /
/usr/bin/tmutil status
/sbin/mount
ps axww -o pid=,command=
```

Use direct `du -sk -- <exact-path>` measurements for candidate accounting.
Treat `df` as volume-level corroboration only: APFS snapshots, clones,
purgeable space, and asynchronous release can make it disagree with summed
deletions.

### 2. Classify every candidate

| Class | Evidence | Action |
|---|---|---|
| Safe now | Literal path; `lstat` type and volume boundary understood; owner understood; no holder; generated, redundant, or redownloadable; protected data excluded | Delete in a bounded batch and verify absence |
| Shutdown tier | Rebuildable, but a live process executes from or writes it | Stop the owning service only when authorized, then recheck holders |
| Review tier | Personal media, archives, app state, model weights, recovery material, or ambiguous provenance | Inventory and prove a canonical/recovery copy first |
| Protected | iCloud provider state, credentials, histories, active plugin state, library packages, live worktrees, sibling path dependencies | Preserve |
| Admin blocked | Exact target is proven disposable but ownership or supported tooling requires root | Preserve and report the exact privileged command and reversal |

Do not upgrade “recognizable cache” to safe-now. Reconcile registries and
installed versions where relevant, inspect process command lines, and check
scoped holders immediately before deletion. Delete the narrowest useful layer:
for example, Rust `target/debug/incremental` before a whole `target`, and an
ignored build tree before `node_modules`, SwiftPM checkouts, or prebuilts.

### 3. Deduplicate by content and storage identity

Keep application-managed packages and cloud-only placeholders out of direct
filesystem mutation. For standalone, fully local regular files:

1. Group by logical byte size.
2. Compute a strong whole-file digest such as SHA-256.
3. Byte-compare before permanent deletion. If a full comparison is impractical,
   defer the item instead of weakening the gate.
4. Compare macOS object metadata that can carry user meaning: resource forks,
   extended attributes, ACLs, flags, Finder tags, and relevant sidecars. A data
   fork hash alone does not prove the whole file object is interchangeable. If
   metadata differs and no canonical copy preserves the required state, defer.
5. Detect same-device/inode hard links so an existing alias is not counted as
   reclaimable duplication.
6. Treat APFS clones as potentially already shared; do not promise physical
   savings from logical equality.
7. Choose one canonical copy by this precedence: authoritative managed record
   confirmed healthy; user-designated stable location; strongest original
   provenance; non-Trash/non-staging location. If equally legitimate copies
   remain tied, preserve both until the user chooses. Never use lexical order,
   age, or path length as destructive authority.
8. Verify the retained copy immediately before deleting each redundant path.
9. Rehash, recompare, and recheck metadata after any concurrent change.

Do not replace active files with hard links or symlinks. Hard links couple
future edits; symlinks change application, sandbox, sync, backup, and portability
semantics. Similar-but-not-identical media belongs in a separate human review,
never an automated exact-duplicate pass.

### 4. Audit repositories before consolidation

For every candidate checkout, capture:

```zsh
git -C <repo> status --short --branch
git -C <repo> rev-parse HEAD
git -C <repo> remote -v
git -C <repo> branch -vv
git -C <repo> log --oneline --branches --not --remotes
git -C <repo> worktree list --porcelain
git -C <repo> stash list
git -C <repo> status --short --ignored
git -C <repo> submodule status --recursive
git -C <repo> lfs ls-files 2>/dev/null || true
```

If deletion depends on upstream recoverability, verify the remote currently
exists and contains the required commit or branch with a live remote query. A
configured URL or stale remote-tracking ref is not current recovery evidence.
Account for LFS objects, submodules, ignored/untracked files, credentials, and
other state not carried by ordinary Git commits. For a material or unique
checkout, prove a fresh temporary clone/restore can resolve the required refs
and assets before deleting the local copy.

- Different histories or products: keep separate and disambiguate names.
- Dirty or unique commits: preserve the source; remove only proven ignored
  artifacts, then park it as an explicit recovery checkout if appropriate.
- Confirmed redundant clean clone: delete only after canonical HEAD, branches,
  worktrees, stashes, untracked files, and remote recoverability are proven.
- Worktree: use Git's worktree topology; never remove its directory manually.
- Relative sibling dependency: preserve the entire required layout.

Repeat HEAD, porcelain status, stash, and worktree checks immediately before an
irreversible checkout deletion. Record a short identity decision: repository
purpose, canonical path, remote/history relation, consumers, and why files are
or are not being combined.

For a project move, verify source and destination device IDs, destination
absence, HEAD/status/worktree state before and after, absolute path consumers,
tool trust registries, and configuration syntax after targeted rewrites. Do not
rewrite historical conversation text merely because it mentions the old path.

### 5. Handle Trash item by item

Inventory top-level Trash entries with exact sizes and classify them using the
same rules as live files. A trashed checkout can still hold the only dirty
working tree or unpushed commit. Root-owned application bundles in a user's
Trash are an admin boundary, not permission to recurse with `sudo rm -rf`.

Delete exact proven entries. If a root-owned bundle is proven redownloadable
but cannot be removed without elevation, leave it in place and report a quoted,
exact path list for the user to remove with administrator authority.

### 6. Manage Time Machine explicitly

Snapshot deletion and backup disablement are separate decisions. Disabling
automatic backups does not itself reclaim meaningful immediate storage.

- List snapshots before and after.
- Delete or thin local snapshots only when the user explicitly names snapshots,
  thinning, or deletion of local rollback state. A generic “clean everything”
  or “be aggressive” does not qualify; a request that explicitly says “delete
  Time Machine snapshots” does.
- Use the installed `tmutil help` as the command authority.
- If `tmutil disable` requires root, do not bypass it. Report the exact command
  `sudo tmutil disable` and reversal `sudo tmutil enable`.
- Do not claim “snapshots are disabled forever.” macOS can create other APFS
  snapshots, and automatic Time Machine state must be verified independently.
- A successful snapshot command does not prove permanent backup disablement.

### 7. Treat command errors as evidence, not verdicts

A recursive removal can exit nonzero with many `ENOENT` messages while the
exact top-level target is nevertheless absent, especially when contents change
during traversal. Conversely, exit zero does not prove the intended scope.

After any noisy or failed destructive command:

1. wait for that command to finish;
2. do not launch a competing removal;
3. re-inventory the exact target and chained targets;
4. verify top-level absence/presence and measured residue;
5. retry only the remaining exact target, if still authorized.

Never widen the command or add force flags merely to silence diagnostics.

### 8. Verify every bounded batch

Before each batch, resolve every candidate with non-following metadata, confirm
it is not a symlink or unexpected mount crossing, and write the enumerated
literal path list to the cleanup ledger. Destructive commands must consume only
that literal list: no globs, recursive discovery substitution, unresolved
variables, or freshly discovered children. Prefer recoverable quarantine for
non-cache personal/review material; permanent deletion requires the same proof
at the final boundary.

For each batch, record:

- exact deleted, moved, retained, and blocked paths;
- direct KiB before and residual KiB after;
- whether removal is recoverable or redownloadable;
- snapshot count and Time Machine automatic-backup state;
- Data-volume available bytes, labeled as APFS-level corroboration;
- retained repository HEAD/status/worktree evidence;
- application, toolchain, and live-service health proportional to what changed.

Report gross selected bytes and measured retained reclaim separately when
concurrent tools can regenerate caches or artifacts.

## Quick reference

| Pressure | Correct counter |
|---|---|
| “Delete by filename” | Size group, strong hash, byte compare, storage identity |
| “It is in Trash” | Audit for unique personal data and repository state |
| “It is a cache” | Prove owner, registry role, holders, and regeneration cost |
| “It is generated” | Delete the narrow ignored layer after a same-moment process check |
| “Same repo name” | Compare remote, history, commits, dirt, and worktrees |
| “sudo is available” | Resolve exact targets; use supported scoped commands only |
| “Disable snapshots” | Separate existing snapshot deletion from automatic-backup state |
| “df increased” | Pair volume delta with direct-path accounting and snapshot state |
| “rm failed” | Inspect postconditions before retrying or widening |

## Example

A Mac has an 85 GiB `~/Music/Logic Pro Library.bundle`, a custom `.logicx`
project elsewhere, Logic itself only in Trash, ten local snapshots, and
root-owned trashed app bundles.

1. Verify the bundle is Apple's relocatable sound content, not the Music
   library or custom project; verify Logic is not installed/running and no
   process holds the bundle.
2. Measure the bundle directly and preserve the separate custom project.
3. Delete the exact user-owned bundle, then verify the top-level path is absent
   even if traversal reported `ENOENT` races.
4. Delete the ten listed local snapshots only under the user's explicit
   authorization; verify the new count.
5. Attempt no ownership workaround for the root-owned Trash apps. Report them
   as a separate admin-gated list.
6. Report direct bytes removed, Data-volume change, snapshot effect, custom
   project preservation, and the still-enabled/disabled Time Machine state as
   separate facts.

## Common mistakes

- Calling all `node_modules`, Swift `.build`, Rust `target`, model weights, or
  plugin caches useless because manifests can theoretically rebuild them.
- Using `df -h /` as the only success proof.
- Treating a Time Machine destination, local snapshots, and automatic backups
  as one state.
- Guessing a large application's storage path from product conventions.
- Running `sudo rm -rf` on an illustrative path or blanket Trash selection.
- Deleting a repository after `git status` alone without checking unique commits,
  untracked files, worktrees, stashes, and current remote recoverability.
- Moving a repo without repairing path consumers or validating config files.
- Calling a cleanup complete while exact admin-gated targets remain undisclosed.
- Treating a momentary no-holder result as service shutdown proof; verify the
  owning service/app is stopped, no supervisor is restarting it, then recheck.
