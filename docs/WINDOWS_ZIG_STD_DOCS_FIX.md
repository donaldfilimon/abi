# Unified Pipeline & Dynamic Interactive CLI/UX for `zig std` on Windows

**Complete End-to-End Pipeline with Integrated CLI Interfaces and Interactive UX Module**

This guide delivers a single, cohesive workflow for restoring access to the Zig standard library documentation on Windows. Instead of juggling scattered instructions, you now get a unified pipeline that validates your environment, rotates through Zig versions, launches the documentation server on Windows or WSL, executes diagnostics, and remembers what worked last time.

The interactive PowerShell module (`Zig-DocsInteractive`) orchestrates each stage through a dynamic menu. Every run captures logs, updates a persistent state file, and adapts its suggestions based on previous outcomes. The result is a continuous **verify → attempt → diagnose → adapt → learn** feedback loop.

---

## Quick Start

1. **Download the helper script** to a safe location (this repository ships it in `tools/zig_docs_interactive.ps1`).
2. **Launch an elevated PowerShell** (if you plan to edit PATH) and import the module:

   ```powershell
   Set-Location C:\path\to\repo
   .\tools\zig_docs_interactive.ps1
   Zig-DocsInteractive
   ```

3. Follow the on-screen menu. Option **1** replays the last known good configuration, option **2** cycles every detected Zig build automatically, option **3** lets you pick or add a version manually, option **4** triggers the WSL fallback, and option **5** opens the live health dashboard.

The module stores its adaptive state under `%LOCALAPPDATA%\ZigDocs\state.json` and keeps execution logs in `%LOCALAPPDATA%\ZigDocs\logs`. Delete these files if you want to start fresh.

---

## Pipeline Architecture

| Stage | What Happens | Key Commands | Saved Signals |
| --- | --- | --- | --- |
| 1. Candidate Discovery & Environment Snapshot | Auto-detects Zig binaries (`C:\tools\zig\**\zig.exe`, PATH, WSL), captures versions, and records the current locale. | `Get-ChildItem`, `zig version`, `zig env` | `zigCandidates`, `versionMetadata`, `lastStatus`, `locale` |
| 2. Windows Docs Launch | Starts `zig std --no-open-browser`, monitors stdout/stderr for regressions (`unable to serve /sources.tar`, port binding errors), and launches a browser with locale-aware arguments. | `zig std --no-open-browser`, browser CLI | `lastPort`, `lastSuccessfulVersion`, `preferredBrowser` |
| 3. Auto-Rotation & Heuristics | Option 2 cycles through detected versions, ranks them by last success and semantic version, and increments per-version failure counters (skipping those above the threshold). | `Invoke-ZigDocsWindowsSession`, failure counter updates | `versionFailures`, `failureThreshold`, `history` |
| 4. Health Dashboard & Self-Learning State | Option 5 renders the color-coded dashboard, lets you tweak thresholds, and manage browser preferences while reviewing the recent action log. | `Show-ZigDocsDashboard` | `totalSuccesses`, `totalFailures`, `preferredBrowser`, `history` |
| 5. WSL Fallback Heuristics | Option 4 launches `zig std` inside WSL, auto-detects mirrored networking vs. bridged IPs, and keeps state aligned with the Windows attempts. | `wsl.exe zig std`, `hostname -I` | `lastSuccessfulMode`, `lastPort` |

The auto-rotation workflow replaces the old pipeline option. You can trigger it any time via menu option 2, and the launcher remembers the last working setup so option 1 can replay it instantly.

---

## Installing Zig Builds for the Pipeline

* Download official Zig archives to `C:\tools\zig\<version>` (or a location of your choice).
* Add each folder—or its `zig.exe`—to the candidate list via the manual picker (menu option 3 → **A**) if the auto-discovery step does not find it automatically.
* The module sanitizes PATH entries so the selected Zig version always takes priority without permanently modifying the system PATH.

For checksums and release discovery, continue to rely on the official Zig download page. The pipeline assumes the binaries you provide are trusted.

---

## Interactive CLI/UX Walkthrough

### Launching the Menu

```powershell
# From a PowerShell session where zig_docs_interactive.ps1 is accessible
. .\tools\zig_docs_interactive.ps1
Zig-DocsInteractive -ZigCandidates @(
  "zig",                   # existing PATH resolution
  "C:\\tools\\zig\\0.14.1",
  "C:\\tools\\zig\\nightly"
)
```

The optional `-ZigCandidates` parameter seeds the selector with known installations. You can add new paths interactively at any time.

### Menu Options

1. **Run docs server with last known good version** – Replays the configuration that succeeded most recently, including PATH rewrites and browser launch parameters.
2. **Try all versions automatically** – Rotates through detected Zig builds, skipping those that exceeded the failure threshold until one succeeds.
3. **Choose a version manually** – Presents the ranked list, lets you launch any candidate immediately, and opens a mini-menu where **A** adds new folders/executables, **D** removes stale entries (including their failure counters), **R** rescans auto-detected paths, and **Enter** returns to the top-level menu.
4. **Run in WSL fallback** – Invokes Zig inside WSL, detects mirrored networking vs. bridged IPs, and launches the docs URL from Windows.
5. **Show health dashboard** – Displays success/failure statistics, per-version counters, preferred browser configuration, and allows adjusting the failure threshold.
6. **Exit** – Saves state and quits.

After each action the module updates its history pane, and the summary banner promotes the WSL fallback whenever repeated failures are detected.

---

## PowerShell Module Internals

The script is implemented in `tools/zig_docs_interactive.ps1` and is designed for PowerShell 5.1+ or PowerShell 7.x on Windows. Candidate discovery now avoids the PowerShell 7-only `Get-ChildItem -Depth` flag by performing a bounded breadth-first scan, so auto-detection works even on stock Windows 10 shells.

```powershell
# tools/zig_docs_interactive.ps1 (excerpt)
function Zig-DocsInteractive {
  param([string[]]$ZigCandidates = @("zig"))

  $storage = Get-ZigDocsStorage
  $state = Load-ZigDocsState -StatePath $storage.StatePath -ZigCandidates $ZigCandidates
  $state | Add-Member -MemberType NoteProperty -Name BaseDir -Value $storage.BaseDir -Force

  while ($true) {
    $candidates = Get-ZigDocsCandidates -State $state -AdditionalCandidates $ZigCandidates
    $sorted     = Sort-ZigDocsCandidates -State $state -Candidates $candidates

    Clear-Host
    Write-Host "Zig Docs Interactive" -ForegroundColor Cyan
    Write-Host "1. Run docs server with last known good version"
    Write-Host "2. Try all versions automatically"
    Write-Host "3. Choose a version manually"
    Write-Host "4. Run in WSL fallback"
    Write-Host "5. Show health dashboard"
    Write-Host "Q. Exit"
    Show-ZigDocsSummary -State $state -Candidates $sorted

    switch ((Read-Host "Select an option").ToUpperInvariant()) {
      '1' { Invoke-ZigDocsWindowsSession -State $state -Candidate (Get-ZigDocsLastGoodCandidate -State $state -Candidates $sorted) -LogPath $storage.LogPath }
      '2' { foreach ($candidate in $sorted) { if (-not $candidate.Skip) { if ((Invoke-ZigDocsWindowsSession -State $state -Candidate $candidate -LogPath $storage.LogPath).success) { break } } } }
      '3' { # manual picker + add new candidate }
      '4' { Invoke-ZigDocsServerWsl -State $state -LogPath $storage.LogPath }
      '5' { Show-ZigDocsDashboard -State $state }
      'Q' { break }
      default { Write-Host "Invalid selection" -ForegroundColor Red }
    }
  }

  Save-ZigDocsState -State $state -StatePath $storage.StatePath
  Write-Host "State saved to $($storage.StatePath)" -ForegroundColor Cyan
}
```

Supporting functions handle storage, logging, PATH rewrites, docs server orchestration, WSL fallbacks, and diagnostics. Review the full script for implementation details or customization.

---

## Adaptive State & Logging

* **State file (`state.json`)** – Tracks the last successful version and path, preferred execution mode (Windows vs. WSL), the failure counter per Zig version, the health dashboard threshold, preferred browser command, locale, and the most recent docs port.
* **History array** – Stores the latest 50 actions with timestamps and success flags. The menu summarizes recent entries and the health dashboard colour-codes streaks.
* **Logs** – Daily log files under `%LOCALAPPDATA%\ZigDocs\logs` record each action, making it easier to correlate PowerShell output with browser issues.

Delete the state file if you want to reset recommendations, or archive the logs to share with teammates when debugging.

---

## Automated Pipeline Output & Next Steps

When you run option 2, the module walks the detected versions in priority order (last known good first, then newest releases with the fewest failures) until it finds a Zig toolchain that can serve docs successfully:

```text
Testing C:\tools\zig\0.12.0\zig.exe
Docs server running at http://127.0.0.1:39999/
Testing C:\tools\zig\0.11.0\zig.exe
Docs server failed: Known Windows regression: sources.tar failure
```

Each failed attempt increments the per-version failure counter. Versions that exceed the configured threshold are skipped automatically until you lower the limit from the dashboard or record a later success. If the Windows server keeps failing, the summary banner recommends the WSL fallback (option 4), which uses mirrored networking when available and otherwise selects the correct IP from `hostname -I` inside the distribution.

---

## Manual Recovery (If You Need It)

The interactive experience wraps the following manual commands, which remain valid when you prefer a lighter touch:

```powershell
# Switch Zig version temporarily
$env:Path = "C:\\tools\\zig\\0.14.1;" + ($env:Path -split ';' | Where-Object { $_ -and ($_ -notlike '*\\zig\\*') }) -join ';'

# Launch docs without the module
zig std --no-open-browser

# Serve from WSL explicitly
wsl zig std --no-open-browser
