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

3. Follow the on-screen menu. Choose **option 6** for the full automated pipeline or drill into individual steps for targeted actions.

The module stores its adaptive state under `%LOCALAPPDATA%\ZigDocs\state.json` and keeps execution logs in `%LOCALAPPDATA%\ZigDocs\logs`. Delete these files if you want to start fresh.

---

## Pipeline Architecture

| Stage | What Happens | Key Commands | Saved Signals |
| --- | --- | --- | --- |
| 1. Environment Verification | Confirms Zig availability, PATH health, and configuration | `zig version`, `zig env` | `lastSuccess`, `failureStreak`, timestamped history |
| 2. Version Switching & PATH Management | Selects or adds Zig installations, rewrites PATH without touching the registry | `Use-Zig` equivalent logic via menu option 2 | Preferred Zig locations |
| 3. Docs Server (Windows) | Starts `zig std --no-open-browser`, auto-detects the port, and opens your browser | `zig std --no-open-browser` | `lastPort`, preferred browser |
| 4. Docs Server (WSL) | Runs the same command via `wsl`, ideal when Windows builds fail | `wsl zig std --no-open-browser` | Host/port captured from WSL output |
| 5. Diagnostics & Network Tests | Probes `sources.tar`, runs `Test-NetConnection`, and suggests DevTools checks | `Invoke-WebRequest`, `Test-NetConnection` | Probe success/failure entries |
| 6. Adaptive Feedback | Consolidates history, warns after repeated failures, and recommends fallbacks | Internal state machine | Menu summaries, failure streak |

The **Full Automated Pipeline** (menu option 6) chains stages 1 → 3 → 5, giving you a one-command recovery path. You can interrupt and resume at any stage because state and logs persist across sessions.

---

## Installing Zig Builds for the Pipeline

* Download official Zig archives to `C:\tools\zig\<version>` (or a location of your choice).
* Add each folder—or its `zig.exe`—to the candidate list when prompted by menu option 2.
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

1. **Check Environment** – Runs `zig version` and `zig env`, logging the results.
2. **Switch Zig Version** – Lets you pick or add Zig locations, then refreshes PATH.
3. **Run Docs Server (Windows)** – Starts the Windows binary, detects the port, and launches your preferred browser.
4. **Run Docs Server (WSL)** – Invokes Zig inside WSL and opens the resulting URL.
5. **Diagnostics & Network Tests** – Downloads `sources.tar`, runs `Test-NetConnection`, and reminds you to inspect browser DevTools.
6. **Full Automated Pipeline** – Executes options 1 → 3 → 5 sequentially.
7. **Set Preferred Browser** – Persists the browser command (e.g., `msedge.exe --inprivate`).
8. **Exit** – Saves state and quits.

After each action the module updates its history pane. Multiple failures trigger yellow warnings nudging you toward the WSL fallback.

---

## PowerShell Module Internals

The script is implemented in `tools/zig_docs_interactive.ps1` and is designed for PowerShell 5.1+ or PowerShell 7.x on Windows.

```powershell
# tools/zig_docs_interactive.ps1 (excerpt)
function Zig-DocsInteractive {
  param([string[]]$ZigCandidates = @("zig"))

  $storage = Get-ZigDocsStorage
  $state = Load-ZigDocsState -StatePath $storage.StatePath -ZigCandidates $ZigCandidates
  $state | Add-Member -MemberType NoteProperty -Name BaseDir -Value $storage.BaseDir -Force

  $zigExe = Resolve-ZigExecutable -Candidates $state.zigCandidates

  while ($true) {
    Clear-Host
    Write-Host "Zig Docs Interactive Menu" -ForegroundColor Cyan
    # ... menu rendering ...
    switch (Read-Host "Select an option") {
      '1' { Invoke-ZigDocsEnvironmentCheck -ZigExe $zigExe -LogPath $storage.LogPath }
      '2' { $zigExe = Invoke-ZigDocsVersionSwitch -State $state -LogPath $storage.LogPath }
      '3' { Invoke-ZigDocsServerWindows -ZigExe $zigExe -State $state -LogPath $storage.LogPath }
      '4' { Invoke-ZigDocsServerWsl -State $state -LogPath $storage.LogPath }
      '5' { Invoke-ZigDocsDiagnostics -State $state -LogPath $storage.LogPath }
      '6' { Invoke-ZigDocsAutomatedPipeline -State $state -ZigExe $zigExe -LogPath $storage.LogPath }
      '7' { $state.preferredBrowser = Read-Host "Browser command or blank to clear" }
      '8' { break }
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

* **State file (`state.json`)** – Captures `lastSuccess`, `failureStreak`, Zig candidates, preferred browser, and the last known docs port.
* **History array** – Stores the latest 50 actions with timestamps and success flags. The menu summarizes the last five entries at the top of every screen.
* **Logs** – Daily log files under `%LOCALAPPDATA%\ZigDocs\logs` record each action, making it easier to correlate PowerShell output with browser issues.

Delete the state file if you want to reset recommendations, or archive the logs to share with teammates when debugging.

---

## Automated Pipeline Output & Next Steps

When you run option 6, the module reports each stage:

```text
Running automated pipeline...
✔ Environment check passed
✔ Docs server running at http://127.0.0.1:39999/
✔ Diagnostics completed (sources.tar reachable)
```

If the Windows server fails, the failure streak increments and the next menu render will highlight the WSL fallback. Select option 4 to launch the same docs server under Linux userland while still using your Windows browser.

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
```

Use these snippets when scripting CI jobs or when running on machines where you cannot persist state.

---

## Browser & Security Considerations

* Launch the docs in private/incognito mode to bypass extensions.
* Add `zig.exe` to antivirus and firewall allow-lists if connections keep resetting.
* If loopback traffic is blocked, use the WSL-reported IP address instead of `127.0.0.1`.
* Hardware acceleration toggles and clean browser profiles remain effective when the SPA shell loads but fails to retrieve `sources.tar`.

---

## Conclusion

With the unified pipeline and interactive CLI/UX module, `zig std` on Windows evolves from a brittle workflow into a guided, stateful experience. The module adapts to repeated failures, keeps comprehensive logs, and offers a one-command automated recovery, ensuring you can always reach the Zig standard library documentation with minimal friction.
