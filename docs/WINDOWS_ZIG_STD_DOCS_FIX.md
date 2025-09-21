# Fix: `zig std` Not Loading in Windows Browsers

**Comprehensive Guide, Workarounds, Diagnostics & Ready-to-Run Scripts**

This guide helps you restore access to **Zig standard library documentation** on Windows when `zig std` opens your browser but the page stays blank (spinner forever) or errors out. You’ll get:

* A fast path to working docs.
* Version switching that won’t wreck your PATH.
* A WSL-based fallback that keeps your Windows browser.
* Deep diagnostics (PowerShell + browser tools) to confirm root cause.
* Polished helper scripts (PowerShell + CMD) you can drop into your dotfiles.
* Extra remedies for browser quirks, AV/firewall interference, and environment issues.

> **Scope.** This focuses on Windows symptoms where the docs SPA fails to load and the CLI may log a `sources.tar` serve error or connection reset. Linux and macOS typically work; WSL is a practical workaround on Windows.

---

## Table of Contents

1. [TL;DR — fastest recovery](#tldr--fastest-recovery)
2. [Before you start (verify your setup)](#before-you-start-verify-your-setup)
3. [Option A — Switch Zig version on Windows (reliable quick fix)](#option-a--switch-zig-version-on-windows-reliable-quick-fix)

   * [One-shot PATH switch (session-only)](#one-shot-path-switch-session-only)
   * [Reusable PATH switcher function](#reusable-path-switcher-function)
   * [Permanent PATH change (optional)](#permanent-path-change-optional)
   * [SHA-256 integrity check (optional)](#sha-256-integrity-check-optional)
4. [Option B — Keep your Windows setup but serve docs from WSL](#option-b--keep-your-windows-setup-but-serve-docs-from-wsl)

   * [Install Zig in WSL](#install-zig-in-wsl)
   * [Run the docs server](#run-the-docs-server)
   * [Open in your Windows browser](#open-in-your-windows-browser)
   * [If localhost doesn’t forward](#if-localhost-doesnt-forward)
5. [Option C — Generate your project’s API docs as a fallback](#option-c--generate-your-projects-api-docs-as-a-fallback)
6. [Diagnostics — confirm the failure mode](#diagnostics--confirm-the-failure-mode)

   * [CLI probes (PowerShell)](#cli-probes-powershell)
   * [Network/socket checks](#networksocket-checks)
   * [Browser-side checks (DevTools)](#browser-side-checks-devtools)
7. [Common errors and quick remedies](#common-errors-and-quick-remedies)
8. [Browser and OS quirks to try](#browser-and-os-quirks-to-try)
9. [Security software & firewall notes](#security-software--firewall-notes)
10. [Self-Learning Doc Launcher (adaptive)](#self-learning-doc-launcher-adaptive)
11. [When to upgrade again](#when-to-upgrade-again)
12. [Appendix A — Handy scripts](#appendix-a--handy-scripts)
13. [Appendix B — Glossary of terms](#appendix-b--glossary-of-terms)

---

## TL;DR — fastest recovery

1. **Use a known-good Zig build** for Windows (e.g., `0.14.1`) **or** a newer Windows build that includes the fix when available.
2. Start the server without auto-opening the browser:

   ```powershell
   zig std --no-open-browser
   ```

   Copy the printed `http://127.0.0.1:<port>/` URL into your browser manually.
3. If you must stay on a problematic Windows build: **run `zig std` in WSL** and open the URL in your Windows browser. This avoids the Windows-side bug while preserving your browsing workflow.

---

## Before you start (verify your setup)

Run these in PowerShell to capture your baseline:

```powershell
zig version
zig env
```

* **Symptom snapshot:** When you run `zig std`, do you see an error mentioning `sources.tar`, `ConnectionResetByPeer`, or `ERR_EMPTY_RESPONSE` in the browser? Make a quick note.
* **Browser:** Use a modern Chromium/Firefox build. Try a private window to bypass extensions and aggressive caches.
* **Ports:** Ensure nothing else is colliding on the ephemeral port `zig std` chooses.
* **Environment:** Confirm your PATH points to the Zig you expect (no shadow copies). If multiple Zig versions are installed, prefer the switch methods below.

---

## Option A — Switch Zig version on Windows (reliable quick fix)

Keep multiple Zig versions side-by-side and toggle your PATH with zero drama. Example layout:

```
C:\tools\zig\0.14.1\zig.exe
C:\tools\zig\0.15.1\zig.exe
C:\tools\zig\dev\zig.exe
```

### One-shot PATH switch (session-only)

```powershell
# Set to your actual install dir for the working version
$zigDir = 'C:\tools\zig\0.14.1'
if (!(Test-Path "$zigDir\zig.exe")) { throw "zig.exe not found in $zigDir" }

# Remove any existing zig entries from PATH, then prepend the chosen one
$cleanPath = ($env:Path -split ';' | Where-Object { $_ -and ($_ -notlike '*\zig\*') }) -join ';'
$env:Path = "$zigDir;" + $cleanPath

zig version
zig std --no-open-browser  # copy the printed http://127.0.0.1:<port>/ into your browser
```

### Reusable PATH switcher function

Save as `zig-switch.ps1` and dot-source it in your profile, or run it ad-hoc:

```powershell
function Use-Zig {
  param(
    [Parameter(Mandatory=$true)][string]$Dir
  )
  if (!(Test-Path "$Dir\zig.exe")) { throw "zig.exe not found in $Dir" }
  $cleanPath = ($env:Path -split ';' | Where-Object { $_ -and ($_ -notlike '*\zig\*') }) -join ';'
  $env:Path = "$Dir;" + $cleanPath
  Write-Host "Using Zig:" -ForegroundColor Cyan
  & "$Dir\zig.exe" version
}

# Examples:
# Use-Zig 'C:\tools\zig\0.14.1'
# Use-Zig 'C:\tools\zig\0.15.1'
```

**Tip.** Keep a tiny `zig\` shim folder out of PATH confusion. The function above deliberately strips older `\zig\` segments before inserting the requested one.

### Permanent PATH change (optional)

To persist across new shells **for your user**:

```powershell
# Append once; new shells will see it
$zigDir = 'C:\tools\zig\0.14.1'
setx PATH ("$zigDir;" + $env:PATH)
```

> For **system-wide** PATH edits, use System Properties → Environment Variables. Prefer user-level changes unless you have a firm reason otherwise.

### SHA-256 integrity check (optional)

When you download a Zig build, verify its checksum:

```powershell
Get-FileHash 'C:\tools\zig\0.14.1\zig.exe' -Algorithm SHA256
```

Compare against the official checksum to rule out corruption.

---

## Option B — Keep your Windows setup but serve docs from WSL

`zig std`’s tiny HTTP server is generally stable on Linux. Running it in WSL while browsing from Windows sidesteps the Windows TCP reset while keeping your usual browser and extensions.

### Install Zig in WSL

Ubuntu example:

```bash
sudo apt update
sudo apt install zig -y   # Version may be older but is sufficient for `zig std`
# Alternatively, download a tarball from ziglang.org and extract to ~/zig
```

If you want `wslview` (to open links in Windows from WSL):

```bash
sudo apt install wslu -y
```

### Run the docs server

```bash
zig std --no-open-browser
```

Copy the printed `http://127.0.0.1:<port>/`.

### Open in your Windows browser

* Paste the URL into Edge/Chrome/Firefox on Windows, or
* From WSL: `wslview http://127.0.0.1:<port>/` (opens your default Windows browser).

### If localhost doesn’t forward

Modern WSL forwards `127.0.0.1` by default. If something is custom in your setup:

```bash
hostname -I     # take the first IPv4, e.g., 172.19.224.1
```

Then open `http://<that-ip>:<port>/` from Windows.

---

## Option C — Generate your project’s API docs as a fallback

This doesn’t replace stdlib docs, but it guarantees offline **project** documentation:

```bash
zig build -femit-docs
# Open the generated docs under zig-out/
```

You can also publish these to a docs server or bundle them in CI artifacts for teammates.

---

## Diagnostics — confirm the failure mode

### CLI probes (PowerShell)

```powershell
zig version
zig env
zig std --no-open-browser
```

* If the browser is blank or spins forever and the terminal logs an error about `sources.tar` or a **connection reset**, you’re likely in the known failure mode.

* Try retrieving the app shell and sources directly:

```powershell
# Replace 127.0.0.1:PORT from the printed URL
Invoke-WebRequest http://127.0.0.1:PORT/ -UseBasicParsing |
  Select-Object -ExpandProperty Content | Select-String -SimpleMatch '<title'

Invoke-WebRequest http://127.0.0.1:PORT/sources.tar -UseBasicParsing -OutFile $env:TEMP\zig_sources.tar
```

If `sources.tar` download fails with a **connection reset**, the server/browser handshake is failing on Windows.

### Network/socket checks

Ensure the docs server is actually listening and not colliding with another process:

```powershell
# Show listeners and find the one on the docs port
netstat -ano | Select-String ':PORT'
# If needed, map PID → process name
Get-Process -Id <PID>
```

### Browser-side checks (DevTools)

Open DevTools → **Network** and reload:

* `GET /` should be **200 OK**.
* `GET /sources.tar` should progress and complete.
* If you see `(failed) net::ERR_EMPTY_RESPONSE` or `NS_ERROR_NET_RESET`, you’re in the failure path.

> Also check **Console** for blocked resources, extensions injecting content, or service-worker errors. Try a private/incognito window to bypass extensions.

---

## Common errors and quick remedies

* **`ConnectionResetByPeer` in CLI; `ERR_EMPTY_RESPONSE` in browser**
  Likely the Windows-side bug path. Use **Option A** (switch Zig) or **Option B** (serve via WSL).

* **Blank page, no network activity**
  The auto-open might be misfiring. Start with `--no-open-browser` and paste the URL manually.

* **Inconsistent results across browsers**
  Disable extensions, clear cache for `http://127.0.0.1:*`, and try a private window. Hardware acceleration toggles can also affect dev servers.

* **Port appears in use**
  Another process is binding that port. Re-run `zig std` (it will choose a new port) or terminate the conflicting process.

---

## Browser and OS quirks to try

* **Private/Incognito window** to kill extensions and cached service-workers.
* **Hard reload** with cache disabled (Ctrl+F5 / Shift+Reload with DevTools open).
* **Hardware acceleration toggle** (on/off) in Edge/Chrome if rendering is funky.
* **New profile** (fresh user data directory) to rule out profile corruption.
* **DNS/Proxy**: Ensure no system proxy or DNS filter is interfering with localhost.

---

## Security software & firewall notes

The docs server binds to `127.0.0.1` and should be safe, but some tools are over-protective:

* **Windows Defender / third-party AV** may intercept or scan archive streams aggressively. Temporarily exclude your Zig folder and the listening port to test.
* **Firewalls**: Ensure outbound HTTP to localhost isn’t blocked by corporate policy or security suites. Create an allow rule for the Zig binary if needed.
* **Controlled Folder Access** (Windows Security) can block unexpected executables. If enabled, add an allow rule for `zig.exe` while you test.

---

## Self-Learning Doc Launcher (adaptive)

Make the launcher **adapt** based on past success/failure. The function below tries a ranked list of Zig binaries, verifies the docs server is healthy, **remembers what worked**, and prefers that path next time. It stores a tiny state file under `%LOCALAPPDATA%\ZigDocs\state.json`.

> **Privacy note.** State is local-only and contains: timestamps, selected strategy, Zig path, last URL, and simple counters. Delete the file to reset.

### Heuristics

* Try **preferred** Zig first (from state). If missing or failing, fall back to the rest.
* A launch is **successful** if `GET /` and `GET /sources.tar` both succeed.
* After repeated failures on Windows, **suggest WSL** and remember that suggestion.

### PowerShell: adaptive launcher

```powershell
function Start-ZigDocsAdaptive {
  param(
    [string[]]$Candidates = @(
      'C:\\tools\\zig\\0.15.2\\zig.exe',
      'C:\\tools\\zig\\0.14.1\\zig.exe',
      'zig'  # fallback to PATH
    ),
    [int]$ProbeTimeoutSec = 10
  )

  $stateDir = Join-Path $env:LOCALAPPDATA 'ZigDocs'
  $statePath = Join-Path $stateDir 'state.json'
  if (!(Test-Path $stateDir)) { New-Item -ItemType Directory -Path $stateDir | Out-Null }

  $state = if (Test-Path $statePath) { Get-Content $statePath -Raw | ConvertFrom-Json } else { @{ preferred = $null; history = @() } }

  # Re-rank candidates with preferred first if present
  if ($state.preferred -and ($Candidates -contains $state.preferred)) {
    $Candidates = @($state.preferred) + ($Candidates | Where-Object { $_ -ne $state.preferred })
  }

  function Save-State($ok, $exe, $url) {
    $entry = [ordered]@{
      ts = (Get-Date).ToString('o')
      ok = $ok
      exe = $exe
      url = $url
    }
    $hist = @($state.history) + ,$entry
    $state | Add-Member -Force NoteProperty history $hist
    if ($ok) { $state | Add-Member -Force NoteProperty preferred $exe }
    ($state | ConvertTo-Json -Depth 5) | Set-Content -Path $statePath -Encoding UTF8
  }

  function Start-ServerAndGetUrl($exe) {
    $tmpOut = [System.IO.Path]::GetTempFileName()
    $p = Start-Process -FilePath $exe -ArgumentList @('std','--no-open-browser') -RedirectStandardOutput $tmpOut -NoNewWindow -PassThru

    $url = $null
    $deadline = (Get-Date).AddSeconds($ProbeTimeoutSec)
    while ((Get-Date) -lt $deadline -and -not $url) {
      Start-Sleep -Milliseconds 200
      if (Test-Path $tmpOut) {
        $text = Get-Content $tmpOut -Raw -ErrorAction SilentlyContinue
        if ($text -match 'http://127\.0\.0\.1:\d+/') { $url = $Matches[0] }
      }
      if ($p.HasExited) { break }
    }

    if (-not $url) {
      try { if (-not $p.HasExited) { $p.Kill() } } catch {}
      return @{ url = $null; proc = $null; tmp = $tmpOut }
    }
    return @{ url = $url; proc = $p; tmp = $tmpOut }
  }

  function Probe-Server($url) {
    try {
      $null = Invoke-WebRequest "$url" -UseBasicParsing -TimeoutSec $ProbeTimeoutSec
      $tmpTar = Join-Path $env:TEMP 'zig_sources_probe.tar'
      $r = Invoke-WebRequest (Join-Path $url 'sources.tar') -OutFile $tmpTar -UseBasicParsing -TimeoutSec $ProbeTimeoutSec
      return $true
    } catch { return $false }
  }

  foreach ($exe in $Candidates) {
    try {
      $server = Start-ServerAndGetUrl -exe $exe
      if (-not $server.url) { Save-State $false $exe $null; continue }

      if (Probe-Server -url $server.url) {
        Save-State $true $exe $server.url
        Start-Process $server.url
        Write-Host "Docs up via $exe → $($server.url)" -ForegroundColor Green
        return
      } else {
        Save-State $false $exe $server.url
        try { if ($server.proc -and -not $server.proc.HasExited) { $server.proc.Kill() } } catch {}
      }
    } catch {
      Save-State $false $exe $null
    }
  }

  Write-Warning 'Windows attempts failed. Consider serving from WSL (see Option B).'
}
```

**Usage:**

```powershell
Start-ZigDocsAdaptive
# or specify your own candidate list
Start-ZigDocsAdaptive -Candidates @('C:\\tools\\zig\\0.14.1\\zig.exe','zig')
```

> The function captures the server URL, probes the home page and `sources.tar`, records success/failure, and remembers the best working Zig for next time.

---

## When to upgrade again

As soon as a Windows build includes the fix, switch back using the PATH helper and re-test `zig std`. Keep the WSL fallback as a “travel-safe” option; it’s reliable and doesn’t interfere with your Windows environment.

---

## Appendix A — Handy scripts

### PowerShell: quick doc launcher

Add this to your PowerShell profile:

```powershell
function Zig-Docs {
  param([string]$ZigExe = "zig")
  & $ZigExe std --no-open-browser | ForEach-Object {
    $_
    if ($_ -match 'http://127\.0\.0\.1:\d+/') {
      $url = $Matches[0]
      Start-Process $url
    }
  }
}

# Usage:
# Zig-Docs                  # uses zig on PATH
# Zig-Docs 'C:\\tools\\zig\\0.14.1\\zig.exe'
```

This launches the docs server, captures the printed URL, and opens it automatically in your default browser without relying on shell auto-open.

### PowerShell: version switcher (reusable)

```powershell
function Use-Zig {
  param(
    [Parameter(Mandatory=$true)][string]$Dir
  )
  if (!(Test-Path "$Dir\zig.exe")) { throw "zig.exe not found in $Dir" }
  $cleanPath = ($env:Path -split ';' | Where-Object { $_ -and ($_ -notlike '*\zig\*') }) -join ';'
  $env:Path = "$Dir;" + $cleanPath
  Write-Host "Using Zig:" -ForegroundColor Cyan
  & "$Dir\zig.exe" version
}
```

### CMD: minimal session-only Zig switcher

Save as `zig_switch.cmd` somewhere convenient and run it in `cmd.exe`:

```bat
@echo off
setlocal ENABLEDELAYEDEXPANSION
set ZIGDIR=C:\tools\zig\0.14.1
if not exist "%ZIGDIR%\zig.exe" (
  echo zig.exe not found in %ZIGDIR%
  exit /b 1
)

REM Remove existing zig entries from PATH (best-effort)
set CLEANPATH=
for %%I in (%PATH:;= % ) do (
  echo %%~I | findstr /I /C:"\zig\" >nul
  if errorlevel 1 (
    if defined CLEANPATH (
      set CLEANPATH=!CLEANPATH!;%%~I
    ) else (
      set CLEANPATH=%%~I
    )
  )
)
set PATH=%ZIGDIR%;!CLEANPATH!

zig version
zig std --no-open-browser
endlocal
```

---

## Appendix B — Glossary of terms

* **SPA (Single-Page Application):** The std docs UI is a small web app that loads once and fetches additional resources (like `sources.tar`).
* **`sources.tar`:** An archive streamed by `zig std` that the docs UI unpacks in the browser to populate symbols and search.
* **Localhost (127.0.0.1):** The loopback network interface; traffic never leaves your machine.
* **WSL:** Windows Subsystem for Linux — provides a Linux userland inside Windows.
* **PATH:** The list of directories the shell searches to find executables like `zig.exe`.

---

**You’re set.** With the version toggle, WSL fallback, adaptive launcher, and diagnostics above, you can get `zig std` docs up reliably and keep coding while upstream fixes land.
