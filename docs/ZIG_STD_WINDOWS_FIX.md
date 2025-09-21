# Fix: `zig std` not showing in browser on Windows — quick fixes & scripts

This guide gets your **Zig standard library docs** working again on Windows when `zig std` opens a tab but the page stays blank or spins forever. It includes a quick workaround, a more robust path switcher, WSL fallback, and a diagnostic checklist.

---

## TL;DR (fastest path)

1. **Use a known-good Zig build** for Windows (e.g., `0.14.1`) **or** a Windows build that includes the fix (e.g., `0.15.2+` when available).
2. Launch docs with:

   ```powershell
   zig std --no-open-browser
   ```

   Copy the printed `http://127.0.0.1:<port>/` into your browser manually.
3. If you must stay on a broken Windows build: **run `zig std` from WSL** and open the URL in your Windows browser (details below).

---

## Option A — Switch Zig version on Windows (temporary but reliable)

Keep multiple Zig versions side-by-side and toggle your PATH. Example folder layout:

```
C:\tools\zig\0.14.1\zig.exe
C:\tools\zig\0.15.1\zig.exe
C:\tools\zig\dev\zig.exe
```

### One-shot PATH switch (current PowerShell session only)

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

**Tip:** Keep a small `zig\` shim folder out of PATH confusion. The function above deliberately strips older `\zig\` entries before inserting the requested one.

---

## Option B — Stay on your current Zig but use WSL to serve docs

`zig std`’s local HTTP server typically works fine on Linux. We can exploit that via WSL and still view docs in your Windows browser.

1. **Install Zig in WSL** (Ubuntu example):

   ```bash
   sudo apt update
   sudo apt install zig -y   # Version may be older but is sufficient for `zig std`
   # Alternatively, download a tarball from ziglang.org and extract to ~/zig
   ```

2. **Run the docs server in WSL**:

   ```bash
   zig std --no-open-browser
   ```

   Copy the printed `http://127.0.0.1:<port>/`.

3. **Open from Windows**:

   * Paste the URL into Edge/Chrome/Firefox on Windows.
   * Or from WSL: `wslview http://127.0.0.1:<port>/` to open the default Windows browser.

> Note: On modern Windows/WSL, localhost port forwarding works out of the box. If it doesn’t, substitute the WSL IP (e.g., `hostname -I`) or `127.0.0.1` usually still forwards correctly.

---

## Option C — Generate project docs as a fallback

While this doesn’t replace stdlib docs, it ensures you can browse **your project’s** API offline:

```bash
zig build -femit-docs
# Open the generated docs under zig-out/
```

---

## Diagnostics (to confirm you’re hitting the known failure mode)

Run from PowerShell and note outputs:

```powershell
zig version
zig env
zig std --no-open-browser
```

* **Symptom:** Browser stays blank or spinner; terminal may show an error about serving `sources.tar` or a connection reset.

* **Check local port:**

  ```powershell
  # Replace 127.0.0.1:PORT from the printed URL
  iwr http://127.0.0.1:PORT/ -UseBasicParsing | select -Expand Content | sls '<title' -SimpleMatch
  iwr http://127.0.0.1:PORT/sources.tar -UseBasicParsing -OutFile $env:TEMP\zig_sources.tar
  ```

  If `sources.tar` download fails with a connection reset, you’re on the bug path.

* **Browser bypass:** Always try the manual URL (with `--no-open-browser`) to eliminate weirdness with shell-launched browser protocols.

* **Security software:** If you run third-party antivirus/firewall, briefly test with it disabled or whitelisted; the docs server binds `127.0.0.1` only.

---

## When to upgrade again

As soon as a Windows build has the fix (e.g., `0.15.2+`), switch back using the PATH helper and re-test `zig std`. Keep the WSL fallback in your pocket for travel-safe demos.

---

## Appendix — Tiny one-liner “doc launcher”

Add this to your PowerShell profile for convenience:

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

This launches the docs server, captures the printed URL, and opens it automatically in your default browser without relying on the shell’s auto-open behavior.
