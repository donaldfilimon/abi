# 🛠️ Fix: `zig std` Not Showing in Browser on Windows

> **Get Zig standard library docs working again when Windows builds open a blank browser tab.**

---

## ⚡ TL;DR (Fastest Path)

1. **Pick a known-good Zig build** (Windows `0.14.1` or any build with the fix such as `0.15.2+`).
2. Launch docs manually so you control the URL:
   ```powershell
   zig std --no-open-browser
   ```
   Copy the printed `http://127.0.0.1:<port>/` into your browser.
3. Stuck on a broken build? **Serve the docs from WSL** instead and open the printed URL in Windows.

> **Affected versions:** Windows builds between `0.15.0` and the fixed `0.15.2+` have been reported to hit this issue. `zig std` on Linux (including WSL) continues to work normally.

---

## 🔁 Option A — Switch Zig Version on Windows

Keep multiple Zig versions side-by-side and toggle your `PATH`. Suggested layout:

```
C:\tools\zig\0.14.1\zig.exe
C:\tools\zig\0.15.1\zig.exe
C:\tools\zig\dev\zig.exe
```

### One-Shot PATH Switch (Current PowerShell Session)

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

### Reusable PATH Switcher Function

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

> **Tip:** The helper strips older `\zig\` entries before inserting the requested one to avoid PATH confusion. Run `Use-Zig` again with your preferred default version once Windows builds regain the fix.

---

## 🐧 Option B — Stay on Your Build but Serve Docs from WSL

`zig std`'s HTTP server behaves correctly on Linux, so use WSL as a shim.

1. **Install Zig in WSL (Ubuntu example):**
   ```bash
   sudo apt update
   sudo apt install zig -y
   # Alternatively, download a tarball from ziglang.org and extract to ~/zig
   ```
2. **Run the docs server in WSL:**
   ```bash
   zig std --no-open-browser
   ```
   Copy the printed `http://127.0.0.1:<port>/`.
3. **Open from Windows:**
   - Paste the URL into Edge/Chrome/Firefox on Windows.
   - Or run `wslview http://127.0.0.1:<port>/` to launch your default Windows browser directly.
4. **Optional — expose a stable port:**
   ```bash
   zig std --listen 0.0.0.0 --port 9000 --no-open-browser
   ```
   Using an explicit port avoids re-copying the URL every run; just adjust firewall rules if prompted.

> Modern Windows/WSL forwards localhost automatically. If it fails, substitute the WSL IP (e.g., `hostname -I`) or `localhost` on the Windows side.

---

## 📦 Option C — Generate Project Docs as a Fallback

While not a substitute for stdlib docs, you can browse your project API offline:

```bash
zig build -femit-docs
# Open the generated docs under zig-out/
```

---

## 🧪 Diagnostics Checklist

Confirm you're hitting the known failure mode before switching versions.

```powershell
zig version
zig env
zig std --no-open-browser
```

- **Symptom:** Browser stays blank or spins; terminal may log an error while serving `sources.tar` or a connection reset.
- **Check local port:**
  ```powershell
  # Replace 127.0.0.1:PORT with the printed address
  iwr http://127.0.0.1:PORT/ -UseBasicParsing | select -Expand Content | sls '<title' -SimpleMatch
  iwr http://127.0.0.1:PORT/sources.tar -UseBasicParsing -OutFile $env:TEMP\zig_sources.tar
  ```
  If the `sources.tar` download fails with a reset, you're on the buggy path.
- **Browser bypass:** Always try the manual URL (`--no-open-browser`) to avoid protocol handler issues.
- **Security software:** Temporarily disable or whitelist third-party antivirus/firewall utilities. The docs server binds to `127.0.0.1` only.
- **Confirm the process is running:**
  ```powershell
  Get-NetTCPConnection -LocalPort PORT -OwningProcess (Get-Process zig).Id
  ```
  If no listener is present, the embedded HTTP server may have crashed—restart `zig std` and check for panic output.

---

## ⏭️ When to Upgrade Again

As soon as a Windows build ships with the fix (e.g., `0.15.2+`), switch back via the PATH helper and re-test `zig std`. Keep the WSL fallback ready for demos or travel scenarios.

---

## 🧰 Appendix — Tiny One-Liner Doc Launcher

Add to your PowerShell profile for convenience:

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

This launches the docs server, captures the printed URL, and opens it in your default browser without relying on the shell's auto-open behavior. Combine it with the PATH switcher to quickly swap versions, start the server, and restore your preferred default.
