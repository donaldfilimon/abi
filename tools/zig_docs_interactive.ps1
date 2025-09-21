function Get-ZigDocsStorage {
  $baseDir = Join-Path $env:LOCALAPPDATA "ZigDocs"
  if (-not (Test-Path $baseDir)) {
    New-Item -ItemType Directory -Path $baseDir -Force | Out-Null
  }
  $logDir = Join-Path $baseDir "logs"
  if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
  }
  return [pscustomobject]@{
    StatePath = Join-Path $baseDir "state.json"
    LogPath   = Join-Path $logDir ("zig-docs-" + (Get-Date -Format "yyyyMMdd") + ".log")
    BaseDir   = $baseDir
  }
}

function Load-ZigDocsState {
  param(
    [string]$StatePath,
    [string[]]$ZigCandidates
  )

  $defaultState = [ordered]@{
    lastSuccess            = $null
    lastSuccessTimestamp   = $null
    lastSuccessfulVersion  = $null
    lastSuccessfulPath     = $null
    lastSuccessfulMode     = "Windows"
    failureStreak          = 0
    history                = @()
    preferredBrowser       = $null
    lastPort               = $null
    zigCandidates          = $ZigCandidates
    versionFailures        = @{}
    failureThreshold       = 3
    totalSuccesses         = 0
    totalFailures          = 0
    lastStatus             = "unknown"
    lastAttemptTimestamp   = $null
    versionMetadata        = @{}
    locale                 = "en-US"
  }

  if (Test-Path $StatePath) {
    try {
      $raw = Get-Content -LiteralPath $StatePath -Raw
      if ($raw.Trim().Length -gt 0) {
        $state = $raw | ConvertFrom-Json -ErrorAction Stop
        foreach ($key in $defaultState.Keys) {
          if (-not $state.PSObject.Properties.Name.Contains($key)) {
            $state | Add-Member -NotePropertyName $key -NotePropertyValue $defaultState[$key]
          }
        }
        if (-not $state.zigCandidates) { $state.zigCandidates = $ZigCandidates }
        $state.zigCandidates = @($state.zigCandidates | Where-Object { $_ })
        if (-not $state.versionFailures) { $state.versionFailures = @{} }
        elseif ($state.versionFailures -isnot [hashtable]) {
          $converted = @{}
          foreach ($prop in $state.versionFailures.PSObject.Properties) {
            $converted[$prop.Name] = $prop.Value
          }
          $state.versionFailures = $converted
        }
        if (-not $state.failureThreshold) { $state.failureThreshold = 3 }
        if (-not $state.totalSuccesses) { $state.totalSuccesses = 0 }
        if (-not $state.totalFailures) { $state.totalFailures = 0 }
        if (-not $state.lastStatus) { $state.lastStatus = "unknown" }
        if (-not $state.versionMetadata) { $state.versionMetadata = @{} }
        elseif ($state.versionMetadata -isnot [hashtable]) {
          $convertedMeta = @{}
          foreach ($prop in $state.versionMetadata.PSObject.Properties) {
            $convertedMeta[$prop.Name] = $prop.Value
          }
          $state.versionMetadata = $convertedMeta
        }
        if (-not $state.locale) { $state.locale = "en-US" }
        if (-not $state.lastSuccessfulMode) { $state.lastSuccessfulMode = "Windows" }
        if (-not $state.lastSuccessfulPath) { $state.lastSuccessfulPath = $null }
        if (-not $state.lastSuccessTimestamp) { $state.lastSuccessTimestamp = $null }
        return $state
      }
    } catch {
      Write-Warning "Failed to parse existing state file. A new state will be created."
    }
  }

  return ([pscustomobject]$defaultState)
}

function Save-ZigDocsState {
  param(
    [pscustomobject]$State,
    [string]$StatePath
  )

  $State | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

function Write-ZigDocsLog {
  param(
    [string]$LogPath,
    [string]$Message,
    [string]$Level = "INFO"
  )

  $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  $entry = "[$timestamp][$Level] $Message"
  Add-Content -LiteralPath $LogPath -Value $entry
}

function Wait-ZigDocsUser {
  param(
    [string]$Message = "Press Enter to continue..."
  )

  [void](Read-Host $Message)
}

function Get-ZigDocsPlatform {
  if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
    return "Windows"
  }
  if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Linux)) {
    return "Linux"
  }
  if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::OSX)) {
    return "macOS"
  }
  return "Unknown"
}

function Get-ZigDocsPathSeparator {
  if (Get-ZigDocsPlatform -eq "Windows") {
    return ';'
  }
  return ':'
}

function Get-ZigDocsLocale {
  param(
    [pscustomobject]$State
  )

  if ($State.locale) {
    return $State.locale
  }
  try {
    return ([System.Globalization.CultureInfo]::CurrentCulture.Name)
  } catch {
    return "en-US"
  }
}

function Get-ZigDocsLimitedDepthFiles {
  param(
    [string]$Root,
    [string]$TargetName,
    [int]$MaxDepth = 2
  )

  if (-not $Root) { return @() }

  $resolvedRoot = $null
  try {
    $resolvedRoot = (Resolve-Path -LiteralPath $Root -ErrorAction Stop).Path
  } catch {
    $resolvedRoot = $Root
  }

  $queue = New-Object System.Collections.Queue
  $queue.Enqueue([pscustomobject]@{ Path = $resolvedRoot; Depth = 0 })
  $seen = @{}
  $matches = @()

  while ($queue.Count -gt 0) {
    $current = $queue.Dequeue()
    if (-not $current.Path) { continue }
    if ($seen.ContainsKey($current.Path)) { continue }
    $seen[$current.Path] = $true

    try {
      $children = Get-ChildItem -LiteralPath $current.Path -Force -ErrorAction Stop
    } catch {
      continue
    }

    foreach ($child in $children) {
      if ($child.PSIsContainer) {
        if ($current.Depth -lt $MaxDepth) {
          $queue.Enqueue([pscustomobject]@{ Path = $child.FullName; Depth = $current.Depth + 1 })
        }
        continue
      }

      if ($child.Name -and $TargetName -and ($child.Name -ieq $TargetName)) {
        $matches += $child.FullName
      }
    }
  }

  return $matches
}

function Get-ZigDocsBinaryMatches {
  param(
    [string]$Root,
    [string]$BinaryName,
    [int]$MaxDepth = 2
  )

  if (-not $Root) { return @() }

  $items = @()
  $hasWildcards = $Root -match '[\*\?]'

  if ($hasWildcards) {
    try {
      $items = Get-ChildItem -Path $Root -Force -ErrorAction Stop
    } catch {
      return @()
    }
  } elseif (Test-Path $Root) {
    try {
      $items = @(Get-Item -LiteralPath $Root -Force -ErrorAction Stop)
    } catch {
      return @()
    }
  } else {
    return @()
  }

  $matches = @()

  foreach ($item in $items) {
    if (-not $item) { continue }
    if ($item.PSIsContainer) {
      $matches += Get-ZigDocsLimitedDepthFiles -Root $item.FullName -TargetName $BinaryName -MaxDepth $MaxDepth
    } elseif ($item.Name -and ($item.Name -ieq $BinaryName)) {
      $matches += $item.FullName
    }
  }

  return $matches
}

function Add-ZigDocsHistory {
  param(
    [pscustomobject]$State,
    [string]$Action,
    [bool]$Success,
    [string]$Details = "",
    [bool]$RecordStats = $true
  )

  $record = [ordered]@{
    timestamp = Get-Date -Format "o"
    action    = $Action
    success   = $Success
    details   = $Details
  }
  $State.history = @($State.history + ([pscustomobject]$record) | Select-Object -Last 50)

  if ($RecordStats) {
    $State.lastAttemptTimestamp = $record.timestamp
    if ($Success) {
      $State.failureStreak = 0
      $State.lastSuccess = $Action
      $State.lastStatus = "success"
      $State.totalSuccesses++
      $State.lastSuccessTimestamp = $record.timestamp
    } else {
      $State.failureStreak++
      $State.lastStatus = "failure"
      $State.totalFailures++
    }
  }
}

function Resolve-ZigCandidatePath {
  param(
    [string]$Candidate
  )

  if (-not $Candidate) { return $null }
  $platform = Get-ZigDocsPlatform
  $binaryName = if ($platform -eq "Windows") { "zig.exe" } else { "zig" }

  if ($Candidate -eq "zig") {
    $cmd = Get-Command zig -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return "zig"
  }

  if (-not (Test-Path $Candidate)) {
    return $null
  }

  $item = Get-Item $Candidate
  if ($item.PSIsContainer) {
    $exe = Join-Path $Candidate $binaryName
    if (Test-Path $exe) { return $exe }
    return $null
  }

  if ($item.Name -ieq $binaryName) { return $item.FullName }
  if ($platform -ne "Windows" -and $item.Extension -eq "") { return $item.FullName }
  return $null
}

function Get-ZigVersionInfo {
  param(
    [string]$ZigPath
  )

  if (-not $ZigPath) { return $null }
  try {
    $versionOutput = (& $ZigPath version 2>$null).Trim()
    if (-not $versionOutput) { return $null }
    $parsed = $null
    try {
      $parsed = [version]$versionOutput
    } catch {
      $parsed = $null
    }
    return [pscustomobject]@{
      Raw    = $versionOutput
      Parsed = $parsed
    }
  } catch {
    return $null
  }
}

function Get-ZigDocsCandidateKey {
  param(
    [string]$Path
  )

  if (-not $Path) { return $null }
  if ($Path -eq "zig") { return "zig" }
  try {
    return [System.IO.Path]::GetFullPath($Path)
  } catch {
    return $Path
  }
}

function Find-ZigDocsInstallations {
  param(
    [pscustomobject]$State,
    [string[]]$AdditionalCandidates
  )

  $platform = Get-ZigDocsPlatform
  $binaryName = if ($platform -eq "Windows") { "zig.exe" } else { "zig" }
  $results = New-Object System.Collections.Generic.List[string]

  foreach ($candidate in @($State.zigCandidates) + $AdditionalCandidates) {
    if ($candidate) { $results.Add($candidate) }
  }

  if ($platform -eq "Windows") {
    $searchRoots = @("C:\\tools\\zig")
    if ($env:ProgramFiles) {
      $searchRoots += @(Join-Path $env:ProgramFiles "zig", Join-Path $env:ProgramFiles "zig-*", Join-Path $env:ProgramFiles "Zig")
    }
    if (${env:ProgramFiles(x86)}) {
      $searchRoots += @(Join-Path ${env:ProgramFiles(x86)} "zig", Join-Path ${env:ProgramFiles(x86)} "zig-*")
    }
    if ($env:LOCALAPPDATA) {
      $searchRoots += @(Join-Path $env:LOCALAPPDATA "Programs\\zig")
    }
    $searchRoots = $searchRoots | Where-Object { $_ }

    foreach ($root in $searchRoots) {
      try {
        if (-not (Test-Path $root)) { continue }
        $items = Get-ChildItem -Path $root -Filter $binaryName -File -Recurse -Depth 2 -ErrorAction SilentlyContinue
        foreach ($item in $items) { $results.Add($item.FullName) }
        $matches = Get-ZigDocsBinaryMatches -Root $root -BinaryName $binaryName -MaxDepth 2
        foreach ($match in $matches) { $results.Add($match) }
      } catch {}
    }
  } else {
    $searchPaths = @(
      "/usr/local/bin/zig",
      "/usr/bin/zig",
      (Join-Path $HOME "zig"),
      (Join-Path $HOME ".local/bin/zig"),
      (Join-Path $HOME "tools/zig"),
      (Join-Path $HOME "Downloads/zig")
    ) | Where-Object { $_ }

    foreach ($path in $searchPaths) {
      if (Test-Path $path) { $results.Add($path) }
    }
  }

  return $results.ToArray() | Where-Object { $_ } | Select-Object -Unique
}

function Remove-ZigDocsCandidate {
  param(
    [pscustomobject]$State,
    [pscustomobject]$Candidate
  )

  if (-not $State -or -not $Candidate) { return }

  $key = $Candidate.Key
  if (-not $key) { return }

  $State.zigCandidates = @(
    $State.zigCandidates | ForEach-Object {
      $resolved = Resolve-ZigCandidatePath -Candidate $_
      $candidateKey = Get-ZigDocsCandidateKey -Path $resolved
      if ($candidateKey -ne $key) { $_ }
    }
  )

  if ($State.versionFailures.ContainsKey($key)) { $State.versionFailures.Remove($key) }
  if ($State.versionMetadata.ContainsKey($key)) { $State.versionMetadata.Remove($key) }

  if ($State.lastSuccessfulVersion -eq $key) {
    $State.lastSuccessfulVersion = $null
    $State.lastSuccessfulPath = $null
  }
}

function Get-ZigDocsCandidates {
  param(
    [pscustomobject]$State,
    [string[]]$AdditionalCandidates
  )

  $rawCandidates = Find-ZigDocsInstallations -State $State -AdditionalCandidates $AdditionalCandidates
  $platform = Get-ZigDocsPlatform
  $list = @()

  foreach ($candidate in $rawCandidates) {
    $resolved = Resolve-ZigCandidatePath -Candidate $candidate
    if (-not $resolved) { continue }
    $key = Get-ZigDocsCandidateKey -Path $resolved
    if (-not $key) { continue }
    $versionInfo = $null
    $versionRaw = $null
    $versionParsed = $null
    if ($State.versionMetadata.ContainsKey($key)) {
      $cached = $State.versionMetadata[$key]
      if ($cached -and $cached.version) {
        $versionRaw = $cached.version
        try { $versionParsed = [version]$cached.version } catch {}
      }
    }
    if (-not $versionRaw) {
      $info = Get-ZigVersionInfo -ZigPath $resolved
      if ($info) {
        $versionRaw = $info.Raw
        $versionParsed = $info.Parsed
      }
      if (-not $State.versionMetadata.ContainsKey($key)) {
        $State.versionMetadata[$key] = @{ path = $resolved; version = $versionRaw }
      } else {
        $State.versionMetadata[$key].path = $resolved
        $State.versionMetadata[$key].version = $versionRaw
      }
    }
    $failureCount = 0
    if ($State.versionFailures.ContainsKey($key)) {
      $failureCount = [int]$State.versionFailures[$key]
    }
    $list += [pscustomobject]@{
      Path          = $resolved
      Key           = $key
      Version       = $versionRaw
      VersionParsed = $versionParsed
      FailureCount  = $failureCount
      Skip          = ($failureCount -ge [int]$State.failureThreshold)
      Platform      = $platform
    }
  }

  return $list | Sort-Object -Property Path -Unique
}

function Sort-ZigDocsCandidates {
  param(
    [pscustomobject]$State,
    [pscustomobject[]]$Candidates
  )

  $ordered = $Candidates | Sort-Object -Property @(
    @{ Expression = { if ($_.Skip) { 1 } else { 0 } } },
    @{ Expression = { if ($State.lastSuccessfulVersion -and ($_.Key -eq $State.lastSuccessfulVersion)) { 0 } else { 1 } } },
    @{ Expression = { $_.FailureCount } },
    @{ Expression = { if ($_.VersionParsed) { $_.VersionParsed } else { [version]::new(0,0) } }; Descending = $true }
  )

  return $ordered
}

function Get-ZigDocsLastGoodCandidate {
  param(
    [pscustomobject]$State,
    [pscustomobject[]]$Candidates
  )

  if ($State.lastSuccessfulVersion) {
    $match = $Candidates | Where-Object { $_.Key -eq $State.lastSuccessfulVersion -and -not $_.Skip }
    if ($match) { return $match | Select-Object -First 1 }
  }
  return ($Candidates | Where-Object { -not $_.Skip } | Select-Object -First 1)
}

function Update-ZigDocsVersionOutcome {
  param(
    [pscustomobject]$State,
    [pscustomobject]$Candidate,
    [bool]$Success,
    [string]$Mode,
    [string]$Details = ""
  )

  if (-not $Candidate) { return }
  $key = $Candidate.Key
  if (-not $State.versionFailures.ContainsKey($key)) { $State.versionFailures[$key] = 0 }

  if ($Success) {
    $State.versionFailures[$key] = 0
    $State.lastSuccessfulVersion = $key
    $State.lastSuccessfulPath = $Candidate.Path
    $State.lastSuccessfulMode = $Mode
    if ($Candidate.Version) {
      $State.versionMetadata[$key] = @{ path = $Candidate.Path; version = $Candidate.Version }
    }
  } else {
    $State.versionFailures[$key] = ([int]$State.versionFailures[$key]) + 1
  }

  $detailsValue = if ([string]::IsNullOrWhiteSpace($Details)) { $Candidate.Path } else { $Details }
  Add-ZigDocsHistory -State $State -Action "Attempt-$Mode" -Success $Success -Details $detailsValue
}

function Launch-ZigDocsBrowser {
  param(
    [pscustomobject]$State,
    [string]$Url,
    [string]$LogPath
  )

  if (-not $Url) { return }

  $locale = Get-ZigDocsLocale -State $State
  $platform = Get-ZigDocsPlatform
  $candidates = @()

  function Convert-BrowserCommand {
    param([string]$Command)
    if (-not $Command) { return $null }
    $parts = [System.Management.Automation.PSParser]::Tokenize($Command, [ref]$null) | Where-Object { $_.Type -eq 'String' -or $_.Type -eq 'CommandArgument' }
    $values = @()
    foreach ($part in $parts) { $values += $part.Content }
    if ($values.Count -eq 0) { return $null }
    $args = @()
    if ($values.Count -gt 1) { $args = $values[1..($values.Count-1)] }
    return [pscustomobject]@{ Command = $values[0]; Args = $args }
  }

  $preferred = Convert-BrowserCommand -Command $State.preferredBrowser
  if ($preferred) { $candidates += $preferred }

  if ($platform -eq "Windows") {
    $candidates += @(
      @{ Command = "chrome.exe"; Args = @("--incognito", "--lang=$locale") },
      @{ Command = "msedge.exe"; Args = @("--inprivate", "--lang=$locale") },
      @{ Command = "firefox.exe"; Args = @("-private-window", "--lang=$locale") }
    )
  } elseif ($platform -eq "macOS") {
    $candidates += @(
      @{ Command = "open"; Args = @("-a", "Google Chrome", "--args", "--incognito", "--lang=$locale") },
      @{ Command = "open"; Args = @("-a", "Microsoft Edge", "--args", "--inprivate", "--lang=$locale") },
      @{ Command = "open"; Args = @("-a", "Firefox", "--args", "-private-window", "--lang=$locale") }
    )
  } else {
    $candidates += @(
      @{ Command = "google-chrome"; Args = @("--incognito", "--lang=$locale") },
      @{ Command = "chromium"; Args = @("--incognito", "--lang=$locale") },
      @{ Command = "firefox"; Args = @("-private-window", "--lang=$locale") },
      @{ Command = "xdg-open"; Args = @() }
    )
  }

  foreach ($browser in $candidates) {
    if (-not $browser) { continue }
    $command = $browser.Command
    $args = @()
    if ($browser.Args) { $args += $browser.Args }
    $args += $Url

    $canLaunch = $true
    if ($command -ne "open" -and $command -ne "xdg-open" -and $command -ne "zig" -and $command -notlike "*\\" -and $command -notlike "*/*") {
      $cmd = Get-Command $command -ErrorAction SilentlyContinue
      if (-not $cmd) {
        $canLaunch = $false
      } elseif ($cmd.Source) {
        $command = $cmd.Source
      }
    } elseif ((Test-Path $command) -eq $false -and ($command -ne "open") -and ($command -ne "xdg-open")) {
      $canLaunch = $false
    }

    if (-not $canLaunch) { continue }

    try {
      Start-Process -FilePath $command -ArgumentList $args -WindowStyle Normal | Out-Null
      Write-ZigDocsLog -LogPath $LogPath -Message "Browser launch via $command" -Level "INFO"
      return
    } catch {
      Write-ZigDocsLog -LogPath $LogPath -Message "Browser launch failed for $command: $($_.Exception.Message)" -Level "WARN"
    }
  }

  try {
    Start-Process $Url | Out-Null
    Write-ZigDocsLog -LogPath $LogPath -Message "Browser fallback via shell" -Level "INFO"
  } catch {
    Write-ZigDocsLog -LogPath $LogPath -Message "Browser fallback failed: $($_.Exception.Message)" -Level "ERROR"
  }
}

function Get-ZigDocsErrorPatterns {
  return @(
    @{ Regex = 'unable to serve /sources\.tar'; Reason = 'Known Windows regression: sources.tar failure' },
    @{ Regex = 'Access is denied'; Reason = 'Access denied while serving docs' },
    @{ Regex = 'error:.*ListenError'; Reason = 'Port binding failure' }
  )
}

function Detect-ZigDocsKnownError {
  param(
    [string]$Content
  )

  if (-not $Content) { return $null }
  foreach ($pattern in Get-ZigDocsErrorPatterns) {
    if ($Content -match $pattern.Regex) {
      return $pattern.Reason
    }
  }
  return $null
}

function Test-WslMirroredNetworking {
  try {
    $status = wsl.exe --status 2>$null
    if (-not $status) { return $false }
    if ($status -match 'Mirrored networking:\s*Enabled') { return $true }
    if ($status -match 'Default Version:\s*2') { return $true }
  } catch {}
  return $false
}

function Get-WslHostAddress {
  try {
    $addresses = wsl.exe -e sh -lc "hostname -I" 2>$null
    if ($addresses) {
      $first = ($addresses -split '\s+')[0]
      if ($first) { return $first }
    }
  } catch {}
  return "127.0.0.1"
}

function Format-ZigDocsTimestamp {
  param(
    [string]$Timestamp
  )

  if (-not $Timestamp) { return "Never" }
  try {
    $dt = [DateTime]::Parse($Timestamp)
    return $dt.ToLocalTime().ToString("MM/dd/yyyy hh:mm tt", [System.Globalization.CultureInfo]::GetCultureInfo("en-US"))
  } catch {
    return $Timestamp
  }
}

function Show-ZigDocsDashboard {
  param(
    [pscustomobject]$State
  )

  Clear-Host
  Write-Host "Zig Docs Health Dashboard" -ForegroundColor Cyan
  Write-Host "==========================="

  $statusColor = "Yellow"
  if ($State.failureStreak -ge $State.failureThreshold) {
    $statusColor = "Red"
  } elseif ($State.lastStatus -eq "success") {
    $statusColor = "Green"
  }

  $statusLine = "Status: $($State.lastStatus.ToUpper())"
  Write-Host $statusLine -ForegroundColor $statusColor

  Write-Host "Last success: $(Format-ZigDocsTimestamp -Timestamp $State.lastSuccessTimestamp)" -ForegroundColor Green
  if ($State.lastSuccessfulPath) {
    Write-Host "Last good version: $($State.lastSuccessfulPath)" -ForegroundColor Green
  }
  Write-Host "Last mode: $($State.lastSuccessfulMode)" -ForegroundColor Cyan
  Write-Host "Total successes: $($State.totalSuccesses)" -ForegroundColor Green
  Write-Host "Total failures: $($State.totalFailures)" -ForegroundColor Yellow

  if ($State.versionFailures.Count -gt 0) {
    Write-Host "Failure counters (skip after $($State.failureThreshold)):" -ForegroundColor Yellow
    foreach ($entry in $State.versionFailures.GetEnumerator() | Sort-Object -Property Value -Descending) {
      $metadata = $null
      if ($State.versionMetadata.ContainsKey($entry.Key)) {
        $metadata = $State.versionMetadata[$entry.Key]
      }
      $label = if ($metadata -and $metadata.version) { "[$($metadata.version)] $($metadata.path)" } else { $entry.Key }
      $color = if ($entry.Value -ge $State.failureThreshold) { "Red" } else { "Yellow" }
      Write-Host "  $label => $($entry.Value) failures" -ForegroundColor $color
    }
  }

  $browserLabel = if ([string]::IsNullOrWhiteSpace($State.preferredBrowser)) { 'auto' } else { $State.preferredBrowser }
  Write-Host "Preferred browser: $browserLabel" -ForegroundColor Cyan
  Write-Host "Locale: $(Get-ZigDocsLocale -State $State)" -ForegroundColor Cyan

  if ($State.history.Count -gt 0) {
    Write-Host "Recent activity:" -ForegroundColor Cyan
    $State.history | Select-Object -Last 5 | ForEach-Object {
      $flag = if ($_.success) { "✔" } else { "✖" }
      Write-Host "  $flag [$($_.timestamp)] $($_.action) — $($_.details)"
    }
  }

  Write-Host ""
  Write-Host "Options:" -ForegroundColor Cyan
  Write-Host "  [B] Update preferred browser"
  Write-Host "  [T] Adjust failure threshold (current: $($State.failureThreshold))"
  Write-Host "  [Enter] Return to menu"

  $choice = Read-Host "Select"
  switch -Regex ($choice) {
    '^[Bb]$' {
      $browser = Read-Host "Enter browser command (leave blank for auto)"
      if ([string]::IsNullOrWhiteSpace($browser)) {
        $State.preferredBrowser = $null
        Write-Host "Preferred browser cleared." -ForegroundColor Yellow
      } else {
        $State.preferredBrowser = $browser
        Write-Host "Preferred browser set to $browser" -ForegroundColor Green
      }
      Wait-ZigDocsUser
    }
    '^[Tt]$' {
      $value = Read-Host "Enter new failure threshold"
      if ($value -as [int]) {
        $State.failureThreshold = [int]$value
        Write-Host "Failure threshold updated to $value" -ForegroundColor Green
      } else {
        Write-Host "Invalid threshold" -ForegroundColor Red
      }
      Wait-ZigDocsUser
    }
    default {
      # no-op, return
    }
  }
}

function Invoke-ZigDocsEnvironmentCheck {
  param(
    [string]$ZigExe,
    [string]$LogPath
  )

  Write-Host "Running environment checks for $ZigExe" -ForegroundColor Cyan
  Write-ZigDocsLog -LogPath $LogPath -Message "Environment check with '$ZigExe'" -Level "INFO"
  try {
    & $ZigExe version
    & $ZigExe env
    return $true
  } catch {
    Write-Warning "Environment check failed: $($_.Exception.Message)"
    Write-ZigDocsLog -LogPath $LogPath -Message "Environment check failed: $($_.Exception.Message)" -Level "ERROR"
    return $false
  }
}

function Update-PathForZig {
  param(
    [string]$ZigPath,
    [string]$LogPath
  )

  if (-not $ZigPath) { return }
  if (Test-Path $ZigPath) {
    $dir = if ((Get-Item $ZigPath).PSIsContainer) { $ZigPath } else { Split-Path $ZigPath }
    $separator = Get-ZigDocsPathSeparator
    $clean = ($env:PATH -split [regex]::Escape($separator) | Where-Object { $_ -and ($_ -notlike '*zig*') }) -join $separator
    $env:PATH = "$dir$separator" + $clean
    Write-Host "PATH updated for $dir" -ForegroundColor Green
    Write-ZigDocsLog -LogPath $LogPath -Message "PATH updated for '$dir'" -Level "INFO"
  } else {
    Write-Warning "Zig path '$ZigPath' does not exist."
    Write-ZigDocsLog -LogPath $LogPath -Message "Attempted to update PATH with missing '$ZigPath'" -Level "WARN"
  }
}

function Invoke-ZigDocsServerWindows {
  param(
    [string]$ZigExe,
    [pscustomobject]$State,
    [string]$LogPath
  )

  $logFile = Join-Path $State.BaseDir "last-run.out"
  $errFile = Join-Path $State.BaseDir "last-run.err"
  if (Test-Path $logFile) { Remove-Item $logFile -Force }
  if (Test-Path $errFile) { Remove-Item $errFile -Force }

  Write-Host "Launching Zig docs server on Windows..." -ForegroundColor Cyan
  Write-ZigDocsLog -LogPath $LogPath -Message "Starting Windows docs server via '$ZigExe'" -Level "INFO"

  $arguments = @("std", "--no-open-browser")
  $proc = Start-Process -FilePath $ZigExe -ArgumentList $arguments -RedirectStandardOutput $logFile -RedirectStandardError $errFile -PassThru

  $url = $null
  $failureReason = $null
  for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if ($proc.HasExited) {
      break
    }
    if (Test-Path $logFile) {
      $content = Get-Content -LiteralPath $logFile -Raw
      $match = [regex]::Match($content, 'http://127\.0\.0\.1:(?<port>\d+)/')
      if ($match.Success) {
        $url = $match.Value
        $State.lastPort = $match.Groups['port'].Value
        break
      }
      $detected = Detect-ZigDocsKnownError -Content $content
      if ($detected) {
        $failureReason = $detected
        break
      }
    }
    if (Test-Path $errFile) {
      $errContent = Get-Content -LiteralPath $errFile -Raw
      $detected = Detect-ZigDocsKnownError -Content $errContent
      if ($detected) {
        $failureReason = $detected
        break
      }
    }
  }

  if ($url -and -not $failureReason) {
    Write-Host "Docs server running at $url" -ForegroundColor Green
    Write-ZigDocsLog -LogPath $LogPath -Message "Docs server detected at $url (PID $($proc.Id))" -Level "INFO"
    Launch-ZigDocsBrowser -State $State -Url $url -LogPath $LogPath
    return @{ success = $true; pid = $proc.Id; url = $url }
  }

  if (-not $proc.HasExited) {
    try { $proc | Stop-Process -Force } catch {}
  }

  if ($failureReason) {
    Write-Warning "Docs server failed: $failureReason"
    Write-ZigDocsLog -LogPath $LogPath -Message "Docs server failure: $failureReason" -Level "ERROR"
    return @{ success = $false; pid = $proc.Id; url = $null; error = $failureReason }
  }

  Write-Warning "Unable to detect docs server URL. See $errFile for details."
  Write-ZigDocsLog -LogPath $LogPath -Message "Failed to detect docs server URL" -Level "ERROR"
  return @{ success = $false; pid = $proc.Id; url = $null }
}

function Invoke-ZigDocsServerWsl {
  param(
    [pscustomobject]$State,
    [string]$LogPath
  )

  Write-Host "Launching Zig docs server inside WSL..." -ForegroundColor Cyan
  Write-ZigDocsLog -LogPath $LogPath -Message "Starting WSL docs server" -Level "INFO"
  try {
    $output = wsl.exe zig std --no-open-browser 2>&1
    $match = [regex]::Match($output, 'http://(?<host>[\d\.]+):(?<port>\d+)/')
    $host = $null
    $port = $null
    if ($match.Success) {
      $host = if ($match.Groups['host'].Value) { $match.Groups['host'].Value } else { '127.0.0.1' }
      $port = $match.Groups['port'].Value
    } else {
      $match = [regex]::Match($output, 'http://127\.0\.0\.1:(?<port>\d+)/')
      if ($match.Success) {
        $host = '127.0.0.1'
        $port = $match.Groups['port'].Value
      }
    }

    if (-not $port) {
      Write-Warning "Unable to parse docs server URL from WSL output."
      Write-ZigDocsLog -LogPath $LogPath -Message "Could not parse WSL docs server output" -Level "ERROR"
      return $false
    }

    $mirrored = Test-WslMirroredNetworking
    if ($mirrored) {
      $host = '127.0.0.1'
    } elseif (-not $host -or $host -eq '127.0.0.1') {
      $host = Get-WslHostAddress
    }

    $url = "http://$host:$port/"
    $State.lastPort = $port
    Write-Host "WSL docs server running at $url" -ForegroundColor Green
    Write-ZigDocsLog -LogPath $LogPath -Message "WSL docs server ready at $url (mirrored: $mirrored)" -Level "INFO"
    Launch-ZigDocsBrowser -State $State -Url $url -LogPath $LogPath
    return $true
  } catch {
    Write-Warning "WSL docs server failed: $($_.Exception.Message)"
    Write-ZigDocsLog -LogPath $LogPath -Message "WSL docs server error: $($_.Exception.Message)" -Level "ERROR"
    return $false
  }
}

function Invoke-ZigDocsDiagnostics {
  param(
    [pscustomobject]$State,
    [string]$LogPath
  )

  Write-Host "Running diagnostics..." -ForegroundColor Cyan
  if ($State.lastPort) {
    $url = "http://127.0.0.1:$($State.lastPort)/sources.tar"
    Write-Host "Probing $url" -ForegroundColor Yellow
    try {
      $tmp = Join-Path $env:TEMP "zig_sources_$($State.lastPort).tar"
      Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing -TimeoutSec 10
      Write-Host "Download succeeded -> $tmp" -ForegroundColor Green
      Remove-Item $tmp -Force
      Write-ZigDocsLog -LogPath $LogPath -Message "sources.tar download succeeded on port $($State.lastPort)" -Level "INFO"
    } catch {
      Write-Warning "sources.tar probe failed: $($_.Exception.Message)"
      Write-ZigDocsLog -LogPath $LogPath -Message "sources.tar probe failed: $($_.Exception.Message)" -Level "ERROR"
    }
    try {
      $connection = Test-NetConnection -ComputerName 127.0.0.1 -Port $State.lastPort
      Write-Host $connection
      Write-ZigDocsLog -LogPath $LogPath -Message "NetConnection: $($connection.TcpTestSucceeded)" -Level "INFO"
    } catch {
      Write-Warning "Test-NetConnection failed: $($_.Exception.Message)"
      Write-ZigDocsLog -LogPath $LogPath -Message "Test-NetConnection failed: $($_.Exception.Message)" -Level "WARN"
    }
  } else {
    Write-Host "No known port. Run the docs server first." -ForegroundColor Yellow
  }

  Write-Host "Open browser DevTools (F12) and inspect network failures." -ForegroundColor Cyan
  Write-ZigDocsLog -LogPath $LogPath -Message "Diagnostics executed" -Level "INFO"
}

function Invoke-ZigDocsWindowsSession {
  param(
    [pscustomobject]$State,
    [pscustomobject]$Candidate,
    [string]$LogPath
  )

  if (-not $Candidate) {
    Write-Warning "No Zig candidate available."
    return @{ success = $false; message = "No candidate" }
  }

  Update-PathForZig -ZigPath $Candidate.Path -LogPath $LogPath
  $envSuccess = Invoke-ZigDocsEnvironmentCheck -ZigExe $Candidate.Path -LogPath $LogPath
  Add-ZigDocsHistory -State $State -Action "Environment" -Success $envSuccess -Details $Candidate.Path -RecordStats:$false
  if (-not $envSuccess) {
    Update-ZigDocsVersionOutcome -State $State -Candidate $Candidate -Success $false -Mode "Windows" -Details "Environment check failed"
    return @{ success = $false; message = "Environment check failed" }
  }

  $result = Invoke-ZigDocsServerWindows -ZigExe $Candidate.Path -State $State -LogPath $LogPath
  $detail = if ($result.url) { $result.url } elseif ($result.error) { $result.error } else { $null }
  Update-ZigDocsVersionOutcome -State $State -Candidate $Candidate -Success $result.success -Mode "Windows" -Details $detail
  $message = if ($result.success) { $result.url } elseif ($result.error) { $result.error } else { "Launch failed" }
  return @{ success = $result.success; message = $message; candidate = $Candidate; url = $result.url }
}

function Invoke-ZigDocsAutomatedPipeline {
  param(
    [pscustomobject]$State,
    [pscustomobject]$Candidate,
    [string]$LogPath
  )

  Write-Host "Running automated pipeline..." -ForegroundColor Cyan
  $results = @()
  $envSuccess = Invoke-ZigDocsEnvironmentCheck -ZigExe $Candidate.Path -LogPath $LogPath
  $results += @{ stage = "environment"; success = $envSuccess }
  if (-not $envSuccess) {
    Update-ZigDocsVersionOutcome -State $State -Candidate $Candidate -Success $false -Mode "Windows" -Details "Environment check failed"
    return $results
  }
  $server = Invoke-ZigDocsServerWindows -ZigExe $Candidate.Path -State $State -LogPath $LogPath
  $results += @{ stage = "docs-server"; success = $server.success }
  if (-not $server.success) {
    $detail = if ($server.error) { $server.error } else { "Launch failed" }
    Update-ZigDocsVersionOutcome -State $State -Candidate $Candidate -Success $false -Mode "Windows" -Details $detail
    return $results
  }
  Invoke-ZigDocsDiagnostics -State $State -LogPath $LogPath
  $results += @{ stage = "diagnostics"; success = $true }
  Update-ZigDocsVersionOutcome -State $State -Candidate $Candidate -Success $true -Mode "Windows" -Details $server.url
  return $results
}

function Show-ZigDocsSummary {
  param(
    [pscustomobject]$State,
    [pscustomobject[]]$Candidates
  )

  $statusColor = if ($State.lastStatus -eq "success") { "Green" } elseif ($State.failureStreak -ge $State.failureThreshold) { "Red" } else { "Yellow" }
  Write-Host "Status: $($State.lastStatus) (failures in a row: $($State.failureStreak))" -ForegroundColor $statusColor
  if ($State.lastSuccessfulPath) {
    Write-Host "Last good path: $($State.lastSuccessfulPath)" -ForegroundColor Green
  }
  if ($Candidates -and $Candidates.Count -gt 0) {
    $nextCandidate = Get-ZigDocsLastGoodCandidate -State $State -Candidates $Candidates
    if ($nextCandidate) {
      $label = if ($nextCandidate.Version) { $nextCandidate.Version } else { "unknown" }
      Write-Host "Next candidate: $label ($($nextCandidate.Path))" -ForegroundColor Cyan
    }
  }
  if ($State.history.Count -gt 0) {
    Write-Host "Recent activity:" -ForegroundColor Cyan
    $State.history | Select-Object -Last 3 | ForEach-Object {
      $flag = if ($_.success) { "✔" } else { "✖" }
      Write-Host "  $flag [$($_.timestamp)] $($_.action) — $($_.details)"
    }
  }
}

function Zig-DocsInteractive {
  param(
    [string[]]$ZigCandidates = @("zig")
  )

  $storage = Get-ZigDocsStorage
  $state = Load-ZigDocsState -StatePath $storage.StatePath -ZigCandidates $ZigCandidates
  $state | Add-Member -MemberType NoteProperty -Name BaseDir -Value $storage.BaseDir -Force

  while ($true) {
    $candidates = Get-ZigDocsCandidates -State $state -AdditionalCandidates $ZigCandidates
    $sorted = Sort-ZigDocsCandidates -State $state -Candidates $candidates

    Clear-Host
    Write-Host "Zig Docs Interactive" -ForegroundColor Cyan
    Write-Host "====================="
    Write-Host "1. Run docs server with last known good version"
    Write-Host "2. Try all versions automatically"
    Write-Host "3. Choose a version manually"
    Write-Host "4. Run in WSL fallback"
    Write-Host "5. Show health dashboard"
    Write-Host "Q. Exit"
    Show-ZigDocsSummary -State $state -Candidates $sorted

    $choice = Read-Host "Select an option"

    switch ($choice.ToUpperInvariant()) {
      '1' {
        $candidate = Get-ZigDocsLastGoodCandidate -State $state -Candidates $sorted
        if (-not $candidate) {
          Write-Warning "No eligible Zig versions detected."
          Wait-ZigDocsUser
          continue
        }
        $result = Invoke-ZigDocsWindowsSession -State $state -Candidate $candidate -LogPath $storage.LogPath
        if ($result.success) {
          Write-Host $result.message -ForegroundColor Green
        } else {
          Write-Host $result.message -ForegroundColor Yellow
        }
        Wait-ZigDocsUser
      }
      '2' {
        $success = $false
        foreach ($candidate in $sorted) {
          if ($candidate.Skip) { continue }
          Write-Host "Testing $($candidate.Path)" -ForegroundColor Cyan
          $result = Invoke-ZigDocsWindowsSession -State $state -Candidate $candidate -LogPath $storage.LogPath
          if ($result.success) {
            Write-Host "Success with $($candidate.Path)" -ForegroundColor Green
            $success = $true
            break
          } else {
            Write-Host "Failed: $($result.message)" -ForegroundColor Yellow
          }
        }
        if (-not $success) {
          Write-Warning "No Zig version completed successfully."
        }
        Wait-ZigDocsUser
      }
      '3' {
        if (-not $sorted) {
          Write-Warning "No Zig versions available."
          Wait-ZigDocsUser
          continue
        }
        Write-Host "Available versions:" -ForegroundColor Cyan
        for ($i = 0; $i -lt $sorted.Count; $i++) {
          $indicator = if ($sorted[$i].Skip) { "(skipped)" } elseif ($sorted[$i].Key -eq $state.lastSuccessfulVersion) { "(last good)" } else { "" }
          $versionLabel = if ($sorted[$i].Version) { $sorted[$i].Version } else { "unknown" }
          Write-Host "[$($i + 1)] $versionLabel — $($sorted[$i].Path) $indicator"
        }
        Write-Host "[A] Add new location" -ForegroundColor Cyan
        Write-Host "[R] Refresh detection" -ForegroundColor Cyan
        $manual = Read-Host "Select version"
        if ($manual.ToUpperInvariant() -eq 'A') {
          $newPath = Read-Host "Enter full path to Zig folder or executable"
          if (-not [string]::IsNullOrWhiteSpace($newPath)) {
            $state.zigCandidates = @($newPath) + ($state.zigCandidates | Where-Object { $_ -ne $newPath })
            Write-ZigDocsLog -LogPath $storage.LogPath -Message "Added manual Zig candidate $newPath" -Level "INFO"
            Write-Host "Candidate added." -ForegroundColor Green
          }
        } elseif ($manual.ToUpperInvariant() -eq 'R') {
          Write-Host "Candidate list refreshed." -ForegroundColor Cyan
        } elseif ($manual -as [int]) {
          $index = [int]$manual - 1
          if ($index -ge 0 -and $index -lt $sorted.Count) {
            $candidate = $sorted[$index]
            if ($candidate.Skip) {
              Write-Warning "Version skipped after repeated failures. Adjust threshold in dashboard to retry."
            } else {
              $result = Invoke-ZigDocsWindowsSession -State $state -Candidate $candidate -LogPath $storage.LogPath
              if ($result.success) {
                Write-Host $result.message -ForegroundColor Green
              } else {
                Write-Host $result.message -ForegroundColor Yellow
              }
            }
          } else {
            Write-Warning "Invalid selection."
          }
        }
        Wait-ZigDocsUser
        $manualLoop = $true
        while ($manualLoop) {
          $manualCandidates = Sort-ZigDocsCandidates -State $state -Candidates (Get-ZigDocsCandidates -State $state -AdditionalCandidates $ZigCandidates)
          if (-not $manualCandidates -or $manualCandidates.Count -eq 0) {
            Write-Warning "No Zig versions available."
            Wait-ZigDocsUser
            break
          }

          Clear-Host
          Write-Host "Manual Candidate Selection" -ForegroundColor Cyan
          Write-Host "=========================="
          for ($i = 0; $i -lt $manualCandidates.Count; $i++) {
            $candidate = $manualCandidates[$i]
            $indicator = if ($candidate.Skip) { "(skipped)" } elseif ($candidate.Key -eq $state.lastSuccessfulVersion) { "(last good)" } else { "" }
            $versionLabel = if ($candidate.Version) { $candidate.Version } else { "unknown" }
            Write-Host "[$($i + 1)] $versionLabel — $($candidate.Path) $indicator"
          }
          Write-Host "[A] Add new location" -ForegroundColor Cyan
          Write-Host "[D] Delete a location" -ForegroundColor Cyan
          Write-Host "[R] Refresh detection" -ForegroundColor Cyan
          Write-Host "[Enter] Return to main menu" -ForegroundColor Cyan

          $manual = Read-Host "Select version"
          $choiceValue = $manual.ToUpperInvariant()

          if ([string]::IsNullOrWhiteSpace($manual)) {
            $manualLoop = $false
            break
          } elseif ($choiceValue -eq 'A') {
            $newPath = (Read-Host "Enter full path to Zig folder or executable").Trim()
            if (-not [string]::IsNullOrWhiteSpace($newPath)) {
              $state.zigCandidates = @($newPath) + ($state.zigCandidates | Where-Object { $_ -ne $newPath })
              $resolved = Resolve-ZigCandidatePath -Candidate $newPath
              $key = Get-ZigDocsCandidateKey -Path $resolved
              if ($resolved -and $key) {
                $info = Get-ZigVersionInfo -ZigPath $resolved
                if ($info) {
                  $state.versionMetadata[$key] = @{ path = $resolved; version = $info.Raw }
                }
              }
              Write-ZigDocsLog -LogPath $storage.LogPath -Message "Added manual Zig candidate $newPath" -Level "INFO"
              Add-ZigDocsHistory -State $state -Action "Candidate-Add" -Success $true -Details $newPath -RecordStats:$false
              Write-Host "Candidate added." -ForegroundColor Green
              Wait-ZigDocsUser
            }
          } elseif ($choiceValue -eq 'D') {
            $removeInput = Read-Host "Enter the number of the candidate to delete"
            if ($removeInput -as [int]) {
              $removeIndex = [int]$removeInput - 1
              if ($removeIndex -ge 0 -and $removeIndex -lt $manualCandidates.Count) {
                $target = $manualCandidates[$removeIndex]
                Remove-ZigDocsCandidate -State $state -Candidate $target
                Write-ZigDocsLog -LogPath $storage.LogPath -Message "Removed Zig candidate $($target.Path)" -Level "INFO"
                Add-ZigDocsHistory -State $state -Action "Candidate-Remove" -Success $true -Details $target.Path -RecordStats:$false
                Write-Host "Candidate removed." -ForegroundColor Yellow
              } else {
                Write-Warning "Invalid selection for removal."
              }
            } else {
              Write-Warning "Removal requires a numeric selection."
            }
            Wait-ZigDocsUser
          } elseif ($choiceValue -eq 'R') {
            Write-Host "Candidate list refreshed." -ForegroundColor Cyan
            Wait-ZigDocsUser
          } elseif ($manual -as [int]) {
            $index = [int]$manual - 1
            if ($index -ge 0 -and $index -lt $manualCandidates.Count) {
              $candidate = $manualCandidates[$index]
              if ($candidate.Skip) {
                Write-Warning "Version skipped after repeated failures. Adjust threshold in dashboard to retry."
              } else {
                $result = Invoke-ZigDocsWindowsSession -State $state -Candidate $candidate -LogPath $storage.LogPath
                if ($result.success) {
                  Write-Host $result.message -ForegroundColor Green
                } else {
                  Write-Host $result.message -ForegroundColor Yellow
                }
              }
            } else {
              Write-Warning "Invalid selection."
            }
            Wait-ZigDocsUser
          } else {
            Write-Warning "Unrecognized selection."
            Wait-ZigDocsUser
          }
        }
      }
      '4' {
        $success = Invoke-ZigDocsServerWsl -State $state -LogPath $storage.LogPath
        if ($success) {
          $state.lastSuccessfulMode = "WSL"
          Add-ZigDocsHistory -State $state -Action "Docs-WSL" -Success $true -Details "WSL server" -RecordStats:$true
        } else {
          Add-ZigDocsHistory -State $state -Action "Docs-WSL" -Success $false -Details "WSL failure" -RecordStats:$true
        }
        Wait-ZigDocsUser
      }
      '5' {
        Show-ZigDocsDashboard -State $state
      }
      'Q' {
        break
      }
      default {
        Write-Host "Invalid selection" -ForegroundColor Red
        Wait-ZigDocsUser
      }
    }
  }

  Save-ZigDocsState -State $state -StatePath $storage.StatePath
  Write-Host "State saved to $($storage.StatePath)" -ForegroundColor Cyan
}
