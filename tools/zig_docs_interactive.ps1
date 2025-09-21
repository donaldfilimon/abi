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
    lastSuccess      = $null
    failureStreak    = 0
    history          = @()
    preferredBrowser = $null
    lastPort         = $null
    zigCandidates    = $ZigCandidates
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
        if (-not $state.zigCandidates) {
          $state.zigCandidates = $ZigCandidates
        }
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

function Add-ZigDocsHistory {
  param(
    [pscustomobject]$State,
    [string]$Action,
    [bool]$Success,
    [string]$Details = ""
  )

  $record = [ordered]@{
    timestamp = Get-Date -Format "o"
    action    = $Action
    success   = $Success
    details   = $Details
  }
  $State.history = @($State.history + ([pscustomobject]$record) | Select-Object -Last 50)

  if ($Success) {
    $State.failureStreak = 0
    $State.lastSuccess = $Action
  } else {
    $State.failureStreak++
  }
}

function Resolve-ZigExecutable {
  param(
    [string[]]$Candidates
  )

  foreach ($candidate in $Candidates) {
    if (-not $candidate) { continue }
    if ($candidate -eq "zig") {
      return $candidate
    }
    if (Test-Path $candidate) {
      if ((Get-Item $candidate).PSIsContainer) {
        $exe = Join-Path $candidate "zig.exe"
        if (Test-Path $exe) { return $exe }
      } elseif ($candidate.ToLower().EndsWith("zig.exe")) {
        return $candidate
      }
    }
  }
  return "zig"
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

function Invoke-ZigDocsVersionSwitch {
  param(
    [pscustomobject]$State,
    [string]$LogPath
  )

  $options = @($State.zigCandidates | Where-Object { $_ })
  if (-not $options) {
    $options = @("zig")
  }
  Write-Host "Available Zig locations:" -ForegroundColor Cyan
  for ($i = 0; $i -lt $options.Count; $i++) {
    Write-Host "[$($i + 1)] $($options[$i])"
  }
  Write-Host "[A] Add new location"
  $choice = Read-Host "Select option"

  if ($choice -match '^[Aa]$') {
    $newPath = Read-Host "Enter full path to Zig folder or zig.exe"
    if ($newPath) {
      $State.zigCandidates = @($newPath) + $State.zigCandidates
      Write-ZigDocsLog -LogPath $LogPath -Message "Added Zig candidate '$newPath'" -Level "INFO"
      return Resolve-ZigExecutable -Candidates @($newPath)
    }
    return $null
  }

  if ($choice -as [int]) {
    $index = [int]$choice - 1
    if ($index -ge 0 -and $index -lt $options.Count) {
      $selected = $options[$index]
      Write-ZigDocsLog -LogPath $LogPath -Message "Selected Zig candidate '$selected'" -Level "INFO"
      return Resolve-ZigExecutable -Candidates @($selected)
    }
  }

  Write-Warning "No Zig path selected."
  return $null
}

function Update-PathForZig {
  param(
    [string]$ZigPath,
    [string]$LogPath
  )

  if (-not $ZigPath) { return }
  if (Test-Path $ZigPath) {
    $dir = if ((Get-Item $ZigPath).PSIsContainer) { $ZigPath } else { Split-Path $ZigPath }
    $clean = ($env:PATH -split ';' | Where-Object { $_ -and ($_ -notlike '*\\zig\\*') }) -join ';'
    $env:PATH = "$dir;" + $clean
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

  $proc = Start-Process -FilePath $ZigExe -ArgumentList "std","--no-open-browser" -RedirectStandardOutput $logFile -RedirectStandardError $errFile -PassThru

  $url = $null
  for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Path $logFile) {
      $content = Get-Content -LiteralPath $logFile -Raw
      $match = [regex]::Match($content, 'http://127\.0\.0\.1:(?<port>\d+)/')
      if ($match.Success) {
        $url = $match.Value
        $State.lastPort = $match.Groups['port'].Value
        break
      }
    }
  }

  if ($url) {
    Write-Host "Docs server running at $url" -ForegroundColor Green
    Write-ZigDocsLog -LogPath $LogPath -Message "Docs server detected at $url (PID $($proc.Id))" -Level "INFO"
    $browser = $State.preferredBrowser
    if ($browser) {
      Start-Process $browser $url
    } else {
      Start-Process $url
    }
    return @{ success = $true; pid = $proc.Id; url = $url }
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
    $output = wsl zig std --no-open-browser 2>&1
    $match = [regex]::Match($output, 'http://127\.0\.0\.1:(?<port>\d+)/')
    if (-not $match.Success) {
      $match = [regex]::Match($output, 'http://(?<host>[\d\.]+):(?<port>\d+)/')
    }
    if ($match.Success) {
      $url = $match.Value
      $State.lastPort = $match.Groups['port'].Value
      Write-Host "WSL docs server running at $url" -ForegroundColor Green
      Write-ZigDocsLog -LogPath $LogPath -Message "WSL docs server ready at $url" -Level "INFO"
      Start-Process $url
      return $true
    }
    Write-Warning "Unable to parse docs server URL from WSL output."
    Write-ZigDocsLog -LogPath $LogPath -Message "Could not parse WSL docs server output" -Level "ERROR"
    return $false
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

function Invoke-ZigDocsAutomatedPipeline {
  param(
    [pscustomobject]$State,
    [string]$ZigExe,
    [string]$LogPath
  )

  Write-Host "Running automated pipeline..." -ForegroundColor Cyan
  $results = @()
  $results += @{ stage = "environment"; success = Invoke-ZigDocsEnvironmentCheck -ZigExe $ZigExe -LogPath $LogPath }
  if (-not $results[-1].success) {
    return $results
  }
  $server = Invoke-ZigDocsServerWindows -ZigExe $ZigExe -State $State -LogPath $LogPath
  $results += @{ stage = "docs-server"; success = $server.success }
  if (-not $server.success) {
    return $results
  }
  Invoke-ZigDocsDiagnostics -State $State -LogPath $LogPath
  $results += @{ stage = "diagnostics"; success = $true }
  return $results
}

function Show-ZigDocsSummary {
  param(
    [pscustomobject]$State
  )

  if ($State.history.Count -gt 0) {
    Write-Host "Recent activity:" -ForegroundColor Cyan
    $State.history | Select-Object -Last 5 | ForEach-Object {
      $flag = if ($_.success) { "✔" } else { "✖" }
      Write-Host "$flag [$($_.timestamp)] $($_.action) — $($_.details)"
    }
  }
  if ($State.failureStreak -ge 2) {
    Write-Host "Multiple failures detected. Consider using option 4 (WSL fallback)." -ForegroundColor Yellow
  } elseif ($State.lastSuccess) {
    Write-Host "Last success: $($State.lastSuccess)" -ForegroundColor Green
  }
}

function Zig-DocsInteractive {
  param(
    [string[]]$ZigCandidates = @("zig")
  )

  $storage = Get-ZigDocsStorage
  $state = Load-ZigDocsState -StatePath $storage.StatePath -ZigCandidates $ZigCandidates
  $state | Add-Member -MemberType NoteProperty -Name BaseDir -Value $storage.BaseDir -Force

  $zigExe = Resolve-ZigExecutable -Candidates $state.zigCandidates

  while ($true) {
    Clear-Host
    Write-Host "Zig Docs Interactive Menu" -ForegroundColor Cyan
    Write-Host "=========================="
    Write-Host "1. Check Environment"
    Write-Host "2. Switch Zig Version"
    Write-Host "3. Run Docs Server (Windows)"
    Write-Host "4. Run Docs Server (WSL)"
    Write-Host "5. Diagnostics & Network Tests"
    Write-Host "6. Full Automated Pipeline"
    Write-Host "7. Set Preferred Browser"
    Write-Host "8. Exit"
    Show-ZigDocsSummary -State $state
    $choice = Read-Host "Select an option"

    switch ($choice) {
      '1' {
        $success = Invoke-ZigDocsEnvironmentCheck -ZigExe $zigExe -LogPath $storage.LogPath
        Add-ZigDocsHistory -State $state -Action "Environment" -Success $success -Details $zigExe
        Pause
      }
      '2' {
        $newExe = Invoke-ZigDocsVersionSwitch -State $state -LogPath $storage.LogPath
        if ($newExe) {
          Update-PathForZig -ZigPath $newExe -LogPath $storage.LogPath
          $zigExe = Resolve-ZigExecutable -Candidates @($newExe)
          Add-ZigDocsHistory -State $state -Action "Switch" -Success $true -Details $zigExe
        }
        Pause
      }
      '3' {
        Update-PathForZig -ZigPath $zigExe -LogPath $storage.LogPath
        $result = Invoke-ZigDocsServerWindows -ZigExe $zigExe -State $state -LogPath $storage.LogPath
        Add-ZigDocsHistory -State $state -Action "Docs-Windows" -Success $result.success -Details $result.url
        Pause
      }
      '4' {
        $success = Invoke-ZigDocsServerWsl -State $state -LogPath $storage.LogPath
        Add-ZigDocsHistory -State $state -Action "Docs-WSL" -Success $success
        Pause
      }
      '5' {
        Invoke-ZigDocsDiagnostics -State $state -LogPath $storage.LogPath
        Add-ZigDocsHistory -State $state -Action "Diagnostics" -Success $true
        Pause
      }
      '6' {
        Update-PathForZig -ZigPath $zigExe -LogPath $storage.LogPath
        $results = Invoke-ZigDocsAutomatedPipeline -State $state -ZigExe $zigExe -LogPath $storage.LogPath
        $success = $results | Where-Object { -not $_.success } | Measure-Object | Select-Object -ExpandProperty Count
        Add-ZigDocsHistory -State $state -Action "Pipeline" -Success ($success -eq 0)
        Pause
      }
      '7' {
        $browser = Read-Host "Enter browser command (e.g. chrome.exe, msedge.exe) or leave blank to reset"
        if ($browser) {
          $state.preferredBrowser = $browser
          Write-Host "Preferred browser set to $browser" -ForegroundColor Green
          Write-ZigDocsLog -LogPath $storage.LogPath -Message "Preferred browser set to $browser" -Level "INFO"
        } else {
          $state.preferredBrowser = $null
          Write-Host "Preferred browser cleared." -ForegroundColor Yellow
          Write-ZigDocsLog -LogPath $storage.LogPath -Message "Preferred browser cleared" -Level "INFO"
        }
        Pause
      }
      '8' {
        break
      }
      Default {
        Write-Host "Invalid selection" -ForegroundColor Red
        Pause
      }
    }
  }

  Save-ZigDocsState -State $state -StatePath $storage.StatePath
  Write-Host "State saved to $($storage.StatePath)" -ForegroundColor Cyan
}
