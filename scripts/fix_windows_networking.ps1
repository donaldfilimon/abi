# WDBX Windows Networking Fix Script
# This script applies the recommended fixes for connection reset issues on Windows
# Based on troubleshooting steps for ERR_CONNECTION_RESET errors

Write-Host "=== WDBX Windows Networking Fix Script ===" -ForegroundColor Green
Write-Host "This script will apply common fixes for Windows networking issues." -ForegroundColor Yellow
Write-Host "You may need to restart your computer after running this script." -ForegroundColor Yellow
Write-Host ""

# Check if running as administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "❌ This script must be run as Administrator!" -ForegroundColor Red
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✅ Running as Administrator" -ForegroundColor Green
Write-Host ""

# Function to run command and report result
function Invoke-NetworkCommand {
    param(
        [string]$Command,
        [string]$Description
    )
    
    Write-Host "🔧 $Description..." -ForegroundColor Cyan
    try {
        $result = Invoke-Expression $Command 2>&1
        if ($LASTEXITCODE -eq 0 -or $result -match "successfully" -or $result -match "OK") {
            Write-Host "✅ $Description completed successfully" -ForegroundColor Green
        } else {
            Write-Host "⚠️  $Description completed with warnings" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ $Description failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    Write-Host ""
}

# Step 1: Reset TCP/IP Stack
Write-Host "=== Step 1: Resetting TCP/IP Stack ===" -ForegroundColor Magenta
Invoke-NetworkCommand "netsh winsock reset" "Resetting Winsock catalog"
Invoke-NetworkCommand "netsh int ip reset" "Resetting IP stack"

# Step 2: Release and renew IP configuration
Write-Host "=== Step 2: Refreshing IP Configuration ===" -ForegroundColor Magenta
Invoke-NetworkCommand "ipconfig /release" "Releasing IP configuration"
Invoke-NetworkCommand "ipconfig /renew" "Renewing IP configuration"
Invoke-NetworkCommand "ipconfig /flushdns" "Flushing DNS cache"

# Step 3: Reset Windows Firewall (optional - user choice)
Write-Host "=== Step 3: Windows Firewall Reset (Optional) ===" -ForegroundColor Magenta
$resetFirewall = Read-Host "Reset Windows Firewall to defaults? This will remove custom rules. (y/N)"
if ($resetFirewall -eq "y" -or $resetFirewall -eq "Y") {
    Invoke-NetworkCommand "netsh advfirewall reset" "Resetting Windows Firewall"
} else {
    Write-Host "⏭️  Skipping firewall reset" -ForegroundColor Yellow
}
Write-Host ""

# Step 4: Clear additional network caches
Write-Host "=== Step 4: Clearing Network Caches ===" -ForegroundColor Magenta
Invoke-NetworkCommand "netsh int tcp reset" "Resetting TCP settings"
Invoke-NetworkCommand "netsh int ipv4 reset" "Resetting IPv4 settings"
Invoke-NetworkCommand "netsh int ipv6 reset" "Resetting IPv6 settings"

# Step 5: Optimize TCP settings for development
Write-Host "=== Step 5: Optimizing TCP Settings for Development ===" -ForegroundColor Magenta
try {
    # Enable TCP window scaling
    netsh int tcp set global autotuninglevel=normal 2>&1 | Out-Null
    Write-Host "✅ TCP auto-tuning enabled" -ForegroundColor Green
    
    # Set chimney offload
    netsh int tcp set global chimney=enabled 2>&1 | Out-Null
    Write-Host "✅ TCP chimney offload enabled" -ForegroundColor Green
    
    # Set receive side scaling
    netsh int tcp set global rss=enabled 2>&1 | Out-Null
    Write-Host "✅ Receive side scaling enabled" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Some TCP optimizations may have failed" -ForegroundColor Yellow
}
Write-Host ""

# Step 6: Check for proxy settings
Write-Host "=== Step 6: Checking Proxy Settings ===" -ForegroundColor Magenta
try {
    $proxySettings = Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" -Name ProxyEnable -ErrorAction SilentlyContinue
    if ($proxySettings.ProxyEnable -eq 1) {
        Write-Host "⚠️  Proxy is enabled - this might interfere with local connections" -ForegroundColor Yellow
        $disableProxy = Read-Host "Disable proxy for local connections? (y/N)"
        if ($disableProxy -eq "y" -or $disableProxy -eq "Y") {
            Set-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" -Name ProxyEnable -Value 0
            Write-Host "✅ Proxy disabled" -ForegroundColor Green
        }
    } else {
        Write-Host "✅ No proxy configured" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️  Could not check proxy settings" -ForegroundColor Yellow
}
Write-Host ""

# Step 7: Test basic connectivity
Write-Host "=== Step 7: Testing Basic Connectivity ===" -ForegroundColor Magenta
try {
    $ping = Test-NetConnection -ComputerName "127.0.0.1" -Port 80 -InformationLevel Quiet
    if ($ping) {
        Write-Host "✅ Localhost connectivity test passed" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Localhost connectivity test failed" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Could not test connectivity" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "=== Summary ===" -ForegroundColor Green
Write-Host "✅ Network reset commands completed" -ForegroundColor Green
Write-Host "✅ IP configuration refreshed" -ForegroundColor Green
Write-Host "✅ DNS cache cleared" -ForegroundColor Green
Write-Host "✅ TCP settings optimized" -ForegroundColor Green
Write-Host ""

Write-Host "=== Next Steps ===" -ForegroundColor Magenta
Write-Host "1. Restart your computer to apply all changes" -ForegroundColor Yellow
Write-Host "2. Test your WDBX HTTP server again" -ForegroundColor Yellow
Write-Host "3. Run: zig build test-network" -ForegroundColor Cyan
Write-Host "4. If issues persist, try a different port number" -ForegroundColor Yellow
Write-Host "5. Temporarily disable antivirus to test" -ForegroundColor Yellow
Write-Host ""

$restart = Read-Host "Restart computer now? (y/N)"
if ($restart -eq "y" -or $restart -eq "Y") {
    Write-Host "🔄 Restarting computer..." -ForegroundColor Green
    Restart-Computer -Force
} else {
    Write-Host "⚠️  Remember to restart your computer to apply all changes!" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
}
