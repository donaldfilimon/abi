#Requires -Version 7.0
<#
.SYNOPSIS
  Provisions an Azure Application Insights component for a web app.
.DESCRIPTION
  Creates an Application Insights resource in the specified resource group.
  Requires Azure CLI and contributor access to the target resource group.
.PARAMETER ResourceGroup
  Name of the resource group (required).
.PARAMETER AppName
  Name of the application (required).
.PARAMETER Location
  Azure region (default: eastus).
.EXAMPLE
  ./appinsights.ps1 -ResourceGroup "my-rg" -AppName "my-app" -Location "eastus"
#>

param(
  [Parameter(Mandatory)]
  [string]$ResourceGroup,

  [Parameter(Mandatory)]
  [string]$AppName,

  [string]$Location = "eastus"
)

$ErrorActionPreference = "Stop"

# Validate Azure CLI availability
if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
  Write-Error "Azure CLI (az) is not installed. Install from https://docs.microsoft.com/cli/azure/install-azure-cli"
}

# Validate not empty
if ([string]::IsNullOrWhiteSpace($ResourceGroup)) {
  Write-Error "Parameter -ResourceGroup is required"
}
if ([string]::IsNullOrWhiteSpace($AppName)) {
  Write-Error "Parameter -AppName is required"
}

$existing = az monitor app-insights component show --app $AppName --resource-group $ResourceGroup 2>$null
if ($existing) {
  Write-Host "App Insights '$AppName' already exists in resource group '$ResourceGroup'"
  exit 0
}

Write-Host "Creating App Insights component '$AppName' in '$ResourceGroup' ($Location)..."
$result = az monitor app-insights component create `
  --app $AppName `
  --location $Location `
  --resource-group $ResourceGroup `
  --kind Web `
  --output json

if ($LASTEXITCODE -ne 0) {
  Write-Error "Failed to create App Insights component"
}

$instrumentationKey = ($result | ConvertFrom-Json).InstrumentationKey
Write-Host "App Insights created. InstrumentationKey: $($instrumentationKey.Substring(0, 8))..."