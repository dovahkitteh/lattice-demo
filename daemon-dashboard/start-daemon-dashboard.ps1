<#
.SYNOPSIS
  Start the Daemon Dashboard React app

.DESCRIPTION
  Starts the React development server for the Gothic Daemon Dashboard on port 3000
  The dashboard will proxy API calls to the Lattice service running on port 8080

.PARAMETER Production
  If specified, serves the production build instead of starting dev server

.EXAMPLE
  .\start-daemon-dashboard.ps1
  
.EXAMPLE
  .\start-daemon-dashboard.ps1 -Production
#>

param(
  [switch]$Production
)

$ErrorActionPreference = 'Stop'

# Get the script directory (daemon-dashboard folder)
$DashboardDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $DashboardDir

Write-Host "🩸 Starting LAIR OF THE DAEMON Dashboard..." -ForegroundColor Magenta

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
  Write-Host "Installing dependencies..." -ForegroundColor Yellow
  npm install
}

if ($Production) {
  # Build and serve production version
  Write-Host "Building production version..." -ForegroundColor Green
  npm run build
  
  Write-Host "Starting production server..." -ForegroundColor Green
  npm run preview
} else {
  # Start development server
  Write-Host "Starting development server..." -ForegroundColor Green
  Write-Host "Dashboard will be available at: http://localhost:3000" -ForegroundColor Cyan
  Write-Host "API calls will be proxied to: http://localhost:8080" -ForegroundColor Cyan
  Write-Host "" 
  Write-Host "Make sure the Lattice service is running on port 8080 first!" -ForegroundColor Yellow
  Write-Host ""
  
  # Start dev server and open browser after delay
  Write-Host "Starting development server..." -ForegroundColor Green
  
  # Start browser after a delay (in background job)
  Start-Job -ScriptBlock {
    Start-Sleep -Seconds 5
    Start-Process "http://localhost:3000"
  } | Out-Null
  
  # Run npm dev directly (this will block and keep the script running)
  npm run dev
}
