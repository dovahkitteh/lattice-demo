<#
.SYNOPSIS
  Start Lattice with Ollama AND the new Gothic Daemon Dashboard

.DESCRIPTION
  This script combines the functionality of start_lattice_with_ollama.ps1 with
  automatic startup of the new React-based Daemon Dashboard. It will:
  1. Start Ollama with your local model
  2. Launch the Lattice service on port 8080
  3. Start the Daemon Dashboard on port 3000
  
.PARAMETER ModelDir
  The folder name under text-generation-webui\models that contains your local model. Default: Hermes

.PARAMETER NumCtx
  Context window to configure in Ollama. Default: 8192

.PARAMETER NumGpu
  Number of GPUs for Ollama runtime options. Default: 1

.PARAMETER SkipDashboard
  If specified, only starts Lattice without the dashboard

.EXAMPLE
  .\start_lattice_with_daemon_dashboard.ps1 -ModelDir Hermes
  
.EXAMPLE
  .\start_lattice_with_daemon_dashboard.ps1 -ModelDir Hermes -SkipDashboard
#>

param(
  [string]$ModelDir = "Hermes",
  [int]$NumCtx = 8192,
  [int]$NumGpu = 1,
  [string]$OllamaExe = $env:OLLAMA_BIN,
  [switch]$SkipDashboard
)

try { Set-StrictMode -Version Latest } catch {}
$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = "Lattice + Daemon Dashboard Launcher"

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red }
function Write-Daemon($msg) { Write-Host "[DAEMON] $msg" -ForegroundColor Magenta }

# Resolve repository root (script lives in scripts/)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Daemon "⸸ LAIR OF THE DAEMON - Full System Startup ⸸"
Write-Info "Repository root: $RepoRoot"
Write-Info ""

# Step 1: Start Lattice with Ollama using the existing script
Write-Info "🚀 Starting Lattice with Ollama..."
$LatticeScript = Join-Path $ScriptDir "start_lattice_with_ollama.ps1"

if (-not (Test-Path $LatticeScript)) {
    Write-Err "Could not find start_lattice_with_ollama.ps1 script at: $LatticeScript"
    exit 1
}

Write-Info "Calling: $LatticeScript -ModelDir $ModelDir -NumCtx $NumCtx -NumGpu $NumGpu"

# Start Lattice in background
$LatticeJob = Start-Job -ScriptBlock {
    param($Script, $Model, $Ctx, $Gpu, $Ollama)
    & $Script -ModelDir $Model -NumCtx $Ctx -NumGpu $Gpu -OllamaExe $Ollama
} -ArgumentList $LatticeScript, $ModelDir, $NumCtx, $NumGpu, $OllamaExe

Write-Info "Lattice service starting in background (Job ID: $($LatticeJob.Id))"
Write-Info "Waiting for Lattice to initialize..."

# Wait for Lattice to start (check if port 8080 is responding)
$MaxWaitTime = 60  # seconds
$WaitStart = Get-Date
$LatticeReady = $false

while (((Get-Date) - $WaitStart).TotalSeconds -lt $MaxWaitTime -and -not $LatticeReady) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $LatticeReady = $true
            Write-Info "✅ Lattice service is ready on port 8080"
        }
    }
    catch {
        Start-Sleep -Seconds 2
        Write-Host "." -NoNewline
    }
}
Write-Host ""

if (-not $LatticeReady) {
    Write-Err "❌ Lattice service failed to start within $MaxWaitTime seconds"
    Write-Info "Check the background job for errors:"
    Receive-Job -Job $LatticeJob
    exit 1
}

Write-Info ""

# Step 2: Start the Daemon Dashboard (unless skipped)
if ($SkipDashboard) {
    Write-Info "Skipping dashboard startup as requested"
    Write-Info "Lattice is running on: http://localhost:8080"
    Write-Info "You can access the old dashboard at: http://localhost:8080/dashboard"
} else {
    Write-Daemon "🩸 Starting Gothic Daemon Dashboard..."
    
    $DashboardDir = Join-Path $RepoRoot "daemon-dashboard"
    $DashboardScript = Join-Path $DashboardDir "start-daemon-dashboard.ps1"
    
    if (-not (Test-Path $DashboardDir)) {
        Write-Err "Daemon dashboard directory not found at: $DashboardDir"
        Write-Info "Continuing with Lattice only..."
    }
    elseif (-not (Test-Path $DashboardScript)) {
        Write-Err "Dashboard startup script not found at: $DashboardScript"
        Write-Info "Continuing with Lattice only..."
    }
    else {
        # Start dashboard in a new window
        Write-Daemon "Launching dashboard in new window..."
        Start-Process -WindowStyle Normal -FilePath "powershell" -ArgumentList @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", 
            "-File", $DashboardScript
        )
        
        Write-Info ""
        Write-Daemon "⸸ SYSTEM STARTUP COMPLETE ⸸"
        Write-Info ""
        Write-Info "🔗 Services Running:"
        Write-Info "  • Lattice Backend:    http://localhost:8080"
        Write-Info "  • Daemon Dashboard:   http://localhost:3000"
        Write-Info "  • Old Dashboard:      http://localhost:8080/dashboard"
        Write-Info ""
        Write-Daemon "The new Gothic Dashboard should open automatically."
        Write-Daemon "If not, navigate to: http://localhost:3000"
        Write-Info ""
    }
}

# Keep the script running and monitor the Lattice job
Write-Info "Monitoring Lattice service... (Press Ctrl+C to stop)"

try {
    while ($true) {
        if ($LatticeJob.State -eq "Completed") {
            Write-Info "Lattice job completed. Checking for errors..."
            $JobOutput = Receive-Job -Job $LatticeJob
            Write-Info $JobOutput
            break
        }
        elseif ($LatticeJob.State -eq "Failed") {
            Write-Err "Lattice job failed!"
            $JobErrors = Receive-Job -Job $LatticeJob
            Write-Err $JobErrors
            break
        }
        
        Start-Sleep -Seconds 5
    }
}
catch {
    Write-Info "Shutting down..."
}
finally {
    # Clean up background job
    if ($LatticeJob) {
        Remove-Job -Job $LatticeJob -Force -ErrorAction SilentlyContinue
    }
    Write-Info "Cleanup complete."
}
