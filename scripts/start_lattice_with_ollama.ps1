<#
.SYNOPSIS
  Start Ollama with a local GGUF model (from text-generation-webui models), warm it with 8K ctx + GPU options, then launch Lattice.

.PARAMETER ModelDir
  The folder name under text-generation-webui\models that contains your local model (with a .gguf file). Default: Hermes

.PARAMETER NumCtx
  Context window to configure in Ollama (PARAMETER num_ctx). Default: 8192

.PARAMETER NumGpu
  Number of GPUs for Ollama runtime options (options.num_gpu). Default: 1

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/start_lattice_with_ollama.ps1 -ModelDir Hermes -NumCtx 8192 -NumGpu 1

Notes:
  - Requires Ollama installed and available on PATH.
  - If the Windows service 'Ollama' is present, this script starts it; otherwise it runs 'ollama serve' in background.
  - This script creates a temporary Modelfile pointing to your local GGUF and registers an Ollama model tag.
#>

param(
  [string]$ModelDir = "Hermes",
  [int]$NumCtx = 8192,
  [int]$NumGpu = 1,
  [string]$OllamaExe = $env:OLLAMA_BIN
)

try { Set-StrictMode -Version Latest } catch {}
$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = "Start Lattice with Ollama"

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Resolve repository root (script lives in scripts/)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Info "Repository root: $RepoRoot"

# Helper: resolve ollama.exe
function Get-OllamaExePath {
  param([string]$Preferred)
  if ($Preferred -and (Test-Path $Preferred)) { 
    Write-Info "Using preferred Ollama path: $Preferred"
    return (Resolve-Path $Preferred).Path 
  }
  
  # Try PATH first
  try {
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd -and (Test-Path $cmd.Source)) { 
      Write-Info "Found Ollama in PATH: $($cmd.Source)"
      return $cmd.Source 
    }
  } catch {}
  
  # Check standard installation locations
  $candidates = @(
    "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",  # Most common for user installs
    "$env:ProgramFiles\Ollama\ollama.exe",
    "$env:ProgramFiles(x86)\Ollama\ollama.exe"
  )
  
  Write-Info "Searching for Ollama in standard locations..."
  foreach ($c in $candidates) { 
    Write-Info "  Checking: $c"
    if (Test-Path $c) { 
      Write-Info "  Found Ollama at: $c"
      return $c 
    }
  }
  
  Write-Err "Ollama not found in any standard location. Checked:"
  foreach ($c in $candidates) { Write-Err "  - $c" }
  return $null
}

# 1) Ensure Ollama daemon is running (service preferred)
function Start-Ollama {
  $global:ResolvedOllama = Get-OllamaExePath -Preferred $OllamaExe
  try {
    $svc = Get-Service -Name "Ollama" -ErrorAction SilentlyContinue
    if ($null -ne $svc) {
      if ($svc.Status -ne 'Running') {
        Write-Info "Starting Ollama Windows service..."
        Start-Service -Name "Ollama"
        Start-Sleep -Seconds 2
      }
      Write-Info "Ollama service is running."
    } else {
      # Fallback: run ollama serve in background via resolved path
      if (-not $global:ResolvedOllama) {
        Write-Err "Could not find ollama.exe. Set OLLAMA_BIN env or pass -OllamaExe 'C:\\Path\\to\\ollama.exe'"
        throw "ollama.exe not found"
      }
      Write-Info "Starting Ollama via: $global:ResolvedOllama serve"
      Start-Process -WindowStyle Minimized -FilePath $global:ResolvedOllama -ArgumentList "serve" | Out-Null
      Start-Sleep -Seconds 2
    }
  } catch {
    Write-Warn "Could not start Ollama via service; attempting 'ollama serve'... $_"
    if (-not $global:ResolvedOllama) { Write-Err "ollama.exe not found"; throw }
    Start-Process -WindowStyle Minimized -FilePath $global:ResolvedOllama -ArgumentList "serve" | Out-Null
    Start-Sleep -Seconds 2
  }
}

function Test-OllamaReady {
  try {
    $resp = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 5
    return $resp.StatusCode -eq 200
  } catch { return $false }
}

Write-Info "Ensuring Ollama is running on 11434..."
Start-Ollama

for ($i=0; $i -lt 15; $i++) {
  if (Test-OllamaReady) { break }
  Start-Sleep -Milliseconds 400
}
if (-not (Test-OllamaReady)) { Write-Err "Ollama API not responding at 11434"; exit 1 }

# 2) Locate local GGUF under text-generation-webui/models
$ModelsRoot = Join-Path $RepoRoot "text-generation-webui\models"
if (-not (Test-Path $ModelsRoot)) { Write-Err "Models directory not found: $ModelsRoot"; exit 1 }

# First, try to find the model as a subdirectory (original behavior)
$TargetDir = Join-Path $ModelsRoot $ModelDir
$gguf = $null

if (Test-Path $TargetDir) {
  Write-Info "Looking for GGUF files in model directory: $TargetDir"
  $gguf = Get-ChildItem -Path $TargetDir -Recurse -Filter *.gguf | Select-Object -First 1
}

# If not found in subdirectory, try to find the model file directly in models root
if ($null -eq $gguf) {
  Write-Info "Model directory not found, searching for GGUF files directly in: $ModelsRoot"
  # Try exact match first
  $exactMatch = Get-ChildItem -Path $ModelsRoot -Filter "$ModelDir*.gguf" | Select-Object -First 1
  if ($exactMatch) {
    $gguf = $exactMatch
    Write-Info "Found exact match: $($gguf.Name)"
  } else {
    # Try partial match (case-insensitive)
    $partialMatch = Get-ChildItem -Path $ModelsRoot -Filter "*.gguf" | Where-Object { $_.Name -like "*$ModelDir*" } | Select-Object -First 1
    if ($partialMatch) {
      $gguf = $partialMatch
      Write-Info "Found partial match: $($gguf.Name)"
    } else {
      # Show available models and exit
      Write-Err "No GGUF file found matching '$ModelDir'"
      $availableModels = Get-ChildItem -Path $ModelsRoot -Filter "*.gguf"
      if ($availableModels) {
        Write-Info "Available GGUF models:"
        foreach ($model in $availableModels) {
          Write-Info "  - $($model.Name)"
        }
        Write-Info "Try running with -ModelDir matching one of these model names (without .gguf extension)"
      }
      exit 1
    }
  }
}

if ($null -eq $gguf) { Write-Err "No .gguf file found"; exit 1 }
Write-Info "Using GGUF: $($gguf.FullName)"

# 3) Create a temporary Modelfile that points to local GGUF and sets num_ctx
$TempDir = Join-Path $env:TEMP "lattice_ollama"
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
$ModelTag = ("{0}-local-8k" -f $ModelDir).ToLower()
$ModelfilePath = Join-Path $TempDir "Modelfile"

$modelfile = @()
$modelfile += "FROM `"$($gguf.FullName)`""
$modelfile += "PARAMETER num_ctx $NumCtx"
# You can add more defaults if desired, e.g., temperature; leave runtime options for per-request control
Set-Content -Path $ModelfilePath -Value ($modelfile -join "`n") -Encoding UTF8
Write-Info "Generated Modelfile at: $ModelfilePath"

# 4) Register Ollama model tag (idempotent)
function Test-OllamaModelExists($name) {
  try {
    $resp = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
    return ($resp.models | Where-Object { $_.name -eq $name }) -ne $null
  } catch { return $false }
}

if (-not (Test-OllamaModelExists $ModelTag)) {
  Write-Info "Creating Ollama model: $ModelTag"
  if (-not $global:ResolvedOllama) { $global:ResolvedOllama = Get-OllamaExePath -Preferred $OllamaExe }
  $p = Start-Process -FilePath $global:ResolvedOllama -ArgumentList @("create", $ModelTag, "-f", $ModelfilePath) -NoNewWindow -PassThru -Wait
  if ($p.ExitCode -ne 0) { Write-Err "ollama create failed with exit code $($p.ExitCode)"; exit 1 }
} else {
  Write-Info "Ollama model already exists: $ModelTag"
}

# 5) Warm the model (non-streaming chat) and verify
Write-Info "Warming model ($ModelTag) with num_ctx=$NumCtx ..."
$payload = @{ model = $ModelTag; messages = @(@{ role = "user"; content = "Hello" }); stream = $false; options = @{ num_predict = 8; num_ctx = $NumCtx; num_gpu = $NumGpu } } | ConvertTo-Json -Depth 6
try {
  $resp = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:11434/api/chat" -Body $payload -ContentType 'application/json' -TimeoutSec 60
  if (-not $resp.done) { Write-Warn "Model warmup did not return done=true; continuing..." }
} catch { Write-Warn "Warmup call failed: $_" }

# 6) Export environment for Lattice and launch service
$env:LLM_BACKEND = 'ollama'
$env:LLM_API     = 'http://127.0.0.1:11434'
$env:OLLAMA_MODEL   = "${ModelTag}:latest"  # Add :latest suffix to match Ollama's model naming
$env:OLLAMA_NUM_CTX = "$NumCtx"
$env:OLLAMA_NUM_GPU = "$NumGpu"

Write-Info "Environment variables set:"
Write-Info "  LLM_BACKEND: $env:LLM_BACKEND"
Write-Info "  LLM_API: $env:LLM_API"
Write-Info "  OLLAMA_MODEL: $env:OLLAMA_MODEL"

# Set Lattice to run on a different port to avoid conflict with Ollama
$env:LATTICE_PORT = '8080'

Write-Info "Starting Lattice Service with Ollama model '${ModelTag}:latest'..."
Write-Info "Ollama API running on: http://127.0.0.1:11434"
Write-Info "Lattice Service will run on: http://127.0.0.1:8080"
Write-Info "Model file: $($gguf.FullName)"

# Start Lattice in background to test LLM endpoint connectivity
Write-Info "Starting Lattice Service in background for connectivity testing..."
$latticeProcess = Start-Process -FilePath "python" -ArgumentList "lattice_service.py" -NoNewWindow -PassThru

# Wait for Lattice to start up
Write-Info "Waiting for Lattice Service to initialize..."
Start-Sleep -Seconds 8

# Test function for Lattice health endpoint
function Test-LatticeHealth {
  try {
    $resp = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8080/health" -TimeoutSec 10
    return $resp.StatusCode -eq 200
  } catch { return $false }
}

# Test function for Ollama endpoint connectivity (what Lattice actually uses)
function Test-OllamaEndpoint {
  try {
    $testPayload = @{
      model = "${ModelTag}:latest"
      messages = @(@{ role = "user"; content = "Connection test" })
      stream = $false
      options = @{ num_predict = 10 }
    } | ConvertTo-Json -Depth 4
    
    $resp = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:11434/api/chat" -Method Post -Body $testPayload -ContentType 'application/json' -TimeoutSec 15
    return $resp.StatusCode -eq 200
  } catch { 
    Write-Warn "Ollama endpoint test failed: $_"
    return $false 
  }
}

# Wait for Lattice health endpoint to be available
$healthReady = $false
Write-Info "Waiting for Lattice health endpoint..."
for ($i = 0; $i -lt 30; $i++) {
  if (Test-LatticeHealth) {
    $healthReady = $true
    Write-Info "Lattice health endpoint is responding"
    break
  }
  Write-Host "." -NoNewline
  Start-Sleep -Seconds 2
}
Write-Host ""

if (-not $healthReady) {
  Write-Err "Lattice health endpoint not responding after 60 seconds"
  if ($latticeProcess -and -not $latticeProcess.HasExited) {
    Write-Info "Terminating Lattice process..."
    $latticeProcess.Kill()
  }
  exit 1
}

# Test Ollama endpoint connectivity (what Lattice actually uses)
Write-Info "Testing Ollama endpoint connectivity..."
$ollamaReady = $false
for ($i = 0; $i -lt 10; $i++) {
  if (Test-OllamaEndpoint) {
    $ollamaReady = $true
    Write-Info "Ollama endpoint connectivity test passed"
    break
  }
  Write-Info "Testing Ollama endpoint... ($($i+1)/10)"
  Start-Sleep -Seconds 3
}

if (-not $ollamaReady) {
  Write-Err "Ollama endpoint connectivity test failed"
  Write-Info "This indicates configuration issues between Lattice and Ollama"
  Write-Info "Diagnostics:"
  Write-Info "  - Ollama API: $env:LLM_API"
  Write-Info "  - Expected model: $env:OLLAMA_MODEL"
  Write-Info "  - Endpoint tested: http://127.0.0.1:11434/api/chat"
  
  # Show available models
  try {
    $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
    if ($tags.models) {
      Write-Info "Available Ollama models:"
      foreach ($model in $tags.models) {
        Write-Info "    - $($model.name)"
      }
    }
  } catch {
    Write-Warn "Could not retrieve available models from Ollama"
  }
  
  if ($latticeProcess -and -not $latticeProcess.HasExited) {
    Write-Info "Terminating Lattice process..."
    $latticeProcess.Kill()
  }
  exit 1
}

Write-Info "All connectivity tests passed! Lattice is ready."
Write-Info "Health check: http://127.0.0.1:8080/health"
Write-Info "Chat endpoint: http://127.0.0.1:8080/v1/chat/completions"
Write-Info "Daemon introspection: http://127.0.0.1:8080/v1/daemon/status"

# Keep the process running in foreground
if ($latticeProcess -and -not $latticeProcess.HasExited) {
  Write-Info "Lattice Service is running. Press Ctrl+C to stop."
  try {
    $latticeProcess.WaitForExit()
  } catch {
    Write-Info "Shutting down Lattice Service..."
  }
} else {
  Write-Err "Lattice process has unexpectedly exited"
  exit 1
}

