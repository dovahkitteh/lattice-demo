<#
.SYNOPSIS
  Smart Lattice launcher that detects configuration and starts appropriately.

.DESCRIPTION
  This script automatically detects whether external APIs (Anthropic/OpenAI) are configured
  or if local Ollama models should be used, then starts Lattice accordingly.
  Priority: Anthropic > OpenAI > Local Ollama

.PARAMETER ModelDir
  The folder name under text-generation-webui\models for local Ollama models. Default: Hermes

.PARAMETER NumCtx
  Context window for Ollama. Default: 8192

.PARAMETER NumGpu
  Number of GPUs for Ollama. Default: 1

.PARAMETER ForceLocal
  Force local Ollama mode even if external API is configured

.PARAMETER ForceExternal
  Force external API mode even if API key not found

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/start_lattice_smart.ps1
  
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/start_lattice_smart.ps1 -ForceLocal

Notes:
  - Priority: ANTHROPIC_API_KEY > OPENAI_API_KEY > Local Ollama
  - Automatically detects API keys in environment or .env file
  - Falls back through priority chain if APIs fail
  - Handles both local and external API configurations gracefully
#>

param(
  [string]$ModelDir = "Hermes",
  [int]$NumCtx = 8192,
  [int]$NumGpu = 1,
  [string]$OllamaExe = $env:OLLAMA_BIN,
  [switch]$ForceLocal,
  [switch]$ForceExternal
)

try { Set-StrictMode -Version Latest } catch {}
$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = "Smart Lattice Launcher"

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red }
function Write-Success($msg) { Write-Host "[SUCCESS] $msg" -ForegroundColor Green }

# Resolve repository root (script lives in scripts/)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Info "Repository root: $RepoRoot"
Write-Info "Smart Lattice Launcher - Detecting configuration..."

# Function to read .env file and check for external API configuration
# Priority: Anthropic > OpenAI > Local
function Get-ExternalApiConfig {
  $envFile = Join-Path $RepoRoot ".env"
  $config = @{
    HasApiKey = $false
    Provider = $null  # "anthropic" or "openai"
    BaseUrl = $null
    Model = $null
    ApiKey = $null
  }
  
  # Check environment variables first - ANTHROPIC has priority
  if ($env:ANTHROPIC_API_KEY) {
    $config.HasApiKey = $true
    $config.Provider = "anthropic"
    $config.ApiKey = $env:ANTHROPIC_API_KEY
    $config.BaseUrl = "https://api.anthropic.com/v1/messages"
    $config.Model = if ($env:ANTHROPIC_MODEL) { $env:ANTHROPIC_MODEL } else { "claude-sonnet-4-5-20250929" }
    Write-Info "Found ANTHROPIC_API_KEY in environment (Priority: HIGHEST)"
    return $config
  }
  
  if ($env:OPENAI_API_KEY) {
    $config.HasApiKey = $true
    $config.Provider = "openai"
    $config.ApiKey = $env:OPENAI_API_KEY
    $config.BaseUrl = if ($env:OPENAI_BASE_URL) { $env:OPENAI_BASE_URL } else { "https://api.openai.com/v1" }
    $config.Model = if ($env:OPENAI_MODEL) { $env:OPENAI_MODEL } else { "gpt-4o" }
    Write-Info "Found OPENAI_API_KEY in environment (Priority: FALLBACK)"
    return $config
  }
  
  # Check .env file if no environment variables - ANTHROPIC has priority
  if (Test-Path $envFile) {
    Write-Info "Checking .env file for external API configuration..."
    $envContent = Get-Content $envFile
    
    $anthropicKey = $null
    $anthropicModel = "claude-sonnet-4-5-20250929"
    $openaiKey = $null
    $openaiBase = "https://api.openai.com/v1"
    $openaiModel = "gpt-4o"
    
    foreach ($line in $envContent) {
      # Check for Anthropic configuration (PRIORITY)
      if ($line -match '^ANTHROPIC_API_KEY=(.+)$' -and -not $line.StartsWith('#')) {
        $key = $matches[1].Trim()
        if ($key -and $key -ne 'your_anthropic_api_key_here') {
          $anthropicKey = $key
        }
      }
      if ($line -match '^ANTHROPIC_MODEL=(.+)$' -and -not $line.StartsWith('#')) {
        $anthropicModel = $matches[1].Trim()
      }
      
      # Check for OpenAI configuration (FALLBACK)
      if ($line -match '^OPENAI_API_KEY=(.+)$' -and -not $line.StartsWith('#')) {
        $key = $matches[1].Trim()
        if ($key -and $key -ne 'your_openai_api_key_here' -and $key -ne 'key') {
          $openaiKey = $key
        }
      }
      if ($line -match '^OPENAI_BASE_URL=(.+)$' -and -not $line.StartsWith('#')) {
        $openaiBase = $matches[1].Trim()
      }
      if ($line -match '^OPENAI_MODEL=(.+)$' -and -not $line.StartsWith('#')) {
        $openaiModel = $matches[1].Trim()
      }
    }
    
    # Apply priority: Anthropic > OpenAI
    if ($anthropicKey) {
      $config.HasApiKey = $true
      $config.Provider = "anthropic"
      $config.ApiKey = $anthropicKey
      $config.BaseUrl = "https://api.anthropic.com/v1/messages"
      $config.Model = $anthropicModel
      Write-Info "Found ANTHROPIC_API_KEY in .env file (Priority: HIGHEST)"
    }
    elseif ($openaiKey) {
      $config.HasApiKey = $true
      $config.Provider = "openai"
      $config.ApiKey = $openaiKey
      $config.BaseUrl = $openaiBase
      $config.Model = $openaiModel
      Write-Info "Found OPENAI_API_KEY in .env file (Priority: FALLBACK)"
    }
  }
  
  return $config
}

# Function to test if external API is working
function Test-ExternalApi($config) {
  if (-not $config.HasApiKey) { return $false }
  
  Write-Info "Testing $($config.Provider.ToUpper()) API connectivity..."
  try {
    if ($config.Provider -eq "anthropic") {
      # Anthropic API test
      $headers = @{
        "x-api-key" = $config.ApiKey
        "anthropic-version" = "2023-06-01"
        "Content-Type" = "application/json"
      }
      
      $testPayload = @{
        model = $config.Model
        messages = @(@{ role = "user"; content = "Test" })
        max_tokens = 10
      } | ConvertTo-Json -Depth 4
      
      $response = Invoke-WebRequest -Uri $config.BaseUrl -Method Post -Body $testPayload -Headers $headers -TimeoutSec 15 -UseBasicParsing
      return $response.StatusCode -eq 200
    }
    else {
      # OpenAI API test
      $headers = @{
        "Authorization" = "Bearer $($config.ApiKey)"
        "Content-Type" = "application/json"
      }
      
      $testPayload = @{
        model = $config.Model
        messages = @(@{ role = "user"; content = "Test connection" })
        max_tokens = 10
        temperature = 0.1
      } | ConvertTo-Json -Depth 4
      
      $response = Invoke-WebRequest -Uri "$($config.BaseUrl.TrimEnd('/'))/chat/completions" -Method Post -Body $testPayload -Headers $headers -TimeoutSec 15 -UseBasicParsing
      return $response.StatusCode -eq 200
    }
  } catch {
    Write-Warn "$($config.Provider.ToUpper()) API test failed: $_"
    return $false
  }
}

# Main configuration detection
$externalConfig = Get-ExternalApiConfig
$useExternal = $false

if ($ForceExternal) {
  Write-Info "Force external API mode requested"
  $useExternal = $true
} elseif ($ForceLocal) {
  Write-Info "Force local Ollama mode requested"
  $useExternal = $false
} else {
  # Auto-detect based on configuration with PRIORITY: Anthropic > OpenAI > Local
  if ($externalConfig.HasApiKey) {
    Write-Info "External API configuration detected:"
    Write-Info "  Provider: $($externalConfig.Provider.ToUpper())"
    Write-Info "  Base URL: $($externalConfig.BaseUrl)"
    Write-Info "  Model: $($externalConfig.Model)"
    if ($externalConfig.Provider -eq "anthropic") {
      Write-Info "  Priority: ANTHROPIC (HIGHEST - Recommended)"
    } else {
      Write-Info "  Priority: OPENAI (FALLBACK)"
    }
    
    if (Test-ExternalApi $externalConfig) {
      Write-Success "$($externalConfig.Provider.ToUpper()) API test successful - using external API"
      $useExternal = $true
    } else {
      Write-Warn "$($externalConfig.Provider.ToUpper()) API test failed - but API key is configured, so using external mode anyway"
      Write-Warn "The system will attempt to use external API and fall back to other options if needed"
      $useExternal = $true
    }
  } else {
    Write-Info "No external API configuration found - using local Ollama"
    Write-Info "  Priority: LOCAL OLLAMA (no API key)"
    $useExternal = $false
  }
}

# Launch appropriate mode
if ($useExternal) {
  Write-Info "=" * 60
  Write-Success "STARTING LATTICE WITH EXTERNAL API ($($externalConfig.Provider.ToUpper()))"
  Write-Info "=" * 60
  Write-Info "Provider: $($externalConfig.Provider.ToUpper())"
  Write-Info "API: $($externalConfig.BaseUrl)"
  Write-Info "Model: $($externalConfig.Model)"
  Write-Info ""
  
  # Set environment variables for external API based on provider
  if ($externalConfig.Provider -eq "anthropic") {
    $env:ANTHROPIC_API_KEY = $externalConfig.ApiKey
    $env:ANTHROPIC_MODEL = $externalConfig.Model
    # Clear OpenAI variables to ensure Anthropic is used
    $env:OPENAI_API_KEY = $null
    $env:OPENAI_BASE_URL = $null
    $env:OPENAI_MODEL = $null
  } else {
    $env:OPENAI_API_KEY = $externalConfig.ApiKey
    $env:OPENAI_BASE_URL = $externalConfig.BaseUrl
    $env:OPENAI_MODEL = $externalConfig.Model
    # Clear Anthropic variables to ensure OpenAI is used
    $env:ANTHROPIC_API_KEY = $null
    $env:ANTHROPIC_MODEL = $null
  }
  
  # Clear any Ollama-specific variables that might interfere
  $env:LLM_BACKEND = $null
  $env:OLLAMA_MODEL = $null
  
  # Ensure Lattice uses its default port (8080), not Ollama's port (11434)
  if (-not $env:LATTICE_PORT) {
    $env:LATTICE_PORT = "8080"
  }
  
  Write-Info "Starting Lattice Service with external API..."
  Write-Info "Lattice will run on port: $($env:LATTICE_PORT)"
  $latticeProcess = Start-Process -FilePath "python" -ArgumentList "lattice_service.py" -NoNewWindow -PassThru
  
} else {
  Write-Info "=" * 60
  Write-Success "STARTING LATTICE WITH LOCAL OLLAMA"
  Write-Info "=" * 60
  Write-Info "Model Directory: $ModelDir"
  Write-Info "Context Size: $NumCtx"
  Write-Info "GPU Count: $NumGpu"
  Write-Info ""
  
  # Call the original Ollama script logic
  Write-Info "Delegating to Ollama launcher..."
  
  # Clear external API variables to ensure local mode
  $env:ANTHROPIC_API_KEY = $null
  $env:ANTHROPIC_MODEL = $null
  $env:OPENAI_API_KEY = $null
  $env:OPENAI_BASE_URL = $null
  $env:OPENAI_MODEL = $null
  
  # Import the Ollama functions and run
  $ollamaScript = Join-Path $ScriptDir "start_lattice_with_ollama.ps1"
  if (Test-Path $ollamaScript) {
    Write-Info "Running Ollama setup script..."
    & $ollamaScript -ModelDir $ModelDir -NumCtx $NumCtx -NumGpu $NumGpu -OllamaExe $OllamaExe
    return
  } else {
    Write-Err "Ollama script not found: $ollamaScript"
    exit 1
  }
}

# Common health check and monitoring for external API mode
if ($useExternal) {
  # Wait for Lattice to start
  Write-Info "Waiting for Lattice Service to initialize..."
  Start-Sleep -Seconds 8
  
  # Test Lattice health
  function Test-LatticeHealth {
    $port = $env:LATTICE_PORT
    if (-not $port) { $port = "8080" }
    try {
      $resp = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:${port}/health" -TimeoutSec 10
      return $resp.StatusCode -eq 200
    } catch { return $false }
  }
  
  $healthReady = $false
  Write-Info "Waiting for Lattice health endpoint..."
  for ($i = 0; $i -lt 30; $i++) {
    if (Test-LatticeHealth) {
      $healthReady = $true
      Write-Success "Lattice health endpoint is responding"
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
  
  $port = $env:LATTICE_PORT
  if (-not $port) { $port = "8080" }
  
  Write-Success "Lattice Service is running with external API!"
  Write-Info "Health check: http://127.0.0.1:${port}/health"
  Write-Info "Chat endpoint: http://127.0.0.1:${port}/v1/chat/completions"
  Write-Info ""
  Write-Info "External API Details:"
  Write-Info "  Provider: $($externalConfig.Provider.ToUpper())"
  Write-Info "  Endpoint: $($externalConfig.BaseUrl)"
  Write-Info "  Model: $($externalConfig.Model)"
  Write-Info ""
  
  # Start the dashboard automatically
  Write-Info ("=" * 60)
  Write-Info "Starting Daemon Dashboard..."
  Write-Info ("=" * 60)
  
  $dashboardDir = Join-Path $RepoRoot "daemon-dashboard"
  if (Test-Path $dashboardDir) {
    # Check if node_modules exists, if not run npm install
    $nodeModules = Join-Path $dashboardDir "node_modules"
    if (-not (Test-Path $nodeModules)) {
      Write-Info "Installing dashboard dependencies..."
      Push-Location $dashboardDir
      try {
        npm install 2>&1 | Out-Null
        Write-Success "Dashboard dependencies installed"
      } catch {
        Write-Warn "Failed to install dashboard dependencies: $_"
      }
      Pop-Location
    }
    
    # Start the dashboard
    Write-Info "Launching dashboard at http://localhost:3000"
    
    # Create a background job to run npm (works reliably on Windows)
    $dashboardJob = Start-Job -ScriptBlock {
      param($dir)
      Set-Location $dir
      npm run dev
    } -ArgumentList $dashboardDir
    
    # Store the job for cleanup
    $global:dashboardJob = $dashboardJob
    
    # Wait a moment for dashboard to start
    Start-Sleep -Seconds 3
    
    # Open browser automatically
    Write-Info "Opening dashboard in browser..."
    Start-Process "http://localhost:3000"
    
    Write-Success "Dashboard started successfully!"
    Write-Info ""
    Write-Info ("=" * 60)
    Write-Success "ALL SERVICES RUNNING"
    Write-Info ("=" * 60)
    Write-Info "  Lattice API: http://127.0.0.1:${port}"
    Write-Info "  Dashboard: http://localhost:3000"
    Write-Info ""
    Write-Info "Press Ctrl+C to stop all services..."
    
    # Cleanup function
    $cleanup = {
      Write-Info ""
      Write-Info "Shutting down services..."
      if ($global:dashboardJob) {
        Write-Info "Stopping dashboard..."
        Stop-Job -Job $global:dashboardJob -ErrorAction SilentlyContinue
        Remove-Job -Job $global:dashboardJob -Force -ErrorAction SilentlyContinue
        # Also kill any node processes on port 3000
        $nodeProcesses = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
        if ($nodeProcesses) {
          $nodeProcesses | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
        }
      }
      if ($latticeProcess -and -not $latticeProcess.HasExited) {
        Write-Info "Stopping Lattice service..."
        Stop-Process -Id $latticeProcess.Id -Force -ErrorAction SilentlyContinue
      }
      Write-Info "All services stopped."
    }
    
    # Register cleanup on Ctrl+C
    Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action $cleanup | Out-Null
    
    # Keep the processes running
    try {
      while ($true) {
        # Check if Lattice is still running
        if ($latticeProcess.HasExited) {
          Write-Err "Lattice process has exited unexpectedly"
          & $cleanup
          exit 1
        }
        # Check if dashboard job is still running
        if ($global:dashboardJob -and $global:dashboardJob.State -eq 'Failed') {
          Write-Warn "Dashboard job failed, but Lattice is still running"
        }
        Start-Sleep -Seconds 5
      }
    } catch {
      & $cleanup
    } finally {
      & $cleanup
    }
  } else {
    Write-Warn "Dashboard directory not found at: $dashboardDir"
    Write-Info "Lattice is running, but dashboard could not be started"
    Write-Info "Dashboard: Manual start with 'cd daemon-dashboard && npm run dev'"
    Write-Info ""
    Write-Info "Press Ctrl+C to stop..."
    
    # Keep the process running
    if ($latticeProcess -and -not $latticeProcess.HasExited) {
      try {
        $latticeProcess.WaitForExit()
      } catch {
        Write-Info "Shutting down Lattice Service..."
      }
    } else {
      Write-Err "Lattice process has unexpectedly exited"
      exit 1
    }
  }
}