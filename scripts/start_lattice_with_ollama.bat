@echo off
setlocal

echo NOTICE: This script is designed for local Ollama models only.
echo.
echo If you have configured external APIs (OPENAI_API_KEY in .env), 
echo this script may conflict with your configuration.
echo.
echo For automatic configuration detection, use:
echo   start_lattice_smart.bat
echo.
echo To force local Ollama mode (ignoring external API config):
echo   start_lattice_smart.bat -ForceLocal
echo.
echo Press any key to continue with Ollama-only mode, or Ctrl+C to cancel...
pause

REM Determine the directory of this script
set "SCRIPT_DIR=%~dp0"

echo Starting Ollama-specific launcher...

REM Launch the PowerShell starter with all passed arguments in background
start "Lattice Service" powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start_lattice_with_ollama.ps1" %*

endlocal
