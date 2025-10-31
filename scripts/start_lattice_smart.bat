@echo off
setlocal

REM Smart Lattice Launcher - Automatically detects external API or local Ollama configuration

REM Determine the directory of this script
set "SCRIPT_DIR=%~dp0"

echo Starting Smart Lattice Launcher...
echo This will automatically detect your configuration and start appropriately.
echo.

REM Launch the PowerShell smart launcher with all passed arguments
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start_lattice_smart.ps1" %*

endlocal
pause