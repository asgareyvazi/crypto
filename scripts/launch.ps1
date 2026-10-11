# Launch script for ICBT on Windows (PowerShell)
#
# Usage:
#   .\scripts\launch.ps1              # launches the GUI dashboard on http://127.0.0.1:8765
#   .\scripts\launch.ps1 -Cli         # shows CLI help
#   .\scripts\launch.ps1 -Install     # (re)installs the package and dependencies
#
param(
    [switch]$Cli,
    [switch]$Install,
    [int]$Port = 8765,
    [string]$Host = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# --- Locate python --------------------------------------------------------
$PythonCmd = $null
foreach ($candidate in @("python", "py -3", "python3")) {
    try {
        $null = & cmd /c "$candidate --version" 2>&1
        if ($LASTEXITCODE -eq 0) { $PythonCmd = $candidate; break }
    } catch {}
}
if (-not $PythonCmd) {
    Write-Host "ERROR: Python 3.10+ not found on PATH. Install Python from https://www.python.org/ and retry." -ForegroundColor Red
    exit 1
}

# --- Create/upgrade venv on first run -------------------------------------
$VenvDir = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment in $VenvDir ..."
    & cmd /c "$PythonCmd -m venv .venv"
}
$Py = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "ERROR: virtualenv python not found at $Py" -ForegroundColor Red
    exit 1
}

# --- Install dependencies -------------------------------------------------
if ($Install -or -not (Test-Path (Join-Path $VenvDir "Lib\site-packages\icbt"))) {
    Write-Host "Installing ICBT and dependencies..."
    & $Py -m pip install --upgrade pip | Out-Null
    & $Py -m pip install -e "$ProjectRoot[dev]"
}

# --- Dispatch -------------------------------------------------------------
if ($Cli) {
    & $Py -m icbt --help
} else {
    Write-Host "ICBT dashboard starting on http://$Host`:$Port" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop."
    & $Py -m icbt gui --host $Host --port $Port
}
