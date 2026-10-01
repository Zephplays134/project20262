param(
    [switch]$IsElevated = $false,
    # Skip the admin self-elevation. Everything below installs into a local
    # .venv, which never needs Administrator. Kept as an opt-in escape hatch for
    # the rare machine where the Python/Node installers themselves require it.
    [switch]$Elevate
)

# 1. Self-Elevate to Admin (only when explicitly asked)
if ($Elevate -and -not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Elevating privileges for installation..."
    $arguments = "& '" + $myinvocation.mycommand.definition + "' -Elevate"
    Start-Process powershell -Verb runAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -Command $arguments"
    exit
}

$ErrorActionPreference = "Stop"
$WorkingDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $WorkingDir

Write-Host "==========================================================================" -ForegroundColor Yellow
Write-Host "  ____  ____      _    _   _ __  __    _      _    ___ " -ForegroundColor Yellow
Write-Host " | __ )|  _ \    / \  | | | |  \/  |  / \    / \  |_ _|" -ForegroundColor Yellow
Write-Host " |  _ \| |_) |  / _ \ | |_| | |\/| | / _ \  / _ \  | | " -ForegroundColor Yellow
Write-Host " | |_) |  _ <  / ___ \|  _  | |  | |/ ___ \/ ___ \ | | " -ForegroundColor Yellow
Write-Host " |____/|_| \_\/_/   \_\_| |_|_|  |_/_/   \_\_/   \_\___|" -ForegroundColor Yellow
Write-Host "                      BRAHMA EVO" -ForegroundColor Cyan
Write-Host "         AUTONOMOUS SELF-EVOLUTION COGNITIVE ENGINE" -ForegroundColor Green
Write-Host "   [Skill Forge // Crucible Sandbox // 180 FPS HoloCore]" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Yellow
Write-Host ""

# 2. Helper to refresh environment variables
function Update-Environment {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# 3. Check for Python
$PythonExe = $null
if (Get-Command "py" -ErrorAction SilentlyContinue) { $PythonExe = "py" }
elseif (Get-Command "python" -ErrorAction SilentlyContinue) { $PythonExe = "python" }

if (-not $PythonExe) {
    Write-Host "Python not found. Downloading Python 3.13.9..." -ForegroundColor Yellow
    $PythonUrl = "https://www.python.org/ftp/python/3.13.9/python-3.13.9-amd64.exe"
    $PythonInstaller = "$env:TEMP\python_installer.exe"
    Invoke-WebRequest -Uri $PythonUrl -OutFile $PythonInstaller

    Write-Host "Installing Python (Silent Mode)..." -ForegroundColor Yellow
    # Per-user rather than InstallAllUsers=1: this no longer needs Administrator,
    # so the installer cannot fail on a locked-down machine or prompt for rights
    # the user did not intend to grant.
    Start-Process -FilePath $PythonInstaller -ArgumentList "/quiet PrependPath=1 Include_test=0 Include_launcher=1" -Wait

    Write-Host "Python installed successfully." -ForegroundColor Green
    Update-Environment
    $PythonExe = "python"
} else {
    Write-Host "Python is already installed: $(Get-Command $PythonExe | Select-Object -ExpandProperty Source)" -ForegroundColor Green
    $FoundVersion = (& $PythonExe -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null)
    if ($FoundVersion) {
        Write-Host "  Detected Python $FoundVersion" -ForegroundColor Green
    } else {
        throw "Found '$PythonExe' but it could not report a version. Install Python 3.11+ and re-run."
    }
}

# 4. Check for Node.js
if (-not (Get-Command "node" -ErrorAction SilentlyContinue)) {
    Write-Host "Node.js not found. Downloading Node v20 LTS..." -ForegroundColor Yellow
    $NodeUrl = "https://nodejs.org/dist/v20.11.1/node-v20.11.1-x64.msi"
    $NodeInstaller = "$env:TEMP\node_installer.msi"
    Invoke-WebRequest -Uri $NodeUrl -OutFile $NodeInstaller
    
    Write-Host "Installing Node.js (Silent Mode)..." -ForegroundColor Yellow
    Start-Process -FilePath "msiexec.exe" -ArgumentList "/i `"$NodeInstaller`" /qn" -Wait
    
    Write-Host "Node.js installed successfully." -ForegroundColor Green
    Update-Environment
} else {
    Write-Host "Node.js is already installed: $(Get-Command node | Select-Object -ExpandProperty Source)" -ForegroundColor Green
}

# 5. Virtual Environment Setup
$VenvDir = Join-Path -Path $WorkingDir -ChildPath ".venv"
$VenvPython = Join-Path -Path $VenvDir -ChildPath "Scripts\python.exe"
$VenvPythonW = Join-Path -Path $VenvDir -ChildPath "Scripts\pythonw.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating Virtual Environment in .venv..." -ForegroundColor Cyan
    if (Test-Path $VenvDir) { Remove-Item -Recurse -Force $VenvDir }
    Invoke-Step -FilePath $PythonExe -Arguments @("-m", "venv", ".venv") -What "Creating virtual environment"
    Write-Host "Virtual Environment created." -ForegroundColor Green
} else {
    Write-Host "Virtual Environment already exists." -ForegroundColor Green
}

# Helper: run a command, surface its exit code, and abort the bootstrap if it fails.
function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$What
    )
    Write-Host "  -> $What" -ForegroundColor Cyan
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        # `Start-Process -Wait -NoNewWindow` was used here previously, which
        # discards the child's exit code: a failed `pip install` printed an error
        # and then the bootstrap carried on as if it had worked, leaving a
        # half-installed venv that only failed much later at runtime. Calling the
        # executable directly lets us check $LASTEXITCODE and stop early.
        throw "$What failed with exit code $LASTEXITCODE. Bootstrap aborted."
    }
    Write-Host "  -> $What OK" -ForegroundColor Green
}

# 6. Install Dependencies
Write-Host "Updating pip and installing dependencies..." -ForegroundColor Cyan
Invoke-Step -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel") -What "Upgrading pip"
Invoke-Step -FilePath $VenvPython -Arguments @("-m", "pip", "install", "-r", "requirements.txt") -What "Installing requirements.txt"

Write-Host "Installing Playwright browsers..." -ForegroundColor Cyan
Invoke-Step -FilePath $VenvPython -Arguments @("-m", "playwright", "install") -What "Installing Playwright browsers"

# 7. Launch App
Write-Host "Starting Brahma AI..." -ForegroundColor Green
if (Test-Path $VenvPythonW) {
    Start-Process -FilePath $VenvPythonW -ArgumentList "main.py --startup" -WorkingDirectory $WorkingDir
} else {
    Start-Process -FilePath $VenvPython -ArgumentList "main.py --startup" -WorkingDirectory $WorkingDir -WindowStyle Hidden
}

Write-Host "Bootstrap complete. You can close this window."
Start-Sleep -Seconds 3
