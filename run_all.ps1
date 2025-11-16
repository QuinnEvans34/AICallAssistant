# Run backend (in background) and then start the WPF UI
# Usage: Open Developer PowerShell (x64) and run `.\run_all.ps1`

$ErrorActionPreference = 'Stop'
# Ensure script runs from repo root (where this script lives)
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

# Ensure logs dir
$logsDir = Join-Path $repo 'logs'
if (-not (Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }

# Check for ASR model
$modelPath = $env:ASR_MODEL_PATH
if (-not $modelPath -or -not (Test-Path $modelPath)) {
    $modelsDir = Join-Path $repo 'models'
    if (Test-Path $modelsDir) {
        $modelFiles = Get-ChildItem -Path $modelsDir -File -Recurse | Where-Object { $_.Extension -in @('.bin', '.ggml', '.gguf') }
        if ($modelFiles) {
            $modelPath = $modelFiles[0].FullName
            $env:ASR_MODEL_PATH = $modelPath
        }
    }
    if (-not $modelPath) {
        Write-Error "No ASR model found. Set ASR_MODEL_PATH or place a model in models/ directory."
        exit 1
    }
}
Write-Host "Using ASR model: $modelPath"

# Kill any process listening on port 8000 (uvicorn) if possible
try {
    $conns = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
    if ($conns) {
        foreach ($c in $conns) {
            if ($c.OwningProcess) {
                Try { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue } Catch { }
            }
        }
    }
} catch {
    # fallback: kill python processes (best-effort)
    Get-Process -Name python -ErrorAction SilentlyContinue | ForEach-Object { Try { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue } Catch { } }
}

# Start backend via existing batch launcher in a new window (detached)
$startBackend = Join-Path $repo 'scripts\start_backend.bat'
if (-not (Test-Path $startBackend)) { Write-Error "start_backend.bat not found at $startBackend"; exit 1 }

# Prepare cmd argument to set PATH and PYTHONPATH for the new cmd session and then call the batch
$batchPath = $startBackend
$cmdInner = "`"$batchPath`""

Write-Host "Launching backend (visible window)"
$backendProcess = Start-Process -FilePath 'C:\Windows\System32\cmd.exe' -ArgumentList @('/k', $cmdInner) -WorkingDirectory $repo -WindowStyle Normal -PassThru

# Wait for backend health
Write-Host 'Waiting for backend to become healthy (up to 20s)...'
$healthy = $false
for ($i=0; $i -lt 20; $i++) {
    try {
        $r = Invoke-RestMethod -Uri 'http://localhost:8000/health' -TimeoutSec 2
        if ($r -and $r.status -eq 'ok') { $healthy = $true; break }
    } catch { }
    Start-Sleep -Seconds 1
}
if (-not $healthy) {
    Write-Warning 'Backend did not become healthy in time; check logs in ./logs for details.'
    Write-Host '--- backend_out.log ---'
    if (Test-Path "$logsDir\\backend_out.log") { Get-Content "$logsDir\\backend_out.log" -Tail 200 }
    Write-Host '--- backend_err.log ---'
    if (Test-Path "$logsDir\\backend_err.log") { Get-Content "$logsDir\\backend_err.log" -Tail 200 }
    exit 1
} else {
    Write-Host 'Backend healthy.'
}

# Start WPF UI (dotnet run)
Write-Host 'Starting WPF UI (dotnet run --project CallAssistantUI\\CallAssistantUI.csproj)'
$uiProcess = Start-Process -FilePath 'C:\Program Files\dotnet\dotnet.exe' -ArgumentList @('run','--project','CallAssistantUI\\CallAssistantUI.csproj') -WorkingDirectory $repo -PassThru

# Wait for processes to exit
try {
    $backendProcess.WaitForExit()
    $uiProcess.WaitForExit()
} catch {
    # If interrupted, kill processes
    if ($backendProcess -and -not $backendProcess.HasExited) { $backendProcess.Kill() }
    if ($uiProcess -and -not $uiProcess.HasExited) { $uiProcess.Kill() }
}

Write-Host 'All processes have exited.'
