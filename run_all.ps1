# Run backend (in background) and then start the WPF UI
# Usage: Open Developer PowerShell (x64) and run `.un_all.ps1`

$ErrorActionPreference = 'Stop'
# Ensure script runs from repo root (where this script lives)
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

# Ensure logs dir
$logsDir = Join-Path $repo 'logs'
if (-not (Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }

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

# Launch the batch directly (Start-Process will create a new window)
Write-Host "Launching backend: $startBackend"
Start-Process -FilePath $startBackend -WorkingDirectory $repo -WindowStyle Minimized | Out-Null

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
} else {
    Write-Host 'Backend healthy.'
}

# Start WPF UI (dotnet run)
Write-Host 'Starting WPF UI (dotnet run --project CallAssistantUI\CallAssistantUI.csproj)'
Start-Process -FilePath 'dotnet' -ArgumentList @('run','--project','CallAssistantUI\CallAssistantUI.csproj') -WorkingDirectory $repo | Out-Null

Write-Host 'Launched UI. Use logs\backend_out.log and logs\backend_err.log for backend output.'
