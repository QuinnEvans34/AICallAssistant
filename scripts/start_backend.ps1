# Robust PowerShell wrapper to launch the backend via batch to avoid PowerShell quoting issues
param()
$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir '..')
$batch = Join-Path $scriptDir 'start_backend.bat'
$logsDir = Join-Path $repoRoot 'logs'
if (-not (Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir | Out-Null }

# Kill any python process that appears to be running uvicorn on port 8000
try {
    $conns = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
    if ($conns) {
        foreach ($c in $conns) {
            if ($c.OwningProcess) {
                try { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue } catch { }
            }
        }
    }
} catch {
    # ignore on systems without Get-NetTCPConnection
}

if (-not (Test-Path $batch)) {
    Write-Error "Launch batch not found: $batch"
    exit 1
}

# Start the batch which runs the venv python and uvicorn, redirecting output
$proc = Start-Process -FilePath $batch -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 3

# Health check
$healthy = $false
for ($i = 0; $i -lt 10; $i++) {
    try {
        $r = Invoke-RestMethod -Uri 'http://localhost:8000/health' -TimeoutSec 2
        if ($r -and $r.status -eq 'ok') { $healthy = $true; break }
    } catch { }
    Start-Sleep -Seconds 1
}

if ($healthy) { Write-Host 'Backend healthy and running.'; exit 0 }
Write-Host 'Backend failed to respond to health check. See logs in' $logsDir
exit 2
