param(
    [switch]$SkipSubmodules,
    [switch]$SkipBuild,
    [switch]$SkipServer,
    [switch]$FakeMode,
    [string]$ModelPath,
    [switch]$StartUI,
    [switch]$SkipHealth,
    [switch]$Verbose
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Write-Host "[INFO] Repo root: $repoRoot"
Set-Location $repoRoot

function Info($m){ Write-Host "[INFO] $m" }
function Warn($m){ Write-Warning $m }
function Fail($m){ Write-Host "[ERROR] $m"; exit 1 }
function Verb($m){ if($Verbose){ Write-Host "[VERBOSE] $m" } }

# Resolve python (avoid PowerShell 7 '??')
$python = Get-Command python -ErrorAction SilentlyContinue
if(-not $python){ $python = Get-Command py -ErrorAction SilentlyContinue }
if(-not $python){ Fail "Python not found" }

# Tool checks
if(-not (Get-Command cmake -ErrorAction SilentlyContinue)) { Fail "cmake not in PATH" }
if(-not (Get-Command git -ErrorAction SilentlyContinue)) { Warn "git not in PATH; skipping submodules"; $SkipSubmodules = $true }
if(-not (Get-Command dotnet -ErrorAction SilentlyContinue)) { Warn "dotnet not found; UI launch disabled"; $StartUI = $false }

# venv
if(-not (Test-Path "$repoRoot/.venv/Scripts/python.exe")){
    Info "Creating virtual environment (.venv)"
    & $python.Source -m venv .venv
}
$venvPy = "$repoRoot/.venv/Scripts/python.exe"
if(-not (Test-Path $venvPy)){ Fail "Virtualenv python missing: $venvPy" }

# Dependencies
Info "Installing Python dependencies"
& $venvPy -m pip install --upgrade pip > $null
$reqArgs = @()
if(Test-Path 'requirements.txt'){ $reqArgs += '-r','requirements.txt' } else { Warn 'Missing requirements.txt' }
if(Test-Path 'backend_server/requirements.txt'){ $reqArgs += '-r','backend_server/requirements.txt' } else { Warn 'Missing backend_server/requirements.txt' }
$reqArgs += 'pybind11'
& $venvPy -m pip install @reqArgs

# pybind11 CMake path (use -c not heredoc)
$pybindCMakePath = (& $venvPy -c "import pybind11, pathlib; p=pathlib.Path(pybind11.__file__).parent/'share'/'cmake'/'pybind11'; print(p if p.exists() else '')").Trim()
if($pybindCMakePath){ Verb "Detected pybind11 CMake config: $pybindCMakePath" } else { Warn "pybind11 CMake config not found; find_package(pybind11) may fail" }

# Submodules
if(-not $SkipSubmodules){
    Info "Updating submodules"
    try { git submodule update --init --recursive } catch { Warn "Submodule update failed: $_" }
}

# Model selection
$modelsDir = Join-Path $repoRoot 'models'
if($ModelPath){
    $env:ASR_MODEL_PATH = (Resolve-Path $ModelPath).Path
    Info "Using explicit model: $env:ASR_MODEL_PATH"
} else {
    $modelFile = Get-ChildItem -Path $modelsDir -Include *.gguf,*.ggml,*.bin -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if($modelFile){ $env:ASR_MODEL_PATH = $modelFile.FullName; Info "Auto-selected model: $env:ASR_MODEL_PATH" }
    elseif($FakeMode){ $env:ASR_ALLOW_FAKE = '1'; Warn "No model found; fake mode enabled" }
    else { Warn "No model file found. Provide -ModelPath or use -FakeMode to continue." }
}

# Build C++
$buildDir = Join-Path $repoRoot 'cpp_asr' 'build'
if(-not $SkipBuild){
    if(-not (Test-Path $buildDir)){ New-Item -ItemType Directory -Path $buildDir | Out-Null }
    Info "Configuring CMake"
    $pyRoot = (Resolve-Path "$repoRoot/.venv").Path
    $cmakeArgs = @('-B','cpp_asr/build','-S','cpp_asr','-DCPP_ASR_ENABLE_PYBIND11=ON','-DCMAKE_BUILD_TYPE=Release',"-DPython3_ROOT_DIR=$pyRoot","-DPython3_EXECUTABLE=$venvPy")
    if($pybindCMakePath){ $cmakeArgs += "-DCMAKE_PREFIX_PATH=$pybindCMakePath" }
    Verb "cmake args: $($cmakeArgs -join ' ')"
    try { cmake @cmakeArgs } catch { Fail "CMake configure failed: $_" }
    Info "Building (Release)"
    try { cmake --build cpp_asr/build --config Release -- /m } catch { Fail "CMake build failed: $_" }
} else { Info "Skipping build" }

# Native module presence
$nativePyd = Get-ChildItem -Path $buildDir -Filter cpp_asr_native*.pyd -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if($nativePyd){ Info "Found native module: $($nativePyd.FullName)" } else { Warn "Native module (.pyd) not found" }

# Verification
Info "Verifying native module"
& $venvPy scripts/verify_cpp_module.py
if($LASTEXITCODE -ne 0){ Warn "Verification returned non-zero exit code ($LASTEXITCODE)" }

# Server start
if(-not $SkipServer){
    $logsDir = Join-Path $repoRoot 'logs'
    if(-not (Test-Path $logsDir)){ New-Item -ItemType Directory -Path $logsDir | Out-Null }
    $env:PYTHONPATH = "$buildDir/Release;$env:PYTHONPATH"
    Info "Launching backend on :8000"
    $outLog = Join-Path $logsDir 'backend_out.log'
    $errLog = Join-Path $logsDir 'backend_err.log'
    try {
        Start-Process -FilePath $venvPy -ArgumentList '-u','-m','uvicorn','backend_server.main_server:app','--host','0.0.0.0','--port','8000' -RedirectStandardOutput $outLog -RedirectStandardError $errLog -WorkingDirectory $repoRoot
    } catch { Fail "Failed to start backend: $_" }
    Info "Logs: $outLog / $errLog"
    if(-not $SkipHealth){
        Info "Waiting for health endpoint"
        $healthy = $false
        for($i=0;$i -lt 30 -and -not $healthy;$i++){
            Start-Sleep -Milliseconds 500
            try { $resp = Invoke-RestMethod -Uri 'http://localhost:8000/health' -TimeoutSec 2; if($resp.status -eq 'ok'){ $healthy = $true } } catch { Verb "Health attempt $i failed" }
        }
        if($healthy){ Info "Backend healthy" } else { Warn "Backend health check failed; inspect backend_err.log" }
    }
} else { Info "SkipServer selected; backend not started" }

# UI launch
if($StartUI){
    $uiProj = Join-Path $repoRoot 'CallAssistantUI' 'CallAssistantUI.csproj'
    if(Test-Path $uiProj){
        Info "Starting UI (dotnet run)"
        try { Start-Process -FilePath 'dotnet' -ArgumentList 'run','--project',$uiProj -WorkingDirectory $repoRoot } catch { Warn "Failed to start UI: $_" }
    } else { Warn "UI project not found: $uiProj" }
} else { Verb "UI launch skipped" }

Info "Done."