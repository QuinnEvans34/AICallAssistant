$ErrorActionPreference = "Stop"

Write-Host "[cpp_asr] Checking for Visual Studio Build Tools..."
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found. Install Visual Studio 2022 or Build Tools."
}

$vsInstall = & $vswhere -latest -requires Microsoft.Component.MSBuild -property installationPath
if (-not $vsInstall) {
    throw "Visual Studio Build Tools were not detected. Install VS 2022 with the Desktop development with C++ workload."
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceDir = Join-Path $repoRoot "cpp_asr"
$buildDir = Join-Path $sourceDir "build"

# Resolve Python interpreter (prefer repo-local venv)
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}
Write-Host "[cpp_asr] Using Python interpreter: $pythonExe"

try {
    $pybindDir = (& $pythonExe -c "import pybind11; print(pybind11.get_cmake_dir())").Trim()
    if (-not $pybindDir) {
        throw "pybind11.get_cmake_dir returned empty path"
    }
} catch {
    throw "pybind11 not found for interpreter '$pythonExe'. Run 'pip install pybind11' inside your environment."
}

Write-Host "[cpp_asr] Configuring project..."
& cmake `
    -S $sourceDir `
    -B $buildDir `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DCPP_ASR_ENABLE_PYBIND11=ON `
    "-DPython3_EXECUTABLE=$pythonExe" `
    "-Dpybind11_DIR=$pybindDir"

Write-Host "[cpp_asr] Building Release DLL..."
& cmake --build $buildDir --config Release

$dllPath = Join-Path $buildDir "Release\cpp_asr.dll"
$pydPath = Join-Path $buildDir "Release\cpp_asr_native.pyd"
if (Test-Path $dllPath) {
    Write-Host "[cpp_asr] Build succeeded: $dllPath"
} else {
    throw "Build completed but cpp_asr.dll was not found at $dllPath"
}

if (Test-Path $pydPath) {
    Write-Host "[cpp_asr] PyBind11 module ready: $pydPath"
} else {
    throw "Build completed but cpp_asr_native.pyd was not found at $pydPath"
}
