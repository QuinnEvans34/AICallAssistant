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

Write-Host "[cpp_asr] Configuring project..."
& cmake -S $sourceDir -B $buildDir -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release

Write-Host "[cpp_asr] Building Release DLL..."
& cmake --build $buildDir --config Release

$dllPath = Join-Path $buildDir "Release\cpp_asr.dll"
if (Test-Path $dllPath) {
    Write-Host "[cpp_asr] Build succeeded: $dllPath"
} else {
    throw "Build completed but cpp_asr.dll was not found at $dllPath"
}
