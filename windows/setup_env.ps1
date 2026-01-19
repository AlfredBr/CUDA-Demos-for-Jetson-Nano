# CUDA Development Environment Setup Script for PowerShell
# Run this script to set up the MSVC environment for CUDA development
#
# Usage: . .\setup_env.ps1   (note the dot-space at the beginning to "source" it)
#
# After running, you can use 'nmake' directly

Write-Host "========================================"
Write-Host "CUDA Development Environment Setup"
Write-Host "========================================"
Write-Host ""

# Find Visual Studio installation
$vcvarsall = $null
$vsName = $null

$vsLocations = @(
    @{ Path = "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat"; Name = "Visual Studio 2025 Insiders" },
    @{ Path = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"; Name = "Visual Studio 2022 Community" },
    @{ Path = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"; Name = "Visual Studio 2022 Professional" },
    @{ Path = "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"; Name = "Visual Studio 2022 Enterprise" },
    @{ Path = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"; Name = "Visual Studio 2022 Build Tools" },
    @{ Path = "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat"; Name = "Visual Studio 2019 Community" }
)

foreach ($vs in $vsLocations) {
    if (Test-Path $vs.Path) {
        $vcvarsall = $vs.Path
        $vsName = $vs.Name
        Write-Host "Found: $vsName" -ForegroundColor Green
        break
    }
}

if (-not $vcvarsall) {
    Write-Host "ERROR: Could not find Visual Studio installation." -ForegroundColor Red
    Write-Host ""
    Write-Host "CUDA on Windows requires MSVC as the host compiler."
    Write-Host "Please install Visual Studio 2019, 2022, or later with the"
    Write-Host "'Desktop development with C++' workload."
    return
}

Write-Host ""
Write-Host "Setting up x64 build environment..."

# Run vcvarsall.bat and capture environment variables
$envBefore = @{}
Get-ChildItem Env: | ForEach-Object { $envBefore[$_.Name] = $_.Value }

# Execute vcvarsall and capture the resulting environment
$result = cmd /c "`"$vcvarsall`" x64 && set" 2>$null

# Check if we got results (vcvarsall succeeded if we have environment output)
if (-not $result -or $result.Count -eq 0) {
    Write-Host "ERROR: Failed to set up MSVC environment." -ForegroundColor Red
    return
}

# Parse and apply the environment variables
foreach ($line in $result) {
    if ($line -match '^([^=]+)=(.*)$') {
        $name = $matches[1]
        $value = $matches[2]

        # Only set if it changed or is new
        if (-not $envBefore.ContainsKey($name) -or $envBefore[$name] -ne $value) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

# Create nmake wrapper function with /nologo
# First clear any leftover MAKEFLAGS that might cause issues
Remove-Item Env:\MAKEFLAGS -ErrorAction SilentlyContinue
Remove-Item -Path Function:\nmake -ErrorAction SilentlyContinue
function global:nmake {
    $nmakePath = (Get-Command nmake.exe -ErrorAction SilentlyContinue).Source
    if ($nmakePath) {
        & $nmakePath /nologo @args
    } else {
        Write-Host "ERROR: nmake.exe not found. Run setup_env.ps1 first." -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================"
Write-Host "Environment configured successfully!" -ForegroundColor Green
Write-Host "========================================"
Write-Host ""
Write-Host "You can now use:"
Write-Host "  nmake                 - Build all demos"
Write-Host "  nmake cuda_flame      - Build specific demo"
Write-Host "  nmake clean           - Clean build artifacts"
Write-Host ""
Write-Host "CUDA compiler:"
nvcc --version | Select-String "release"
Write-Host ""
