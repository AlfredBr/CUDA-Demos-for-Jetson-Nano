@echo off
REM Windows CUDA Demos Build Script
REM Sets up MSVC environment and runs make
REM
REM Usage: build.bat [target]
REM   build.bat          - Build all demos
REM   build.bat clean    - Clean build artifacts
REM   build.bat cuda_XXX - Build specific demo

setlocal EnableDelayedExpansion

echo ====================================
echo CUDA Graphics Demos - Windows Build
echo ====================================
echo.

REM Try to find Visual Studio installation
set "VCVARSALL="

REM Check Visual Studio 2025 Insiders
if exist "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARSALL=C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat"
    echo Found: Visual Studio 2025 Insiders
    goto :found_vs
)

REM Check Visual Studio 2022 Community
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARSALL=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
    echo Found: Visual Studio 2022 Community
    goto :found_vs
)

REM Check Visual Studio 2022 Professional
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARSALL=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
    echo Found: Visual Studio 2022 Professional
    goto :found_vs
)

REM Check Visual Studio 2022 Enterprise
if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARSALL=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
    echo Found: Visual Studio 2022 Enterprise
    goto :found_vs
)

REM Check Visual Studio 2022 Build Tools
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARSALL=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
    echo Found: Visual Studio 2022 Build Tools
    goto :found_vs
)

REM Check Visual Studio 2019
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VCVARSALL=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat"
    echo Found: Visual Studio 2019 Community
    goto :found_vs
)

echo ERROR: Could not find Visual Studio installation.
echo.
echo CUDA on Windows requires MSVC as the host compiler.
echo Please install Visual Studio 2019, 2022, or later with the
echo "Desktop development with C++" workload.
echo.
echo You can also install just the Build Tools from:
echo https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
exit /b 1

:found_vs
echo.
echo Setting up x64 build environment...
call "%VCVARSALL%" x64 >nul 2>&1

if errorlevel 1 (
    echo ERROR: Failed to set up MSVC environment.
    exit /b 1
)

echo Environment configured successfully.
echo.

REM Check for CUDA
where nvcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvcc not found in PATH.
    echo Please install CUDA Toolkit and ensure it's in your PATH.
    exit /b 1
)

echo CUDA compiler found:
nvcc --version | findstr /C:"release"
echo.

REM Change to the script's directory
cd /d "%~dp0"

REM Run make with the provided target (or 'all' by default)
if "%~1"=="" (
    echo Building all demos...
    echo.
    nmake /nologo /f Makefile all
) else (
    echo Building target: %~1
    echo.
    nmake /nologo /f Makefile %~1
)

if errorlevel 1 (
    echo.
    echo Build failed!
    exit /b 1
)

echo.
echo Build complete!
