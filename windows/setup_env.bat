@echo off
REM CUDA Development Environment Setup Script
REM Run this script to set up the MSVC environment for CUDA development
REM
REM Usage: setup_env.bat
REM After running, you can use 'make' or 'nmake' directly
REM
REM NOTE: Run this script using 'call setup_env.bat' or open a new cmd window
REM       and run it there. Running from PowerShell requires special handling.

echo ========================================
echo CUDA Development Environment Setup
echo ========================================
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
goto :eof

:found_vs
echo.
echo Setting up x64 build environment...
call "%VCVARSALL%" x64

if errorlevel 1 (
    echo ERROR: Failed to set up MSVC environment.
    goto :eof
)

REM Suppress nmake banner for all calls
set MAKEFLAGS=nologo

echo.
echo ========================================
echo Environment configured successfully!
echo ========================================
echo.
echo You can now use:
echo   nmake                 - Build all demos
echo   nmake cuda_flame      - Build specific demo
echo   nmake clean           - Clean build artifacts
echo.
echo CUDA compiler:
nvcc --version | findstr /i "release"
echo.
